"""
Streaming PubChem for pretraining on the whole release.

The cached datasets build all graphs up front and keep them in memory, which does not
scale beyond a few hundred thousand molecules. Here the dataloader workers build the
graphs from the SMILES file while training. The molecules are read in a fixed random
order (a seeded permutation of the release), so a long run starts with the same
molecules as a shorter one. The first molecules of the permutation are held out for
validation.
"""

import os
import pickle
import warnings
from typing import Iterator

import numpy as np
import torch
import tqdm
from chebi_utils.read_molecule import smiles_or_inchi_to_mol
from lightning_utilities.core.rank_zero import rank_zero_info
from rdkit import RDLogger
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from chebai_graph.preprocessing.properties import MoleculeProperty
from chebai_graph.preprocessing.property_encoder import IndexEncoder
from chebai_graph.preprocessing.reader import AtomFGReader_WithFGEdges_WithGraphNode

from .base import GraphPropAsPerNodeType, GraphPropertiesMixIn
from .pretraining import PretrainingTargetsMixIn
from .pubchem import PubChemGraphMixIn


def _line_offsets(path: str, chunk_size: int = 1 << 26) -> np.ndarray:
    """Byte offset of the start of every line of a text file."""
    offsets = [np.zeros(1, dtype=np.uint64)]
    position = 0
    with open(path, "rb") as f:
        with tqdm.tqdm(total=os.path.getsize(path), unit="B", unit_scale=True) as bar:
            while chunk := f.read(chunk_size):
                newlines = np.flatnonzero(np.frombuffer(chunk, dtype=np.uint8) == 10)
                offsets.append(newlines.astype(np.uint64) + np.uint64(position + 1))
                position += len(chunk)
                bar.update(len(chunk))
    offsets = np.concatenate(offsets)
    # The offset after the last newline is the end of the file
    return offsets[offsets < position]


class _MoleculeStream(IterableDataset):
    """Graphs of the molecules at positions [start, stop) of the permutation."""

    def __init__(self, module: "StreamingPubChemMixIn", start: int, stop: int) -> None:
        self.module = module
        self.start = start
        self.stop = stop

    def __iter__(self) -> Iterator[dict]:
        """Worker k of n builds the positions start + k, start + k + n, ..."""
        info = get_worker_info()
        worker, num_workers = (info.id, info.num_workers) if info else (0, 1)
        RDLogger.DisableLog("rdApp.*")
        warnings.filterwarnings("ignore", message="Failed to sanitize")
        module = self.module
        order = np.load(module.permutation_path, mmap_mode="r")
        offsets = np.load(module.offsets_path, mmap_mode="r")
        with open(module.smiles_path, "rb") as f:
            for i in range(self.start + worker, self.stop, num_workers):
                f.seek(int(offsets[order[i]]))
                line = f.readline().decode().rstrip("\n")
                cid, _, smiles = line.partition("\t")
                # The ident contains the stream position, a resumed run continues after it
                try:
                    row = module.featurize(f"{i}:{cid}", smiles)
                except Exception as error:
                    # RDKit fails on a few unusual fused ring systems, skip them
                    warnings.warn(f"skipped molecule {cid}: {error}")
                    row = None
                if row is not None:
                    yield row


def stream_position(ident: str) -> int:
    """Position in the stream, from an ident of the form "<position>:<CID>"."""
    return int(ident.split(":", 1)[0])


class StreamingPubChemMixIn:
    """
    Turns a PubChem pretraining dataset into a stream of one epoch. Must be the first
    base class.
    """

    def __init__(
        self,
        *args,
        stream_size: int | None = None,
        validation_size: int = 10000,
        stream_seed: int = 42,
        max_atoms: int | None = None,
        zero_graph_node_descriptors: bool = False,
        **kwargs,
    ):
        """
        Args:
            stream_size (int | None): Number of training molecules, None for the whole
                release without the validation molecules.
            validation_size (int): Number of held-out molecules, kept in memory.
            stream_seed (int): Seed of the order in which the molecules are read.
            max_atoms (int | None): Skip molecules with more heavy atoms.
            zero_graph_node_descriptors (bool): Use zeros instead of the RDKit descriptors
                of the graph node. Only valid if the model hides them anyway
                (hide_graph_node_features); saves most of the featurization time.
        """
        super().__init__(*args, **kwargs)
        self.stream_size = stream_size
        self.validation_size = validation_size
        self.stream_seed = stream_seed
        self.max_atoms = max_atoms
        self.zero_graph_node_descriptors = zero_graph_node_descriptors
        self._validation_rows = None
        # Set by MoleculeMilestoneCheckpoint when a run is resumed
        self.resume_position: int | None = None

    # ------------------------------------------------------------------ files

    @property
    def smiles_path(self) -> str:
        return self._full_smiles_path

    @property
    def _stream_dir(self) -> str:
        return os.path.join(self.base_dir, "stream")

    @property
    def offsets_path(self) -> str:
        return os.path.join(self._stream_dir, "line_offsets.npy")

    @property
    def permutation_path(self) -> str:
        return os.path.join(self._stream_dir, f"permutation_seed{self.stream_seed}.npy")

    def prepare_data(self, *args, **kwargs) -> None:
        """Download the release, index its lines and draw the permutation, once."""
        self._download_full()
        os.makedirs(self._stream_dir, exist_ok=True)
        if not os.path.isfile(self.offsets_path):
            rank_zero_info(f"Indexing lines of {self.smiles_path}...")
            np.save(self.offsets_path + ".part.npy", _line_offsets(self.smiles_path))
            os.replace(self.offsets_path + ".part.npy", self.offsets_path)
        if not os.path.isfile(self.permutation_path):
            n = len(np.load(self.offsets_path, mmap_mode="r"))
            order = np.random.default_rng(self.stream_seed).permutation(n)
            np.save(self.permutation_path + ".part.npy", order.astype(np.uint32))
            os.replace(self.permutation_path + ".part.npy", self.permutation_path)

    def setup(self, *args, **kwargs) -> None:
        """Fix the vocabularies and set the end of the training stream."""
        self._num_of_labels = 0
        self._feature_vector_size = 0
        # Encode graph by graph with the fixed vocabulary, unknown tokens become zeros
        for prop in self.properties:
            if isinstance(prop.encoder, IndexEncoder):
                prop.encoder.eval = True
        n = len(np.load(self.permutation_path, mmap_mode="r"))
        available = n - self.validation_size
        if self.stream_size is not None and self.stream_size > available:
            raise ValueError(f"stream_size {self.stream_size} > {available} molecules")
        self._train_stop = self.validation_size + (self.stream_size or available)

    # ----------------------------------------------------------- featurizing

    def featurize(self, ident: str, smiles: str) -> dict | None:
        """
        Build the graph with features and pretraining targets for one SMILES.

        Returns:
            dict | None: Row with features, labels (None) and ident, or None if the
                molecule cannot be read or has more than max_atoms atoms.
        """
        mol = smiles_or_inchi_to_mol(smiles)
        if mol is None:
            return None
        if self.max_atoms is not None and mol.GetNumAtoms() > self.max_atoms:
            return None
        result = self.reader._read_data(mol)
        # The augmented reader keeps every molecule it reads, clear it to save memory
        getattr(self.reader, "mol_object_buffer", {}).clear()
        if result is None:
            return None
        graph, source = result
        row = {"features": graph}
        for prop in self.properties:
            if self.zero_graph_node_descriptors and isinstance(prop, MoleculeProperty):
                row[prop.name] = torch.zeros((1, prop.encoder.get_encoding_length()))
                continue
            values = self.reader.read_property(source, prop)
            if values is None or len(values) == 0:
                row[prop.name] = float("nan")
                continue
            encoded = [prop.encoder.encode(v) for v in values]
            row[prop.name] = torch.cat(
                [e if e.dim() == 2 else e.reshape(1, -1) for e in encoded]
            )
        features = self._merge_props_into_base(row, **self._merge_kwargs())
        if features is None:
            return None
        return {"features": features, "labels": None, "ident": ident}

    # ----------------------------------------------------------- dataloaders

    def _loader(self, dataset: Dataset, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self.reader.collator,
            num_workers=self.num_workers,
            persistent_workers=False,
            **kwargs,
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        start = self.validation_size
        if self.resume_position is not None:
            start = max(start, self.resume_position)
            rank_zero_info(f"Resuming the molecule stream at position {start}")
        stream = _MoleculeStream(self, start, self._train_stop)
        return self._loader(stream)

    def _validation_data(self) -> list[dict]:
        """Graphs of the held-out molecules, built once."""
        if self._validation_rows is None:
            rank_zero_info(f"Building {self.validation_size} validation graphs...")
            stream = _MoleculeStream(self, 0, self.validation_size)
            # Workers return pickled chunks, single tensors need one shared-memory file
            # each and exceed the limit of open files
            loader = DataLoader(
                stream,
                batch_size=256,
                collate_fn=pickle.dumps,
                num_workers=self.num_workers,
            )
            self._validation_rows = [
                row for chunk in loader for row in pickle.loads(chunk)
            ]
            if not self._validation_rows:
                raise RuntimeError("No validation graphs could be built")
        return self._validation_rows

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self._validation_data(),
            batch_size=self.batch_size,
            collate_fn=self.reader.collator,
            shuffle=False,
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.val_dataloader()


class PubChemFlatStream(
    StreamingPubChemMixIn,
    PretrainingTargetsMixIn,
    GraphPropertiesMixIn,
    PubChemGraphMixIn,
):
    """Flat PubChem graphs, built while training."""


class PubChemAugmentedStream(
    StreamingPubChemMixIn,
    PretrainingTargetsMixIn,
    GraphPropAsPerNodeType,
    PubChemGraphMixIn,
):
    """Augmented PubChem graphs (atom, FG and graph nodes), built while training."""

    READER = AtomFGReader_WithFGEdges_WithGraphNode
