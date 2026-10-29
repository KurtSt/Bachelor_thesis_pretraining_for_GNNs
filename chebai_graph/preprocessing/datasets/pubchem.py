import gzip
import os
import random
import shutil
from typing import Generator

import requests
import tqdm
from chebai.preprocessing.datasets.pubchem import PubChem
from chebi_utils.read_molecule import smiles_or_inchi_to_mol

from chebai_graph.preprocessing.property_encoder import IndexEncoder
from chebai_graph.preprocessing.reader import AtomFGReader_WithFGEdges_WithGraphNode

from .base import GraphPropAsPerNodeType, GraphPropertiesMixIn
from .pretraining import PretrainingTargetsMixIn


class PubChemGraphMixIn(PubChem):
    """
    PubChem as unlabeled graph data.

    Unlike PubChem of chebai, it downloads a fixed monthly release and draws the n_samples
    subset with a fixed seed. It yields RDKit molecules like ChEBI, so the graph readers
    treat both datasets the same way, and it uses the token vocabularies of ChEBI. Must
    come after the property mixins in the base classes.
    """

    PUBCHEM_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Monthly/{release}/Extras/CID-SMILES.gz"

    def __init__(
        self,
        *args,
        pubchem_release: str = "2026-09-01",
        sample_seed: int = 42,
        **kwargs,
    ):
        """
        Args:
            pubchem_release (str): Monthly PubChem release, the folder name on the PubChem
                FTP server.
            sample_seed (int): Seed for drawing the n_samples subset.
        """
        self.pubchem_release = pubchem_release
        self.sample_seed = sample_seed
        super().__init__(*args, **kwargs)
        self.pubchem_url = self.PUBCHEM_URL.format(release=pubchem_release)

    @property
    def data_type(self) -> str:
        # Same vocabulary as the ChEBI data used for fine-tuning
        return "chebi"

    @property
    def base_dir(self) -> str:
        if self._base_dir is not None:
            return self._base_dir
        return os.path.join("data", self._name, self.pubchem_release)

    @property
    def _full_smiles_path(self) -> str:
        return os.path.join(self.base_dir, "raw", "full", "smiles.txt")

    def _download_full(self) -> None:
        """Download and unpack the SMILES file of the release if it is missing."""
        path = self._full_smiles_path
        if os.path.isfile(path):
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        gz_path = path + ".gz"
        print(f"Download from {self.pubchem_url}")
        with requests.get(self.pubchem_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(gz_path + ".part", "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        os.replace(gz_path + ".part", gz_path)
        print("Unpacking...")
        with gzip.open(gz_path, "rb") as f_in, open(path + ".part", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.replace(path + ".part", path)
        os.remove(gz_path)

    def download(self) -> None:
        """Raw SMILES file: the whole release or a subset of n_samples molecules."""
        target = os.path.join(self.raw_dir, "smiles.txt")
        if os.path.isfile(target):
            return
        self._download_full()
        if self._n_samples == PubChem.FULL:
            return
        with open(self._full_smiles_path, "r") as f_in:
            n_lines = sum(1 for _ in f_in)
        selected = set(
            random.Random(self.sample_seed).sample(range(n_lines), k=self._n_samples)
        )
        os.makedirs(self.raw_dir, exist_ok=True)
        with (
            open(self._full_smiles_path, "r") as f_in,
            open(target + ".part", "w") as f_out,
        ):
            for i, line in enumerate(tqdm.tqdm(f_in, total=n_lines)):
                if i in selected:
                    f_out.write(line)
        os.replace(target + ".part", target)

    @property
    def raw_dir(self) -> str:
        if self._n_samples == PubChem.FULL:
            return os.path.dirname(self._full_smiles_path)
        return os.path.join(
            self.base_dir, "raw", f"{self.split_label}_seed{self.sample_seed}"
        )

    @property
    def processed_dir_main(self) -> str:
        return os.path.join(
            self.base_dir, "processed", f"{self.split_label}_seed{self.sample_seed}"
        )

    def _load_dict(self, input_file_path: str) -> Generator[dict, None, None]:
        """Rows with RDKit molecules instead of SMILES; unreadable molecules are skipped."""
        for row in super()._load_dict(input_file_path):
            mol = smiles_or_inchi_to_mol(row["features"])
            if mol is None:
                continue
            row["features"] = mol
            yield row


class _FrozenVocabularyMixIn:
    """Freezes the token vocabularies of all index encoders. Must be the first base class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for prop in self.properties:
            if isinstance(prop.encoder, IndexEncoder):
                prop.encoder.frozen = True


class PubChemGraphProperties(
    _FrozenVocabularyMixIn, GraphPropertiesMixIn, PubChemGraphMixIn
):
    """Flat molecular graphs of PubChem molecules."""


class PubChemFlatPretraining(PretrainingTargetsMixIn, PubChemGraphProperties):
    """Flat PubChem graphs with pretraining targets."""


class PubChemAugmentedPretraining(
    _FrozenVocabularyMixIn,
    PretrainingTargetsMixIn,
    GraphPropAsPerNodeType,
    PubChemGraphMixIn,
):
    """
    Augmented PubChem graphs with pretraining targets, built with the reader and feature
    layout of ChEBI50_WFGE_WGN_AsPerNodeType.
    """

    READER = AtomFGReader_WithFGEdges_WithGraphNode
