"""
The streaming PubChem datasets build the same graphs and targets as the cached ones.

Edges are compared as a set (edge with its features): the augmented reader does not
always emit the edges of a molecule in the same order, which does not matter for a GNN.

Run with: python -m unittest tests.integration.testPubChemStream
"""

import os
import tempfile
import unittest

import torch

from chebai_graph.preprocessing.datasets import (
    PubChemAugmentedPretraining,
    PubChemAugmentedStream,
    PubChemFlatPretraining,
    PubChemFlatStream,
)
from tests.integration.testPubChemPretraining import (
    AUGMENTED_PROPERTIES,
    FLAT_PROPERTIES,
    SMILES,
    _class_paths,
)

UNIQUE = list(dict.fromkeys(SMILES))


def _edges(graph):
    """Edges of a graph with their features, sorted."""
    pairs = graph.edge_index.long().t().tolist()
    attrs = graph.edge_attr.float().tolist()
    return sorted((tuple(p), tuple(a)) for p, a in zip(pairs, attrs))


class _StreamCase:
    """Shared tests of the streaming PubChem datasets against the cached ones."""

    CACHED = None
    STREAM = None
    PROPERTIES = None

    def setUp(self) -> None:
        """Process the test molecules with the cached and with the streaming dataset."""
        self.tmp = tempfile.TemporaryDirectory()
        common = dict(
            properties=_class_paths(self.PROPERTIES),
            base_dir=self.tmp.name,
            batch_size=4,
            num_workers=0,
            persistent_workers=False,
        )
        lines = [f"{i}\t{s}\n" for i, s in enumerate(UNIQUE)]
        self.cached = self.CACHED(n_samples=len(UNIQUE), **common)
        os.makedirs(self.cached.raw_dir, exist_ok=True)
        with open(os.path.join(self.cached.raw_dir, "smiles.txt"), "w") as f:
            f.writelines(lines)
        self.cached.prepare_data()
        self.cached.setup()
        self.stream = self.STREAM(validation_size=3, **common)
        os.makedirs(os.path.dirname(self.stream.smiles_path), exist_ok=True)
        with open(self.stream.smiles_path, "w") as f:
            f.writelines(lines)
        self.stream.prepare_data()
        self.stream.setup()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_same_graphs_as_cached_dataset(self) -> None:
        """Test that the stream builds the same graphs as the cached dataset."""
        cached = {
            str(r["ident"]): r["features"]
            for kind in ("train", "validation", "test")
            for r in self.cached.load_processed_data(kind)
        }
        for i, smiles in enumerate(UNIQUE):
            ref = cached[str(i)]
            out = self.stream.featurize(str(i), smiles)["features"]
            keys = sorted(k for k in ref.keys() if isinstance(ref[k], torch.Tensor))
            self.assertEqual(
                keys, sorted(k for k in out.keys() if isinstance(out[k], torch.Tensor))
            )
            self.assertEqual(_edges(ref), _edges(out), msg=smiles)
            for k in keys:
                if k in ("edge_index", "edge_attr", "y_bond_type"):
                    continue
                self.assertTrue(
                    torch.equal(ref[k].float(), out[k].float()), msg=f"{smiles} {k}"
                )

    def test_stream_covers_all_molecules_once(self) -> None:
        """Test that training and validation contain every molecule exactly once."""
        train = [
            i
            for b in self.stream.train_dataloader()
            for i in b.additional_fields["idents"]
        ]
        val = [r["ident"] for r in self.stream._validation_data()]
        self.assertEqual(len(val), 3)
        cids = [i.split(":", 1)[1] for i in train + val]
        self.assertEqual(sorted(cids, key=int), [str(i) for i in range(len(UNIQUE))])

    def test_prefix_of_longer_stream(self) -> None:
        """Test that a shorter stream is a prefix of a longer one."""
        self.stream.stream_size = 4
        self.stream.setup()
        short = [
            i
            for b in self.stream.train_dataloader()
            for i in b.additional_fields["idents"]
        ]
        self.stream.stream_size = None
        self.stream.setup()
        full = [
            i
            for b in self.stream.train_dataloader()
            for i in b.additional_fields["idents"]
        ]
        self.assertEqual(full[: len(short)], short)


class TestFlatStream(_StreamCase, unittest.TestCase):
    """Integration tests for PubChemFlatStream."""

    CACHED = PubChemFlatPretraining
    STREAM = PubChemFlatStream
    PROPERTIES = FLAT_PROPERTIES


class TestAugmentedStream(_StreamCase, unittest.TestCase):
    """Integration tests for PubChemAugmentedStream."""

    CACHED = PubChemAugmentedPretraining
    STREAM = PubChemAugmentedStream
    PROPERTIES = AUGMENTED_PROPERTIES


if __name__ == "__main__":
    unittest.main()
