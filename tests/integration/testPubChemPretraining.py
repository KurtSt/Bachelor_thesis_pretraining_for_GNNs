"""
End-to-end check of the pretraining data path on a handful of PubChem-like molecules:
dataset processing -> batching -> masking -> all pretext tasks.

Run with: python -m unittest tests.integration.testPubChemPretraining
"""

import os
import tempfile
import unittest

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from chebai_graph.models.pretraining import GATGraphPretraining
from chebai_graph.preprocessing import properties as P
from chebai_graph.preprocessing.datasets import (
    PubChemAugmentedPretraining,
    PubChemFlatPretraining,
)

SMILES = [
    "CCO",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "C1=CC=C(C=C1)C=O",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "C[N+](C)(C)CC(=O)[O-]",
    "OC1C(O)C(O)C(O)C(O)C1O",
    "CCN(CC)CC",
    "NC(=O)C1=CC=CN=C1",
    "O=C(O)CCC(=O)O",
    "CC(=O)NC1=CC=C(O)C=C1",
    "C1CCNCC1",
    "OCC(O)CO",
    "CC1=CC(=O)C=CC1=O",
    "CSCCC(N)C(=O)O",
    "C#N",
    "[Na+].[Cl-]",
] * 4

FLAT_PROPERTIES = [
    P.AtomType,
    P.NumAtomBonds,
    P.AtomCharge,
    P.AtomAromaticity,
    P.AtomHybridization,
    P.AtomNumHs,
    P.BondType,
    P.BondInRing,
    P.BondAromaticity,
]
AUGMENTED_PROPERTIES = [
    P.AtomNodeLevel,
    P.AugAtomAromaticity,
    P.AugAtomCharge,
    P.AugAtomHybridization,
    P.AugAtomNumHs,
    P.AugAtomType,
    P.AugNumAtomBonds,
    P.AtomFunctionalGroup,
    P.IsHydrogenBondDonorFG,
    P.IsHydrogenBondAcceptorFG,
    P.IsFGAlkyl,
    P.AugRDKit2DNormalized,
    P.BondLevel,
    P.AugBondAromaticity,
    P.AugBondInRing,
    P.AugBondType,
]
ALL_TASKS = [
    "atom_type",
    "atom_num_hs",
    "bond_type",
    "bond_link",
    "fg_type",
    "atom_count",
    "bond_count",
    "hydrogen_count",
    "mol_weight",
    "total_charge",
]


def _class_paths(props):
    """Class paths of property classes, as in the configs."""
    return [f"{p.__module__}.{p.__name__}" for p in props]


def _vocabulary_snapshot() -> dict[str, str]:
    """Read all ChEBI token files, to check that they stay unchanged."""
    root = os.path.join(os.path.dirname(P.__file__), "..", "bin", "chebi")
    snapshot = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            with open(path) as f:
                snapshot[path] = f.read()
    return snapshot


class _PubChemPretrainingCase:
    """Shared tests of the cached PubChem pretraining datasets."""

    DATASET = None
    PROPERTIES = None
    IN_CHANNELS = None
    EDGE_DIM = None
    TASKS = None
    SHARED_COLUMNS = 0

    def setUp(self) -> None:
        """Write the test molecules as raw SMILES file and process the dataset."""
        self.tmp = tempfile.TemporaryDirectory()
        self.vocabulary = _vocabulary_snapshot()
        self.data = self.DATASET(
            properties=_class_paths(self.PROPERTIES),
            n_samples=len(SMILES),
            base_dir=self.tmp.name,
            batch_size=8,
            num_workers=0,
            persistent_workers=False,
        )
        os.makedirs(self.data.raw_dir, exist_ok=True)
        with open(os.path.join(self.data.raw_dir, "smiles.txt"), "w") as f:
            f.writelines(f"{i}\t{s}\n" for i, s in enumerate(SMILES))
        self.data.prepare_data()
        self.data.setup()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_features_match_finetuning_dimensions(self) -> None:
        """Test that the features have the dimensions of the fine-tuning data."""
        batch = next(iter(self.data.train_dataloader()))
        graph = batch.x[0]
        self.assertEqual(graph.x.shape[1], self.IN_CHANNELS)
        self.assertEqual(graph.edge_attr.shape[1], self.EDGE_DIM)
        self.assertIsNone(batch.y)

    def test_vocabulary_files_unchanged(self) -> None:
        """Test that processing PubChem does not extend the ChEBI token files."""
        self.assertEqual(_vocabulary_snapshot(), self.vocabulary)

    def test_targets_are_consistent(self) -> None:
        """Test that atom targets and counts agree with the graph structure."""
        graph = next(iter(self.data.train_dataloader())).x[0]
        is_atom = graph.is_atom_node.bool()
        atom_type = graph.y_atom_type.long()
        self.assertTrue(bool((atom_type[is_atom] >= 0).all()))
        self.assertTrue(bool((atom_type[~is_atom] == -1).all()))
        n_graphs = int(graph.batch.max()) + 1
        atoms = torch.bincount(graph.batch[is_atom], minlength=n_graphs).float()
        self.assertTrue(torch.equal(atoms, graph.y_num_atoms))
        n_bond_edges = int((graph.y_bond_type >= 0).sum())
        self.assertEqual(n_bond_edges, int(2 * graph.y_num_bonds.sum()))

    def test_targets_match_rdkit(self) -> None:
        """Test that the targets equal the values RDKit computes for each molecule."""
        for kind in ("train", "validation", "test"):
            for row in self.data.load_processed_data(kind):
                mol = Chem.MolFromSmiles(SMILES[int(row["ident"])])
                g = row["features"]
                self.assertEqual(int(g.y_num_atoms), mol.GetNumAtoms())
                self.assertEqual(int(g.y_num_bonds), mol.GetNumBonds())
                self.assertEqual(
                    int(g.y_num_hydrogens),
                    sum(a.GetTotalNumHs() for a in mol.GetAtoms()),
                )
                self.assertEqual(int(g.y_total_charge), Chem.GetFormalCharge(mol))
                self.assertAlmostEqual(
                    float(g.y_mol_weight), Descriptors.MolWt(mol), places=2
                )
                # per-atom hydrogens: classes of the AtomNumHs vocabulary, compare as values
                tokens = {
                    i: int(t)
                    for t, i in self.data._property("AtomNumHs").encoder.cache.items()
                }
                atom_hs = [
                    tokens[int(c)] for c in g.y_atom_num_hs[g.is_atom_node.bool()]
                ]
                self.assertEqual(atom_hs, [a.GetTotalNumHs() for a in mol.GetAtoms()])
                atom_type = g.y_atom_type[g.is_atom_node.bool()]
                self.assertEqual(
                    len(set(atom_type.tolist())),
                    len({a.GetSymbol() for a in mol.GetAtoms()}),
                )

    def test_training_step(self) -> None:
        """Test that a pretraining step on a batch gives a finite loss."""
        config = dict(
            in_channels=self.IN_CHANNELS,
            hidden_channels=32,
            out_channels=32,
            num_layers=2,
            edge_dim=self.EDGE_DIM,
            heads=2,
            v2=True,
            dropout=0,
        )
        model = GATGraphPretraining(
            config=config,
            tasks=self.TASKS,
            out_dim=0,
            input_dim=0,
            n_shared_node_columns=self.SHARED_COLUMNS,
            mask_rate=0.3,
        )
        batch = next(iter(self.data.train_dataloader()))
        out = model._execute(batch, 0, prefix="train_", log=False)
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()


class TestPubChemFlatPretraining(_PubChemPretrainingCase, unittest.TestCase):
    """Integration tests for PubChemFlatPretraining."""

    DATASET = PubChemFlatPretraining
    PROPERTIES = FLAT_PROPERTIES
    IN_CHANNELS = 161
    EDGE_DIM = 8
    TASKS = ["atom_type"]


class TestPubChemAugmentedPretraining(_PubChemPretrainingCase, unittest.TestCase):
    """Integration tests for PubChemAugmentedPretraining."""

    DATASET = PubChemAugmentedPretraining
    PROPERTIES = AUGMENTED_PROPERTIES
    IN_CHANNELS = 203
    EDGE_DIM = 12
    TASKS = ALL_TASKS
    SHARED_COLUMNS = 3

    def test_fg_targets(self) -> None:
        """Test that only FG nodes have FG targets."""
        graph = next(iter(self.data.train_dataloader())).x[0]
        is_fg = graph.is_fg_node.bool()
        self.assertTrue(bool(is_fg.any()))
        self.assertTrue(bool((graph.y_fg_type[is_fg] >= 0).all()))
        self.assertTrue(bool((graph.y_fg_type[~is_fg] == -1).all()))


if __name__ == "__main__":
    unittest.main()
