import unittest

import torch

from chebai_graph.models.pretraining.masking import mask_batch
from tests.unit.pretraining.utils import chain_graph, collate


class TestMaskBatch(unittest.TestCase):
    """Unit tests for mask_batch."""

    def setUp(self) -> None:
        """Build a batch of two augmented chain molecules."""
        torch.manual_seed(0)
        self.graph = collate(
            [chain_graph(12, augmented=True), chain_graph(9, augmented=True)]
        ).x[0]

    def _mask(self, **kwargs):
        """Mask the batch with a fixed generator and the given settings."""
        defaults = dict(
            atom_rate=0.5, bond_rate=0.5, fg_rate=1.0, n_shared_node_columns=2
        )
        defaults.update(kwargs)
        g = torch.Generator().manual_seed(1)
        return mask_batch(self.graph, generator=g, **defaults)

    def test_masked_atoms_keep_only_shared_columns(self) -> None:
        """Test that masked atoms keep only the shared node type columns."""
        m = self._mask()
        self.assertGreater(m.masked_atoms.numel(), 0)
        self.assertTrue(torch.all(m.x[m.masked_atoms, 2:] == 0))
        self.assertTrue(torch.all(m.x[m.masked_atoms, :2] > 0))
        self.assertTrue(bool(self.graph.is_atom_node[m.masked_atoms].bool().all()))

    def test_masked_bonds_removed_in_both_directions(self) -> None:
        """Test that removed bonds are missing in both directions."""
        m = self._mask()
        self.assertGreater(m.masked_bonds.shape[1], 0)
        remaining = {tuple(e) for e in m.edge_index.t().tolist()}
        for u, v in m.masked_bonds.t().tolist():
            self.assertNotIn((u, v), remaining)
            self.assertNotIn((v, u), remaining)
        n_removed = self.graph.edge_index.shape[1] - m.edge_index.shape[1]
        self.assertEqual(n_removed, 2 * m.masked_bonds.shape[1])
        self.assertEqual(m.edge_attr.shape[0], m.edge_index.shape[1])
        self.assertTrue(torch.all(m.masked_bond_types == 1))

    def test_negative_pairs_are_unbonded_atoms_of_same_molecule(self) -> None:
        """Test that negative pairs are unbonded atoms of the same molecule."""
        m = self._mask()
        edges = {tuple(e) for e in self.graph.edge_index.long().t().tolist()}
        batch = self.graph.batch
        is_atom = self.graph.is_atom_node.bool()
        self.assertGreater(m.negative_pairs.shape[1], 0)
        for u, v in m.negative_pairs.t().tolist():
            self.assertNotEqual(u, v)
            self.assertNotIn((u, v), edges)
            self.assertEqual(int(batch[u]), int(batch[v]))
            self.assertTrue(bool(is_atom[u]) and bool(is_atom[v]))

    def test_fg_and_graph_node_hidden(self) -> None:
        """Test that masked FG nodes and the graph node keep only the shared columns."""
        m = self._mask(hide_graph_node_features=True)
        is_graph = self.graph.is_graph_node.bool()
        self.assertTrue(torch.all(m.x[is_graph, 2:] == 0))
        self.assertTrue(torch.all(m.x[m.masked_fgs, 2:] == 0))
        self.assertEqual(m.masked_fgs.numel(), 2)

    def test_generator_makes_masks_reproducible(self) -> None:
        """Test that the same generator seed gives the same masks."""
        a, b = self._mask(), self._mask()
        self.assertTrue(torch.equal(a.masked_atoms, b.masked_atoms))
        self.assertTrue(torch.equal(a.masked_bonds, b.masked_bonds))

    def test_no_masking_leaves_input_unchanged(self) -> None:
        """Test that rates of zero leave features and edges unchanged."""
        m = self._mask(atom_rate=0, bond_rate=0, fg_rate=0)
        self.assertTrue(torch.equal(m.x, self.graph.x.float()))
        self.assertTrue(torch.equal(m.edge_index, self.graph.edge_index.long()))


if __name__ == "__main__":
    unittest.main()
