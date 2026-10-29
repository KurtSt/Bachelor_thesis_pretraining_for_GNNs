import unittest

import torch
from torch_geometric.data import Data as GeomData

from chebai_graph.preprocessing.collate import GraphCollator


def _row(x: torch.Tensor, labels) -> dict:
    """Build a dataset row with a two-node graph and the given labels."""
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_attr = torch.ones((2, 3))
    return {
        "features": GeomData(x=x, edge_index=edge_index, edge_attr=edge_attr),
        "labels": labels,
        "ident": "id",
    }


class TestGraphCollator(unittest.TestCase):
    """Unit tests for GraphCollator."""

    def test_float_node_features_are_kept(self) -> None:
        """Test that float node features are not cast to integers."""
        x = torch.tensor([[0.25, 1.0], [0.75, 0.0]])
        batch = GraphCollator()([_row(x, [True, False]), _row(x, [False, True])])
        graph = batch.x[0]
        self.assertTrue(graph.x.is_floating_point())
        self.assertTrue(torch.allclose(graph.x, torch.cat([x, x])))

    def test_unlabeled_batch(self) -> None:
        """Test that a batch without labels has no targets and no non-null labels."""
        x = torch.zeros((2, 2))
        batch = GraphCollator()([_row(x, None), _row(x, None)])
        self.assertIsNone(batch.y)
        self.assertEqual(batch.additional_fields["loss_kwargs"]["non_null_labels"], [])


if __name__ == "__main__":
    unittest.main()
