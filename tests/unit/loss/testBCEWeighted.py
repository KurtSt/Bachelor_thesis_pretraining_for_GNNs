import os
import tempfile
import unittest
from unittest import mock

import torch
from chebai.loss.bce_weighted import BCEWeighted

from chebai_graph.loss.bce_weighted import BCEWeightedFromLabels


class _Extractor:
    """Returns fixed rows instead of a processed dataset."""

    chebi_version = 0
    processed_file_names = ["data.pt"]

    def __init__(self, rows, directory):
        self.rows = rows
        self.processed_dir = directory

    def load_processed_data(self, filename):
        return self.rows

    def load_processed_data_from_file(self, filename):
        return self.rows


class TestBCEWeightedFromLabels(unittest.TestCase):
    """Unit tests for BCEWeightedFromLabels."""

    def test_same_weights_as_chebai(self) -> None:
        """Test that the weights equal those of BCEWeighted and are cached."""
        torch.manual_seed(0)
        labels = (torch.rand(50, 7) < 0.3).tolist()
        rows = [{"labels": row, "features": None} for row in labels]
        with tempfile.TemporaryDirectory() as tmp:
            extractor = _Extractor(rows, tmp)
            with mock.patch.object(
                BCEWeighted,
                "__init__",
                lambda self, beta, data_extractor: (
                    torch.nn.BCEWithLogitsLoss.__init__(self),
                    setattr(self, "beta", beta),
                    setattr(self, "data_extractor", data_extractor),
                    setattr(self, "pos_weight", None),
                )
                and None,
            ):
                reference = BCEWeighted(beta=0.99, data_extractor=extractor)
                ours = BCEWeightedFromLabels(beta=0.99, data_extractor=extractor)
                x = torch.zeros(1, 7)
                reference.set_pos_weight(x)
                ours.set_pos_weight(x)
                self.assertTrue(torch.allclose(reference.pos_weight, ours.pos_weight))
                self.assertTrue(os.path.isfile(os.path.join(tmp, "label_counts.pt")))
                cached = BCEWeightedFromLabels(beta=0.99, data_extractor=extractor)
                cached.set_pos_weight(x)
                self.assertTrue(torch.allclose(cached.pos_weight, ours.pos_weight))


if __name__ == "__main__":
    unittest.main()
