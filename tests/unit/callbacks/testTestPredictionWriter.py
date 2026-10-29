import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from chebai_graph.callbacks import TestPredictionWriter


class TestTestPredictionWriter(unittest.TestCase):
    """Unit tests for TestPredictionWriter, which saves the test predictions."""

    def test_saves_probabilities_labels_and_idents(self) -> None:
        """Test that probabilities, labels and idents of all batches are saved."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = SimpleNamespace(log_dir=tmp, default_root_dir=tmp)
            writer = TestPredictionWriter()
            writer.on_test_epoch_start(trainer, None)
            for i in range(2):
                outputs = {
                    "preds": torch.rand(3, 4),
                    "labels": torch.randint(0, 2, (3, 4)),
                }
                batch = SimpleNamespace(
                    additional_fields={"idents": [f"m{i}{j}" for j in range(3)]}
                )
                writer.on_test_batch_end(trainer, None, outputs, batch, i)
            writer.on_test_epoch_end(trainer, None)
            saved = torch.load(os.path.join(tmp, "test_predictions.pt"))
            self.assertEqual(saved["probs"].shape, (6, 4))
            self.assertEqual(saved["probs"].dtype, torch.float32)
            self.assertEqual(saved["labels"].dtype, torch.bool)
            self.assertEqual(
                saved["idents"], ["m00", "m01", "m02", "m10", "m11", "m12"]
            )

    def test_nothing_written_without_test_batches(self) -> None:
        """Test that no file is written if no test batch was seen."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = SimpleNamespace(log_dir=tmp, default_root_dir=tmp)
            writer = TestPredictionWriter()
            writer.on_test_epoch_start(trainer, None)
            writer.on_test_epoch_end(trainer, None)
            self.assertFalse(os.path.exists(os.path.join(tmp, "test_predictions.pt")))


if __name__ == "__main__":
    unittest.main()
