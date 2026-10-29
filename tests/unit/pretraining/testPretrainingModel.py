import os
import tempfile
import unittest

import torch

from chebai_graph.models.pretraining import (
    GATAMGPoolGraphPredFromPretraining,
    GATGraphPretraining,
)
from tests.unit.pretraining.utils import chain_graph, collate

CONFIG = dict(
    in_channels=6,
    hidden_channels=16,
    out_channels=16,
    num_layers=2,
    edge_dim=3,
    heads=2,
    v2=True,
    dropout=0,
)
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


def _pretraining_model(tasks, **kwargs) -> GATGraphPretraining:
    """Create a small pretraining model for the test graphs."""
    return GATGraphPretraining(
        config=CONFIG,
        tasks=tasks,
        out_dim=0,
        input_dim=0,
        n_shared_node_columns=2,
        **kwargs,
    )


class TestGATGraphPretraining(unittest.TestCase):
    """Unit tests for GATGraphPretraining."""

    def setUp(self) -> None:
        """Build a batch of four augmented chain molecules."""
        torch.manual_seed(0)
        graphs = [chain_graph(n, augmented=True) for n in (12, 9, 15, 7)]
        self.batch = collate(graphs)

    def test_all_tasks_give_finite_loss_and_gradients(self) -> None:
        """Test that all tasks together give a finite loss and encoder gradients."""
        model = _pretraining_model(ALL_TASKS, mask_rate=0.3)
        out = model._execute(self.batch, 0, prefix="train_", log=False)
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        grads = [p.grad for p in model.gnn.parameters()]
        self.assertTrue(all(g is not None for g in grads))

    def test_targets_outside_the_vocabulary_are_ignored(self) -> None:
        """Test that targets of -1 keep the loss finite."""
        graph = self.batch.x[0]
        graph.y_atom_num_hs[::2] = -1
        model = _pretraining_model(["atom_type", "atom_num_hs"], mask_rate=1.0)
        out = model._execute(self.batch, 0, prefix="train_", log=False)
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_attrmask_masks_only_atoms(self) -> None:
        """Test that AttrMask masks atoms but no bonds or FG nodes."""
        model = _pretraining_model(["atom_type"])
        self.assertGreater(model.atom_rate, 0)
        self.assertEqual(model.bond_rate, 0)
        self.assertEqual(model.fg_rate, 0)

    def test_validation_masks_are_fixed(self) -> None:
        """Test that the validation loss of a batch is the same in every call."""
        model = _pretraining_model(ALL_TASKS).eval()
        with torch.no_grad():
            a = model._execute(self.batch, 3, prefix="val_", log=False)["loss"]
            b = model._execute(self.batch, 3, prefix="val_", log=False)["loss"]
        self.assertTrue(torch.equal(a, b))

    def test_learned_task_weights_are_trainable(self) -> None:
        """Test that learned task weights receive gradients."""
        model = _pretraining_model(ALL_TASKS, task_weighting="learned")
        model._execute(self.batch, 0, prefix="train_", log=False)["loss"].backward()
        self.assertIsNotNone(model.task_logits.grad)

    def test_batch_without_targets_allows_backward(self) -> None:
        """Test that a batch in which no task has a target gives a zero loss."""
        model = _pretraining_model(["fg_type"], mask_rate=0.0)
        out = model._execute(self.batch, 0, prefix="train_", log=False)
        self.assertEqual(float(out["loss"]), 0.0)
        out["loss"].backward()

    def test_unknown_task_is_rejected(self) -> None:
        """Test that an unknown task name raises a ValueError."""
        with self.assertRaises(ValueError):
            _pretraining_model(["no_such_task"])


class TestEncoderTransfer(unittest.TestCase):
    """Unit tests for the transfer of the pretrained encoder to fine-tuning."""

    def test_encoder_weights_are_loaded(self) -> None:
        """Test that the fine-tuning model gets the pretrained encoder weights."""
        pretrained = _pretraining_model(ALL_TASKS)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pretrain.ckpt")
            torch.save({"state_dict": pretrained.state_dict()}, path)
            model = GATAMGPoolGraphPredFromPretraining(
                config=CONFIG,
                n_linear_layers=1,
                out_dim=5,
                input_dim=0,
                pretrained_checkpoint=path,
            )
        for (name, p), q in zip(
            pretrained.gnn.state_dict().items(), model.gnn.state_dict().values()
        ):
            self.assertTrue(torch.equal(p, q), msg=name)

    def test_cold_start_without_checkpoint(self) -> None:
        """Test that the model works without a checkpoint."""
        model = GATAMGPoolGraphPredFromPretraining(
            config=CONFIG, n_linear_layers=1, out_dim=5, input_dim=0
        )
        batch = collate(
            [chain_graph(5, augmented=True), chain_graph(4, augmented=True)]
        )
        out = model({"features": batch.x})
        self.assertEqual(tuple(out.shape), (2, 5))

    def test_mismatching_encoder_is_rejected(self) -> None:
        """Test that an encoder with other dimensions cannot be loaded."""
        other = dict(CONFIG, in_channels=7)
        pretrained = GATGraphPretraining(
            config=other, tasks=["atom_type"], out_dim=0, input_dim=0
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pretrain.ckpt")
            torch.save({"state_dict": pretrained.state_dict()}, path)
            with self.assertRaises(RuntimeError):
                GATAMGPoolGraphPredFromPretraining(
                    config=CONFIG,
                    n_linear_layers=1,
                    out_dim=5,
                    input_dim=0,
                    pretrained_checkpoint=path,
                )


if __name__ == "__main__":
    unittest.main()
