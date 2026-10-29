import unittest
from types import SimpleNamespace

from chebai_graph.models.pretraining.callbacks import MoleculeMilestoneCheckpoint


class _Trainer:
    """Records the paths of the saved checkpoints."""

    def __init__(self) -> None:
        self.saved = []
        self.log_dir = "run"
        self.default_root_dir = "logs"
        self.datamodule = SimpleNamespace(resume_position=None)

    def save_checkpoint(self, path: str) -> None:
        self.saved.append(path)


class _Module:
    def log(self, *args, **kwargs) -> None:
        pass


def _batch(positions):
    """Build a batch with the idents of the given stream positions."""
    return SimpleNamespace(
        additional_fields={"idents": [f"{p}:cid{p}" for p in positions]}
    )


class TestMoleculeMilestoneCheckpoint(unittest.TestCase):
    """Unit tests for MoleculeMilestoneCheckpoint."""

    def test_checkpoint_when_milestone_is_passed(self) -> None:
        """Test that a checkpoint is saved once per passed milestone."""
        callback = MoleculeMilestoneCheckpoint(milestones=[5, 10])
        trainer = _Trainer()
        for step in range(4):  # 3 molecules per batch -> 3, 6, 9, 12
            callback.on_train_batch_end(
                trainer, _Module(), None, _batch(range(3 * step, 3 * step + 3)), step
            )
        self.assertEqual(
            trainer.saved,
            ["run/checkpoints/molecules_5.ckpt", "run/checkpoints/molecules_10.ckpt"],
        )
        self.assertEqual(callback.molecules_seen, 12)
        self.assertEqual(callback.last_position, 11)

    def test_final_checkpoint_when_training_ends(self) -> None:
        """Test that final.ckpt is saved when training ends."""
        callback = MoleculeMilestoneCheckpoint(milestones=[100])
        trainer = _Trainer()
        callback.on_train_end(trainer, _Module())
        self.assertEqual(trainer.saved, ["run/checkpoints/final.ckpt"])

    def test_resume_sets_stream_position(self) -> None:
        """Test that restoring the state sets the resume position of the data module."""
        callback = MoleculeMilestoneCheckpoint(milestones=[100])
        trainer = _Trainer()
        callback.setup(trainer, _Module(), "fit")
        callback.load_state_dict({"molecules_seen": 40, "last_position": 57})
        self.assertEqual(trainer.datamodule.resume_position, 58)
        self.assertEqual(callback.molecules_seen, 40)


if __name__ == "__main__":
    unittest.main()
