import unittest
from types import SimpleNamespace

from chebai_graph.callbacks import KeepAllLoggers


class TestKeepAllLoggers(unittest.TestCase):
    """Unit tests for KeepAllLoggers, which restores dropped loggers."""

    def test_restores_dropped_loggers(self) -> None:
        """Test that all loggers of a list are restored during setup."""
        csv, wandb = object(), object()
        trainer = SimpleNamespace(init_kwargs={"logger": [csv, wandb]}, loggers=[csv])
        KeepAllLoggers().setup(trainer, None, "fit")
        self.assertEqual(trainer.loggers, [csv, wandb])

    def test_single_logger_unchanged(self) -> None:
        """Test that a single configured logger is left as it is."""
        csv = object()
        trainer = SimpleNamespace(init_kwargs={"logger": csv}, loggers=[csv])
        KeepAllLoggers().setup(trainer, None, "fit")
        self.assertEqual(trainer.loggers, [csv])


if __name__ == "__main__":
    unittest.main()
