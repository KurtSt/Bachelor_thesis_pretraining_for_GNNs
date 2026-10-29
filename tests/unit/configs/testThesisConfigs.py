"""Pretraining and fine-tuning configs must use the same features and encoder."""

import os
import unittest

import yaml

CFG = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "thesis")


def _load(*parts: str) -> dict:
    """Load a YAML file of configs/thesis."""
    with open(os.path.join(CFG, *parts)) as f:
        return yaml.safe_load(f)


class TestThesisConfigs(unittest.TestCase):
    """Unit tests for the consistency of the thesis configs."""

    PAIRS = [
        # (pretraining data, fine-tuning data, pretraining model, fine-tuning model)
        (
            "pubchem_stream_flat.yml",
            "chebi50_flat.yml",
            "pretrain_flat.yml",
            "gat_flat.yml",
        ),
        (
            "pubchem_stream_augmented.yml",
            "chebi50_augmented.yml",
            "pretrain_augmented.yml",
            "gat_augmented.yml",
        ),
    ]

    def test_same_properties(self) -> None:
        """Test that pretraining and fine-tuning data use the same properties."""
        for pre, fine, _, _ in self.PAIRS:
            self.assertEqual(
                _load("data", pre)["init_args"]["properties"],
                _load("data", fine)["init_args"]["properties"],
                msg=f"{pre} vs {fine}",
            )

    def test_same_encoder(self) -> None:
        """Test that pretraining and fine-tuning models use the same encoder."""
        for _, _, pre, fine in self.PAIRS:
            self.assertEqual(
                _load("model", pre)["init_args"]["config"],
                _load("model", fine)["init_args"]["config"],
                msg=f"{pre} vs {fine}",
            )


if __name__ == "__main__":
    unittest.main()
