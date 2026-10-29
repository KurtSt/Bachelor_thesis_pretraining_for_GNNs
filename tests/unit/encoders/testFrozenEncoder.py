import os
import tempfile
import unittest

import torch

from chebai_graph.preprocessing.property_encoder import OneHotEncoder


class _Property:
    name = "Dummy"


class TestFrozenOneHotEncoder(unittest.TestCase):
    """Unit tests for the frozen vocabulary of OneHotEncoder."""

    def setUp(self) -> None:
        """Write a token file with the tokens C, N and O to a temporary directory."""
        self.tmp = tempfile.TemporaryDirectory()
        path = os.path.join(
            self.tmp.name, "bin", "test", "Dummy", "indices_one_hot.txt"
        )
        os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            f.write("C\nN\nO\n")
        self.path = path

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _encoder(self) -> OneHotEncoder:
        """Create an encoder that reads the temporary token file."""
        return OneHotEncoder(_Property(), data_type="test", indices_dir=self.tmp.name)

    def test_unknown_token_is_zero_row_and_vocab_unchanged(self) -> None:
        """Test that unknown tokens become zero rows and the token file stays."""
        enc = self._encoder()
        enc.frozen = True
        values = [["C", "Fe"], ["O"]]
        enc.on_start(property_values=values)
        encoded = torch.cat([enc.encode(v) for v in values[0]])
        self.assertTrue(torch.equal(encoded, torch.tensor([[1, 0, 0], [0, 0, 0]])))
        enc.on_finish()
        with open(self.path) as f:
            self.assertEqual(f.read().split(), ["C", "N", "O"])
        self.assertEqual(enc.get_encoding_length(), 3)

    def test_unfrozen_encoder_extends_vocab(self) -> None:
        """Test that an encoder that is not frozen adds unknown tokens."""
        enc = self._encoder()
        enc.on_start(property_values=[["Fe"]])
        self.assertEqual(enc.get_encoding_length(), 4)


if __name__ == "__main__":
    unittest.main()
