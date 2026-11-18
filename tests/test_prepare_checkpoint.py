"""Author release normalization may only initialize reviewed constant buffers."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_checkpoint import canonical_state


class CheckpointPreparationTests(unittest.TestCase):
    def test_known_constants_preserve_every_matching_tensor(self):
        expected = {"weight": torch.zeros(2), "constant": torch.ones(2)}
        actual = {"weight": torch.full((2,), 3.), "unused": torch.zeros(1)}
        policy = {"missing_constant_buffers": ["constant"], "unused_checkpoint_keys": ["unused"]}
        result = canonical_state(expected, actual, ["weight"], policy)
        self.assertTrue(torch.equal(result["weight"], actual["weight"]))
        self.assertTrue(torch.equal(result["constant"], expected["constant"]))
        with self.assertRaises(ValueError):
            canonical_state(expected, actual, ["weight", "constant"], policy)
        with self.assertRaises(ValueError):
            canonical_state(expected, dict(actual, unknown=torch.zeros(1)), ["weight"], policy)
