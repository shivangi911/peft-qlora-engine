import numpy as np
import pytest

from peft_engine.fuser import fuse_weights


def test_fused_matches_adapter_forward():
    rng = np.random.default_rng(0)
    d, k, r, alpha = 32, 24, 4, 8.0
    w0 = rng.normal(size=(d, k))
    a = rng.normal(size=(r, k))
    b = rng.normal(size=(d, r))
    x = rng.normal(size=(k,))

    adapter_out = w0 @ x + (alpha / r) * (b @ (a @ x))
    fused_out = fuse_weights(w0, a, b, alpha, r) @ x
    np.testing.assert_allclose(adapter_out, fused_out, atol=1e-10)


def test_zero_b_is_identity():
    w0 = np.eye(4)
    out = fuse_weights(w0, np.ones((2, 4)), np.zeros((4, 2)), 4.0, 2)
    np.testing.assert_array_equal(out, w0)


def test_rank_mismatch_raises():
    with pytest.raises(ValueError):
        fuse_weights(np.eye(4), np.ones((3, 4)), np.zeros((4, 3)), 4.0, 2)
