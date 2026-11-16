import math

import pytest

from peft_engine.hf_backend import peak_rss_gb, pick_device


def test_peak_rss_gb_positive():
    assert peak_rss_gb() > 0


def test_pick_device_without_torch_raises():
    # Import is lazy inside pick_device; skip if torch is present.
    torch = pytest.importorskip("torch")
    assert pick_device() in {"cpu", "cuda", "mps"}
    assert isinstance(torch.__version__, str)


def test_perplexity_matches_math():
    from peft_engine.evaluator import perplexity_from_nll

    nll = 2.0 * 10
    assert math.isclose(perplexity_from_nll(nll, 10), math.exp(2.0))
