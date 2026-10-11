import json

import pytest
import yaml

from peft_engine.config import TrainingSettings
from peft_engine.distill import count_lines, split_counts
from peft_engine.evaluator import merge_metrics, parse_variant, perplexity_from_nll
from peft_engine.trainer import build_lora_config, run_qlora_training


def test_lora_scale_is_alpha_over_rank():
    assert TrainingSettings(lora_rank=16, lora_alpha=32.0).lora_scale == 2.0


def test_build_lora_config():
    cfg = TrainingSettings(lora_rank=8, lora_alpha=16.0, lora_layers=12)
    out = build_lora_config(cfg, data_dir="d", adapter_dir="a", iters=50)
    assert out["lora_parameters"]["rank"] == 8
    assert out["lora_parameters"]["scale"] == 2.0
    assert out["num_layers"] == 12
    assert out["iters"] == 50
    assert out["save_every"] <= 50


def test_dry_run_writes_yaml(tmp_path):
    path = run_qlora_training(
        data_dir="data", adapter_dir=str(tmp_path / "ad"), iters=10, dry_run=True
    )
    assert yaml.safe_load(path.read_text())["iters"] == 10


def test_perplexity():
    assert perplexity_from_nll(0.0, 10) == 1.0
    with pytest.raises(ValueError):
        perplexity_from_nll(1.0, 0)


def test_parse_variant():
    assert parse_variant("base=") == ("base", None)
    assert parse_variant("ft=models/a") == ("ft", "models/a")


def test_merge_metrics(tmp_path):
    p = tmp_path / "r" / "metrics.json"
    merge_metrics(p, "a", {"x": 1})
    merge_metrics(p, "b", {"x": 2})
    assert set(json.loads(p.read_text())) == {"a", "b"}


def test_split_counts_and_count_lines(tmp_path):
    assert split_counts(100) == (90, 10)
    f = tmp_path / "x.jsonl"
    assert count_lines(f) == 0
    f.write_text('{"a":1}\n{"a":2}\n')
    assert count_lines(f) == 2
