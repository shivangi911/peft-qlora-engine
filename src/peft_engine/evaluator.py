"""Evaluate model variants: perplexity, peak memory, generation speed -> results/metrics.json."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

from peft_engine.config import settings


def perplexity_from_nll(total_nll: float, total_tokens: int) -> float:
    if total_tokens <= 0:
        raise ValueError("no tokens to evaluate")
    return math.exp(total_nll / total_tokens)


def parse_variant(spec: str) -> tuple[str, str | None]:
    """'name=adapter_dir' or 'name' / 'name=' for the plain base model."""
    name, _, adapter = spec.partition("=")
    return name, (adapter or None)


def merge_metrics(path: Path, name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    existing: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing[name] = metrics
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2))
    return existing


def _peak_memory_gb() -> float:
    import mlx.core as mx

    fn = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    return float(fn()) / 1e9


def _reset_peak_memory() -> None:
    import mlx.core as mx

    fn = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
    fn()


def evaluate_variant(
    model_path: str,
    adapter_path: str | None,
    valid_file: Path,
    num_samples: int,
    max_len: int,
    gen_tokens: int,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, stream_generate

    _reset_peak_memory()
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    rows = [json.loads(line) for line in valid_file.read_text().splitlines() if line.strip()]
    total_nll, total_tokens = 0.0, 0
    for row in rows[:num_samples]:
        ids = tokenizer.encode(row["text"])[:max_len]
        if len(ids) < 2:
            continue
        x = mx.array(ids[:-1])[None]
        y = mx.array(ids[1:])[None]
        loss = nn.losses.cross_entropy(model(x), y, reduction="sum")
        mx.eval(loss)
        total_nll += float(loss.item())
        total_tokens += len(ids) - 1

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain the difference between QuickSort and MergeSort."}],
        add_generation_prompt=True,
        tokenize=False,
    )
    tps, last = 0.0, None
    start = time.time()
    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=gen_tokens):
        last = resp
    if last is not None:
        tps = float(getattr(last, "generation_tps", 0.0)) or gen_tokens / (time.time() - start)

    return {
        "model": model_path,
        "adapter": adapter_path,
        "perplexity": round(perplexity_from_nll(total_nll, total_tokens), 4),
        "eval_tokens": total_tokens,
        "peak_memory_gb": round(_peak_memory_gb(), 2),
        "generation_tokens_per_sec": round(tps, 2),
        "machine": platform.platform(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base / fine-tuned variants")
    parser.add_argument(
        "--variant",
        action="append",
        help="name=adapter_dir (empty adapter = base model). Repeatable.",
    )
    parser.add_argument("--model", default=settings.base_model)
    parser.add_argument("--valid-file", default=f"{settings.data_dir}/valid.jsonl")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--gen-tokens", type=int, default=256)
    parser.add_argument("--output", default=f"{settings.results_dir}/metrics.json")
    args = parser.parse_args()

    variants = args.variant or [
        "base=",
        f"slimorca_ft={settings.adapter_output_dir}",
        f"distilled_ft={settings.distilled_adapter_dir}",
    ]
    for spec in variants:
        name, adapter = parse_variant(spec)
        if adapter and not Path(adapter).exists():
            print(f"Skipping {name}: adapter dir {adapter} not found")
            continue
        print(f"Evaluating {name}")
        metrics = evaluate_variant(
            args.model,
            adapter,
            Path(args.valid_file),
            args.num_samples,
            args.max_len,
            args.gen_tokens,
        )
        merge_metrics(Path(args.output), name, metrics)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
