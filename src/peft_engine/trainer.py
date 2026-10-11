"""LoRA / QLoRA training orchestrator (Apple MLX).

Training a 4-bit checkpoint with mlx_lm.lora is QLoRA: frozen quantized base weights
plus trainable low-rank adapters.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from peft_engine.config import TrainingSettings, settings


def build_lora_config(
    cfg: TrainingSettings,
    *,
    data_dir: str,
    adapter_dir: str,
    model: str | None = None,
    iters: int | None = None,
) -> dict[str, Any]:
    iters = iters if iters is not None else cfg.iters
    return {
        "model": model or cfg.base_model,
        "train": True,
        "fine_tune_type": "lora",
        "data": data_dir,
        "adapter_path": adapter_dir,
        "num_layers": cfg.lora_layers,
        "batch_size": cfg.batch_size,
        "iters": iters,
        "learning_rate": cfg.learning_rate,
        "val_batches": cfg.val_batches,
        "steps_per_eval": min(cfg.steps_per_eval, iters),
        "steps_per_report": 10,
        "save_every": min(cfg.save_every, iters),
        "max_seq_length": cfg.max_seq_length,
        "grad_checkpoint": cfg.grad_checkpoint,
        "lora_parameters": {
            "rank": cfg.lora_rank,
            "scale": cfg.lora_scale,
            "dropout": cfg.lora_dropout,
        },
        "seed": 0,
    }


def run_qlora_training(
    data_dir: str | None = None,
    adapter_dir: str | None = None,
    iters: int | None = None,
    dry_run: bool = False,
) -> Path:
    data_dir = data_dir or settings.data_dir
    adapter_dir = adapter_dir or settings.adapter_output_dir
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)

    config = build_lora_config(settings, data_dir=data_dir, adapter_dir=adapter_dir, iters=iters)
    config_path = Path(adapter_dir) / "lora_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Wrote training config to {config_path}")

    if dry_run:
        return config_path
    subprocess.run([sys.executable, "-m", "mlx_lm.lora", "-c", str(config_path)], check=True)
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoRA/QLoRA fine-tuning with MLX")
    parser.add_argument("--data-dir", default=None, help="Dir with train.jsonl / valid.jsonl")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--iters", type=int, default=None, help="Override iterations (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="Only write the YAML config")
    args = parser.parse_args()
    run_qlora_training(args.data_dir, args.adapter_dir, args.iters, args.dry_run)


if __name__ == "__main__":
    main()
