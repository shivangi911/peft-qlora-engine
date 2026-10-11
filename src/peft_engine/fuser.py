"""Adapter fusion: W_fused = W_0 + (alpha / r) * B @ A."""

from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np

from peft_engine.config import settings


def fuse_weights(
    w0: np.ndarray, a: np.ndarray, b: np.ndarray, alpha: float, rank: int
) -> np.ndarray:
    """Reference implementation of LoRA merging.

    Shapes follow the paper: W0 (d, k), B (d, r), A (r, k).
    """
    if b.shape[1] != rank or a.shape[0] != rank:
        raise ValueError("rank does not match adapter shapes")
    return w0 + (alpha / rank) * (b @ a)


def fuse_adapters_to_base(
    adapter_dir: str | None = None, save_path: str | None = None, dequantize: bool = False
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.fuse",
        "--model",
        settings.base_model,
        "--adapter-path",
        adapter_dir or settings.adapter_output_dir,
        "--save-path",
        save_path or settings.fused_output_dir,
    ]
    if dequantize:
        cmd.append("--dequantize")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse LoRA adapters into the base model")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--dequantize", action="store_true", help="Export a de-quantized model")
    args = parser.parse_args()
    fuse_adapters_to_base(args.adapter_dir, args.save_path, args.dequantize)


if __name__ == "__main__":
    main()
