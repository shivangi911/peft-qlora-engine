#!/usr/bin/env bash
# Small end-to-end run of the PyTorch backend on CPU: base eval, LoRA on SlimOrca,
# distillation from a 1.5B teacher, LoRA on distilled data, final evaluation.
set -euo pipefail

PY="python -u -m peft_engine.hf_backend"
mkdir -p logs

python -m peft_engine.dataset --sample-size 300

$PY train --iters 150 --batch-size 1 --max-len 256 --adapter-dir models/hf_adapters/slimorca \
    2>&1 | tee logs/train_slimorca.log

$PY distill --num-prompts 40 --max-new-tokens 96 --output-dir data/distilled_hf \
    2>&1 | tee logs/distill.log

$PY train --iters 150 --batch-size 1 --max-len 256 --data-dir data/distilled_hf \
    --adapter-dir models/hf_adapters/distilled 2>&1 | tee logs/train_distilled.log

$PY eval --num-samples 30 --max-len 256 --gen-tokens 64 \
    --variant base= \
    --variant slimorca_ft=models/hf_adapters/slimorca \
    --variant distilled_ft=models/hf_adapters/distilled \
    2>&1 | tee logs/eval.log
