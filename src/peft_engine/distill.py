"""Sequence-level distillation: a large teacher writes answers, the student trains on them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from peft_engine.config import settings
from peft_engine.dataset import build_chatml


def read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def split_counts(num_prompts: int, train_frac: float = 0.9) -> tuple[int, int]:
    n_train = int(num_prompts * train_frac)
    return n_train, num_prompts - n_train


def generate_split(
    model: object,
    tokenizer: object,
    prompts: list[dict[str, str]],
    out_path: Path,
    max_tokens: int,
    temperature: float,
) -> None:
    """Generate teacher answers, appending to `out_path`. Resumes where a previous run stopped."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = count_lines(out_path)
    sampler = make_sampler(temp=temperature)
    with out_path.open("a", encoding="utf-8") as f:
        for i, p in enumerate(prompts[done:], start=done):
            messages = [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"]},
            ]
            chat_prompt = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                messages, add_generation_prompt=True, tokenize=False
            )
            answer = generate(
                model, tokenizer, prompt=chat_prompt, max_tokens=max_tokens, sampler=sampler
            ).strip()
            text = build_chatml(p["system"], p["user"], answer)
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{out_path.name}] {i + 1}/{len(prompts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher outputs for distillation")
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--teacher", default=settings.teacher_model)
    parser.add_argument("--data-dir", default=settings.data_dir)
    parser.add_argument("--output-dir", default=settings.distilled_dir)
    args = parser.parse_args()

    from mlx_lm import load

    data_dir, out_dir = Path(args.data_dir), Path(args.output_dir)
    n_train, n_valid = split_counts(args.num_prompts)
    train_prompts = read_jsonl(data_dir / "prompts_train.jsonl")[:n_train]
    valid_prompts = read_jsonl(data_dir / "prompts_valid.jsonl")[:n_valid]

    print(f"Loading teacher {args.teacher}")
    model, tokenizer = load(args.teacher)
    generate_split(
        model, tokenizer, train_prompts, out_dir / "train.jsonl", args.max_tokens, args.temperature
    )
    generate_split(
        model, tokenizer, valid_prompts, out_dir / "valid.jsonl", args.max_tokens, args.temperature
    )
    print(f"Distilled dataset written to {out_dir}")


if __name__ == "__main__":
    main()
