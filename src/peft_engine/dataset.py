"""SlimOrca ingestion and ChatML formatting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from peft_engine.config import settings

DATASET_NAME = "Open-Orca/SlimOrca"
DEFAULT_SYSTEM = "You are an expert AI assistant that thinks step-by-step."

_ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}


def format_to_chatml(conversation: list[dict[str, str]]) -> str:
    """Convert a SlimOrca conversation (`from`/`value` turns) into a ChatML string."""
    out = ""
    for turn in conversation:
        role = _ROLE_MAP.get(turn["from"])
        if role is None:
            continue
        out += f"<|im_start|>{role}\n{turn['value']}<|im_end|>\n"
    return out


def extract_prompt(conversation: list[dict[str, str]]) -> dict[str, str] | None:
    """Return {"system", "user"} from the first system/human turns, or None if no human turn."""
    system = DEFAULT_SYSTEM
    for turn in conversation:
        if turn["from"] == "system" and turn["value"].strip():
            system = turn["value"]
        elif turn["from"] == "human":
            return {"system": system, "user": turn["value"]}
    return None


def build_chatml(system: str, user: str, assistant: str) -> str:
    return format_to_chatml(
        [
            {"from": "system", "value": system},
            {"from": "human", "value": user},
            {"from": "gpt", "value": assistant},
        ]
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(rows: list[Any], train_frac: float = 0.9) -> tuple[list[Any], list[Any]]:
    idx = int(len(rows) * train_frac)
    return rows[:idx], rows[idx:]


def prepare_and_export_dataset(output_dir: str, sample_size: int = 5000) -> None:
    """Stream `sample_size` SlimOrca rows and write train/valid ChatML JSONL plus prompt files."""
    from datasets import load_dataset

    out = Path(output_dir)
    stream = load_dataset(DATASET_NAME, split="train", streaming=True).take(sample_size)

    texts: list[dict[str, str]] = []
    prompts: list[dict[str, str]] = []
    for item in stream:
        conv = item["conversations"]
        prompt = extract_prompt(conv)
        if prompt is None:
            continue
        texts.append({"text": format_to_chatml(conv)})
        prompts.append(prompt)

    train, valid = split_rows(texts)
    p_train, p_valid = split_rows(prompts)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "valid.jsonl", valid)
    write_jsonl(out / "prompts_train.jsonl", p_train)
    write_jsonl(out / "prompts_valid.jsonl", p_valid)
    print(f"Wrote {len(train)} train / {len(valid)} valid examples to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SlimOrca as ChatML JSONL")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--output-dir", default=settings.data_dir)
    args = parser.parse_args()
    prepare_and_export_dataset(args.output_dir, args.sample_size)


if __name__ == "__main__":
    main()
