"""PyTorch / Hugging Face backend (CPU, CUDA or Apple MPS).

Mirrors the MLX pipeline so the project can run on Linux/Windows or a free Colab GPU:
train (LoRA, or QLoRA with NF4 + double quantization on CUDA), distill, evaluate, merge.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import resource
import time
from pathlib import Path
from typing import Any

from peft_engine.config import settings
from peft_engine.dataset import build_chatml
from peft_engine.distill import count_lines, read_jsonl, split_counts
from peft_engine.evaluator import merge_metrics, parse_variant, perplexity_from_nll

DEFAULT_STUDENT = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TEACHER = "Qwen/Qwen2.5-1.5B-Instruct"


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def peak_rss_gb() -> float:
    """Peak resident memory of this process (Linux reports KiB, macOS bytes)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if platform.system() == "Linux" else 1024**3)


def load_texts(path: Path) -> list[str]:
    return [row["text"] for row in read_jsonl(path)]


def load_model(name: str, adapter_dir: str | None = None, quant4: bool = False) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs: dict[str, Any] = {}
    if quant4:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": 0}
    if not quant4:
        # fp32 on CPU/MPS: bf16 matmuls are extremely slow on CPUs without native support
        kwargs["dtype"] = torch.bfloat16 if pick_device() == "cuda" else torch.float32
    model: Any = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if not quant4:
        model.to(pick_device())
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    return model, tok


def train(
    model_name: str,
    data_dir: str,
    adapter_dir: str,
    iters: int,
    batch_size: int,
    lr: float,
    rank: int,
    alpha: float,
    dropout: float,
    max_len: int,
    quant4: bool,
    seed: int = 0,
) -> list[dict[str, float]]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    random.seed(seed)
    torch.manual_seed(seed)
    model, tok = load_model(model_name, quant4=quant4)
    if quant4:
        model = prepare_model_for_kbit_training(model)
    cfg = LoraConfig(
        r=rank,
        lora_alpha=int(alpha),
        lora_dropout=dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    device = next(model.parameters()).device

    texts = load_texts(Path(data_dir) / "train.jsonl")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    log: list[dict[str, float]] = []
    order: list[int] = []
    start = time.time()
    for step in range(1, iters + 1):
        if len(order) < batch_size:
            order = random.sample(range(len(texts)), len(texts))
        idx, order = order[:batch_size], order[batch_size:]
        enc = tok(
            [texts[i] for i in idx],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        loss = model(**enc, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        log.append({"step": step, "loss": float(loss.item()), "elapsed_s": time.time() - start})
        if step % 10 == 0 or step == iters:
            print(f"step {step}/{iters} loss {loss.item():.4f}")

    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    (Path(adapter_dir) / "train_log.json").write_text(json.dumps(log, indent=2))
    return log


def distill(
    teacher: str, data_dir: str, out_dir: str, num_prompts: int, max_new_tokens: int
) -> None:
    import torch

    model, tok = load_model(teacher)
    model.eval()
    device = next(model.parameters()).device
    n_train, n_valid = split_counts(num_prompts)
    for split, n in (("train", n_train), ("valid", n_valid)):
        prompts = read_jsonl(Path(data_dir) / f"prompts_{split}.jsonl")[:n]
        out = Path(out_dir) / f"{split}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = count_lines(out)
        with out.open("a", encoding="utf-8") as f:
            for i, p in enumerate(prompts[done:], start=done):
                chat = tok.apply_chat_template(
                    [
                        {"role": "system", "content": p["system"]},
                        {"role": "user", "content": p["user"]},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                enc = tok(chat, return_tensors="pt", truncation=True, max_length=768).to(device)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
                answer = tok.decode(gen[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
                text = build_chatml(p["system"], p["user"], answer.strip())
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{split}] {i + 1}/{len(prompts)}")


def evaluate(
    model_name: str,
    adapter_dir: str | None,
    valid_file: Path,
    num_samples: int,
    max_len: int,
    gen_tokens: int,
    quant4: bool,
) -> dict[str, Any]:
    import torch

    model, tok = load_model(model_name, adapter_dir, quant4)
    model.eval()
    device = next(model.parameters()).device
    nll, ntok = 0.0, 0
    with torch.no_grad():
        for text in load_texts(valid_file)[:num_samples]:
            ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            n = ids["input_ids"].shape[1] - 1
            if n < 1:
                continue
            loss = model(**ids, labels=ids["input_ids"]).loss  # mean over n shifted tokens
            nll += float(loss.item()) * n
            ntok += n

        prompt = tok.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "Explain the difference between QuickSort and MergeSort.",
                }
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = tok(prompt, return_tensors="pt").to(device)
        start = time.time()
        out = model.generate(**enc, max_new_tokens=gen_tokens, min_new_tokens=gen_tokens)
        elapsed = time.time() - start
    new_tokens = out.shape[1] - enc["input_ids"].shape[1]

    return {
        "model": model_name,
        "adapter": adapter_dir,
        "backend": "pytorch",
        "device": str(device),
        "perplexity": round(perplexity_from_nll(nll, ntok), 4),
        "eval_tokens": ntok,
        "peak_memory_gb": round(peak_rss_gb(), 2),
        "generation_tokens_per_sec": round(new_tokens / elapsed, 2),
        "machine": platform.platform(),
    }


def merge(model_name: str, adapter_dir: str, save_path: str) -> None:
    model, tok = load_model(model_name, adapter_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(save_path)
    tok.save_pretrained(save_path)
    print(f"Merged model saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch backend for the PEFT engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--model", default=DEFAULT_STUDENT)
    t.add_argument("--data-dir", default=settings.data_dir)
    t.add_argument("--adapter-dir", default="models/hf_adapters/slimorca")
    t.add_argument("--iters", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=2)
    t.add_argument("--lr", type=float, default=settings.learning_rate)
    t.add_argument("--rank", type=int, default=settings.lora_rank)
    t.add_argument("--alpha", type=float, default=settings.lora_alpha)
    t.add_argument("--dropout", type=float, default=settings.lora_dropout)
    t.add_argument("--max-len", type=int, default=512)
    t.add_argument("--quant4", action="store_true", help="NF4 + double quant (CUDA only)")

    d = sub.add_parser("distill")
    d.add_argument("--teacher", default=DEFAULT_TEACHER)
    d.add_argument("--data-dir", default=settings.data_dir)
    d.add_argument("--output-dir", default="data/distilled_hf")
    d.add_argument("--num-prompts", type=int, default=100)
    d.add_argument("--max-new-tokens", type=int, default=128)

    e = sub.add_parser("eval")
    e.add_argument("--model", default=DEFAULT_STUDENT)
    e.add_argument("--variant", action="append", required=True, help="name=adapter_dir or name=")
    e.add_argument("--valid-file", default=f"{settings.data_dir}/valid.jsonl")
    e.add_argument("--num-samples", type=int, default=50)
    e.add_argument("--max-len", type=int, default=512)
    e.add_argument("--gen-tokens", type=int, default=64)
    e.add_argument("--quant4", action="store_true")
    e.add_argument("--output", default=f"{settings.results_dir}/metrics_pytorch.json")

    m = sub.add_parser("merge")
    m.add_argument("--model", default=DEFAULT_STUDENT)
    m.add_argument("--adapter-dir", required=True)
    m.add_argument("--save-path", default="models/hf_fused_model")

    a = parser.parse_args()
    if a.cmd == "train":
        train(
            a.model, a.data_dir, a.adapter_dir, a.iters, a.batch_size, a.lr,
            a.rank, a.alpha, a.dropout, a.max_len, a.quant4,
        )  # fmt: skip
    elif a.cmd == "distill":
        distill(a.teacher, a.data_dir, a.output_dir, a.num_prompts, a.max_new_tokens)
    elif a.cmd == "eval":
        for spec in a.variant:
            name, adapter = parse_variant(spec)
            print(f"Evaluating {name}")
            metrics = evaluate(
                a.model, adapter, Path(a.valid_file), a.num_samples, a.max_len, a.gen_tokens,
                a.quant4,
            )  # fmt: skip
            merge_metrics(Path(a.output), name, metrics)
            print(json.dumps(metrics, indent=2))
    else:
        merge(a.model, a.adapter_dir, a.save_path)


if __name__ == "__main__":
    main()
