"""
NURBGen Inference Script
Base model : Qwen/Qwen3-4B
LoRA adapter: SadilKhan/NURBGen (Hugging Face)

Supports:
  - Single text prompt  (--prompt "...")
  - Plain .txt file     (--input file.txt)
  - JSON file           (--input file.json)  rows: {"uid": "...", "caption": "..."}
                        uid is optional; falls back to "_".join(caption.split())
  - Batch processing    (--batch_size N)
  - Output saved to     --output_dir  (default: ./nurbgen_outputs)
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

# ── ms-swift imports ──────────────────────────────────────────────────────────
from swift.llm import (
    PtEngine,
    RequestConfig,
    get_template,
    load_dataset,
)
from swift.llm import InferRequest


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen3-4B"
LORA_ADAPTER = "SadilKhan/NURBGen"
DEFAULT_TEXT = "Generate NURBS for the following: "

# ─────────────────────────────────────────────────────────────────────────────
# Engine initialisation (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────
_engine = None

def get_engine(max_new_tokens: int = 8192):
    """Return a cached PtEngine with the LoRA adapter loaded."""
    global _engine
    if _engine is None:
        print(f"[NURBGen] Loading base model  : {BASE_MODEL}")
        print(f"[NURBGen] Loading LoRA adapter: {LORA_ADAPTER}")
        _engine = PtEngine(
            BASE_MODEL,
            adapters=[LORA_ADAPTER],
            use_hf=True,          # pull from Hugging Face
            max_model_len=max_new_tokens
        )
        print("[NURBGen] Engine ready.\n")
    return _engine


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def caption_to_uid(caption: str) -> str:
    """Fallback uid: join words with underscores, strip non-alphanumeric."""
    slug = "_".join(caption.strip().split())
    slug = re.sub(r"[^\w]", "_", slug)          # replace special chars
    slug = re.sub(r"_+", "_", slug).strip("_")  # collapse multiple underscores
    return slug[:128]                            # cap length


def build_requests(items: list[dict]) -> list[InferRequest]:
    """Convert list of {uid, caption} dicts into InferRequest objects."""
    requests = []
    for item in items:
        caption = item["caption"]
        requests.append(
            InferRequest(messages=[{"role": "user", "content": DEFAULT_TEXT + caption}])
        )
    return requests


def save_result(uid: str, caption: str, response: str, output_dir: Path):
    """Save a single inference result as a .json file named by uid."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{uid}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"uid": uid, "caption": caption, "response": response}, f)
    return out_file


# ─────────────────────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(
    items: list[dict],
    output_dir: Path,
    batch_size: int = 4,
    max_new_tokens: int = 8192,
    temperature: float = 0.3,
) -> list[dict]:
    """
    Run inference on a list of {uid, caption} dicts.
    Returns list of {uid, caption, response} dicts.
    """
    engine = get_engine(max_new_tokens=max_new_tokens, temperature=temperature)

    request_config = RequestConfig(
        max_tokens=max_new_tokens,
        temperature=temperature,
    )

    results = []
    total   = len(items)

    for batch_start in range(0, total, batch_size):
        batch = items[batch_start : batch_start + batch_size]
        batch_requests = build_requests(batch)

        print(
            f"[NURBGen] Processing batch {batch_start // batch_size + 1} "
            f"({batch_start + 1}–{min(batch_start + batch_size, total)} / {total}) …"
        )
        t0 = time.time()

        # Synchronous batched inference via ms-swift PtEngine
        responses = engine.infer(batch_requests, request_config=request_config)

        elapsed = time.time() - t0
        print(f"          Done in {elapsed:.1f}s")

        for item, resp in zip(batch, responses):
            text = resp.choices[0].message.content
            uid  = item.get("uid") or caption_to_uid(item["caption"])

            out_file = save_result(uid, item["caption"], text, output_dir)
            print(f"          → saved: {out_file}")

            results.append({"uid": uid, "caption": item["caption"], "response": text})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Input loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_txt(path: str) -> list[dict]:
    """One caption per non-empty line."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            caption = line.strip()
            if caption:
                items.append({"uid": None, "caption": DEFAULT_TEXT + caption})
    return items


def load_json(path: str) -> list[dict]:
    """
    Accepts:
      - A JSON array  : [{"uid": "...", "caption": "..."}, ...]
      - A JSON object : {"uid": "...", "caption": "..."}
      - JSONL         : one JSON object per line
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # Try standard JSON first
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
    except json.JSONDecodeError:
        # Try JSONL
        data = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                data.append(json.loads(line))

    items = []
    for entry in data:
        caption = entry.get("caption") or entry.get("text") or entry.get("prompt")
        if not caption:
            raise ValueError(f"Entry missing 'caption' field: {entry}")
        uid = entry.get("uid") or None
        items.append({"uid": uid, "caption": DEFAULT_TEXT + caption})
    return items


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="NURBGen inference — Qwen3-4B + LoRA (SadilKhan/NURBGen)"
    )

    # Input modes (mutually exclusive)
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument(
        "--prompt", "-p",
        type=str,
        help="Single text prompt (inline string).",
    )
    inp.add_argument(
        "--input", "-i",
        type=str,
        help="Path to a .txt or .json / .jsonl input file.",
    )

    # Output
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./nurbgen_outputs",
        help="Directory where result .json files are saved (default: ./nurbgen_outputs).",
    )

    # Generation settings
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate (default: 8192).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature; 0 = greedy (default: 0.3).",
    )

    # Batch
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of prompts to process in one engine call (default: 4).",
    )

    # Optional: save a consolidated JSON summary
    parser.add_argument(
        "--save_summary",
        action="store_true",
        help="Also save a results_summary.json with all outputs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    # ── Build item list ───────────────────────────────────────────────────────
    if args.prompt:
        items = [{"uid": None, "caption": DEFAULT_TEXT + args.prompt}]

    else:
        path = args.input
        ext  = Path(path).suffix.lower()

        if ext == ".txt":
            items = load_txt(path)
        elif ext in (".json", ".jsonl"):
            items = load_json(path)
        else:
            # Try JSON first, fall back to plain text
            try:
                items = load_json(path)
            except Exception:
                items = load_txt(path)

        if not items:
            print("[NURBGen] No items found in input file. Exiting.")
            return

    print(f"[NURBGen] {len(items)} prompt(s) to process.")
    print(f"[NURBGen] Output dir : {output_dir.resolve()}\n")

    # ── Run ───────────────────────────────────────────────────────────────────
    results = run_inference(
        items,
        output_dir      = output_dir,
        batch_size      = args.batch_size,
        max_new_tokens  = args.max_new_tokens,
        temperature     = args.temperature,
    )

    # ── Optional summary ──────────────────────────────────────────────────────
    if args.save_summary:
        summary_path = output_dir / "results_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[NURBGen] Summary saved → {summary_path}")

    print(f"\n[NURBGen] ✓ Finished. {len(results)} result(s) written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()