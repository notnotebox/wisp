#!/usr/bin/env python3
"""Merge the Wisp v0.1.2 4B PEFT adapter into the pinned Qwen3.5-4B base for Candle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    marker = args.output / "wisp_candle_manifest.json"
    if marker.exists() and not args.force:
        print(json.dumps({"status": "already_prepared", "output": str(args.output)}))
        return
    args.output.mkdir(parents=True, exist_ok=True)

    base = Qwen3_5ForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B",
        revision=REVISION,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    model = model.merge_and_unload(safe_merge=True, progressbar=True)
    model.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", revision=REVISION).save_pretrained(args.output)
    (args.output / "wisp_revision.txt").write_text(REVISION + "\n", encoding="utf-8")
    files = sorted(args.output.glob("*.safetensors"))
    manifest = {
        "status": "complete",
        "base_model": "Qwen/Qwen3.5-4B",
        "base_revision": REVISION,
        "adapter": str(args.adapter.resolve()),
        "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "merge_dtype": "bfloat16",
        "weights": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ],
    }
    marker.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output), "shards": len(files)}))


if __name__ == "__main__":
    main()

