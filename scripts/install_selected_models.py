#!/usr/bin/env python3
"""Instala o classificador final e configura a geração remota do GenLocalNews."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "model_selection.json"
CLASS_DEST = ROOT / "models" / "selected" / "classification" / "bundle"
DEUCALION_ADAPTER_DEST = ROOT / "models" / "selected" / "generation" / "deucalion" / "adapter"


def copy_directory(source: Path, destination: Path, force: bool) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Diretório inexistente: {source}")
    if destination.exists() and any(p.is_file() and p.name != ".gitkeep" for p in destination.rglob("*")):
        if not force:
            raise FileExistsError(f"{destination} já contém artefactos; use --force para substituir")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-bundle", required=True)
    parser.add_argument("--classification-base-model")
    parser.add_argument("--deucalion-adapter", help="Cópia opcional do LoRA para transporte/deployment")
    parser.add_argument("--remote-url", default="http://127.0.0.1:8000")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    copy_directory(Path(args.classification_bundle).expanduser().resolve(), CLASS_DEST, args.force)
    checkpoint = CLASS_DEST / "checkpoint"
    required = [checkpoint / "model_metadata.json", checkpoint / "model_state.pt"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise ValueError(f"Bundle de classificação incompleto: {missing}")

    metadata_path = checkpoint / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if args.classification_base_model:
        base = Path(args.classification_base_model).expanduser().resolve()
        if not base.is_dir():
            raise FileNotFoundError(f"Encoder/base inexistente: {base}")
        metadata.setdefault("model_config", {})["model_id"] = str(base)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.deucalion_adapter:
        adapter = Path(args.deucalion_adapter).expanduser().resolve()
        copy_directory(adapter, DEUCALION_ADAPTER_DEST, args.force)
        if not (DEUCALION_ADAPTER_DEST / "adapter_config.json").exists():
            raise ValueError("O adaptador copiado não contém adapter_config.json")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["classification"].update({
        "enabled": True,
        "bundle_path": str(CLASS_DEST.relative_to(ROOT)),
        "device": args.device,
        "batch_size": args.batch_size,
    })
    config["generation"].update({
        "enabled": True,
        "backend": "remote",
        "remote_url": args.remote_url,
        "model_key": "llama31_8b_4096_lora",
        "model_version": "llama31_8b_4096_lora",
        "base_model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "max_seq_length": 4096,
        "max_new_tokens": 450,
        "temperature": 0.0,
    })
    CONFIG.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Classificador instalado em: {CLASS_DEST}")
    if args.deucalion_adapter:
        print(f"Cópia do LoRA: {DEUCALION_ADAPTER_DEST}")
    print(f"Geração remota configurada em: {args.remote_url}")


if __name__ == "__main__":
    main()
