"""Utilitários de execução e persistência dos resultados."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from .dataset_loader import Segment


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def export_results(segments: list[Segment], output_dir: Path, run_id: str | None = None, **metadata) -> Path:
    run_id = run_id or generate_run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"results_{run_id}.json"
    payload = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "total": len(segments),
        "newsworthy": sum(segment.is_newsworthy for segment in segments),
        **metadata,
        "segments": [segment.to_dict() for segment in segments],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
