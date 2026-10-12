from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .registry import SelectedModelRegistry


NEWS_VALUES = (
    "relevancia", "proximidade", "novidade", "atualidade",
    "continuidade", "notoriedade", "negatividade",
)


class SelectedClassificationService:
    """Serviço de classificação da noticiabilidade."""

    def __init__(self, registry: SelectedModelRegistry | None = None):
        self.registry = registry or SelectedModelRegistry()

    @staticmethod
    def _frame(segments: Iterable[Any]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "segment_id": segment.segment_id,
                    "minute_id": segment.minute_id or "unassigned",
                    "text": segment.operational_text,
                }
                for segment in segments
            ]
        )

    def predict(self, segments: list[Any]) -> list[dict[str, Any]]:
        cfg = self.registry.require_ready("classification")
        bundle = Path(cfg["bundle_path"])
        from .vendor_classification.predictor import predict_transformer

        frame = self._frame(segments)
        checkpoint = bundle / "checkpoint" if (bundle / "checkpoint").exists() else bundle
        predicted = predict_transformer(
            frame, checkpoint, batch_size=int(cfg.get("batch_size", 8)), device=cfg.get("device")
        )
        records = predicted.to_dict(orient="records")
        self._validate(records, {segment.segment_id for segment in segments})
        return records

    @staticmethod
    def _validate(records: list[dict[str, Any]], expected_ids: set[str]) -> None:
        returned = {str(record.get("segment_id", "")) for record in records}
        if returned != expected_ids:
            missing = sorted(expected_ids - returned)
            extra = sorted(returned - expected_ids)
            raise ValueError(f"Contrato de classificação desalinhado; missing={missing}, extra={extra}")
        required = {
            "segment_id", "minute_id", "newsworthy", "binary_probability",
            "binary_threshold", "news_values", "iip_direct", "iip_derived",
            "priority_rank", "model_version",
        }
        for record in records:
            absent = required - set(record)
            if absent:
                raise ValueError(f"Previsão {record.get('segment_id')} sem campos {sorted(absent)}")
            if set(record["news_values"]) != set(NEWS_VALUES):
                raise ValueError(f"Valores-notícia incompletos em {record.get('segment_id')}")

    @staticmethod
    def apply(segments: list[Any], records: list[dict[str, Any]]) -> list[Any]:
        by_id = {str(record["segment_id"]): record for record in records}
        for segment in segments:
            record = by_id[segment.segment_id]
            segment.is_newsworthy = bool(record["newsworthy"])
            segment.binary_probability = float(record["binary_probability"])
            segment.binary_threshold = float(record["binary_threshold"])
            segment.iip_scores = {key: int(value) for key, value in record["news_values"].items()}
            segment.iip_direct = float(record["iip_direct"])
            segment.iip_derived = float(record["iip_derived"])
            segment.iip_total = segment.iip_direct
            segment.iip_level = (
                "alto" if segment.iip_total >= 0.62 else
                "medio" if segment.iip_total >= 0.45 else "baixo"
            )
            segment.priority_rank = int(record["priority_rank"])
            segment.model_version = str(record["model_version"])
            segment.provenance["classification"] = {
                "model_version": segment.model_version,
                "binary_threshold": segment.binary_threshold,
                "decision_source": "binary_head",
                "ranking_source": "iip_direct_descending_within_minute",
            }
        return sorted(segments, key=lambda item: (item.minute_id, item.priority_rank, item.segment_id))
