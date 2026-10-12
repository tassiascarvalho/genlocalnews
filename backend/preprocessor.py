"""Limpeza textual, atributos derivados e estatísticas dos segmentos."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from statistics import median
from typing import Any

from .dataset_loader import Segment

logger = logging.getLogger(__name__)


class Preprocessor:
    # Não há filtragem por tamanho.
    MIN_WORDS = 0
    MAX_WORDS = 10_000_000

    def process(self, segments: list[Segment]) -> list[Segment]:
        cleaned = [self._clean(s) for s in segments]
        enriched = [self._enrich(s) for s in cleaned]
        filtered = [s for s in enriched if self._is_valid(s)]
        removed = len(segments) - len(filtered)
        if removed:
            logger.info("Pré-processamento: %d segmento(s) removidos.", removed)
        return filtered

    def to_jsonl(self, segments: list[Segment], output_path: Path) -> Path:
        output_path.write_text("\n".join(json.dumps(s.to_dict(), ensure_ascii=False) for s in segments), encoding="utf-8")
        return output_path

    def get_stats(self, segments: list[Segment]) -> dict[str, Any]:
        if not segments:
            return {}
        word_counts = [s.derived_features.get("word_count", len(s.full_text.split())) for s in segments]
        municipalities = sorted({s.municipality for s in segments if s.municipality})
        topic_freq: dict[str, int] = {}
        annotated_segments = 0
        reference_news = 0
        with_offsets = 0
        with_voting = 0

        for seg in segments:
            for topic in seg.topics:
                topic_freq[topic] = topic_freq.get(topic, 0) + 1
            if seg.annotations.criteria_scores or seg.annotations.newsworthiness_label or seg.annotations.adjudicated_label:
                annotated_segments += 1
            if seg.annotations.reference_title or seg.annotations.reference_lead or seg.annotations.reference_body:
                reference_news += 1
            if seg.segment_start is not None and seg.segment_end is not None:
                with_offsets += 1
            if seg.voting:
                with_voting += 1

        return {
            "total_segments": len(segments),
            "n_municipalities": len(municipalities),
            "municipalities": municipalities,
            "word_count_min": min(word_counts),
            "word_count_max": max(word_counts),
            "word_count_mean": round(sum(word_counts) / len(word_counts), 1),
            "word_count_median": median(word_counts),
            "top_topics": sorted(topic_freq, key=lambda k: (-topic_freq[k], k))[:10],
            "segments_with_offsets": with_offsets,
            "segments_with_voting": with_voting,
            "segments_with_annotations": annotated_segments,
            "segments_with_reference_news": reference_news,
        }

    def _clean(self, segment: Segment) -> Segment:
        segment.full_text = self._normalize_text(segment.full_text)
        segment.subject_text = self._normalize_text(segment.subject_text)
        segment.item_title = self._normalize_text(segment.item_title)
        segment.summary = self._normalize_text(segment.summary)
        return segment

    def _enrich(self, segment: Segment) -> Segment:
        txt = segment.operational_text
        routine_hits = self._count_patterns(txt.lower(), ["aprovação da ata", "ata n.º", "ata nº", "balancete", "não foram agendados assuntos", "ausência de assuntos", "expediente"])
        negative_hits = self._count_patterns(txt.lower(), ["encerramento", "incêndio", "acidente", "risco", "desemprego", "morte", "desastre", "contestação", "conflito", "denúncia"])
        institutional_hits = self._count_patterns(txt.lower(), ["presidente da câmara", "câmara municipal", "vereador", "junta de freguesia", "executivo"])
        local_hits = self._count_patterns(txt.lower(), [segment.municipality.lower() if segment.municipality else "", "freguesia", "bairro", "escola", "rua", "praça"])
        novelty_hits = self._count_patterns(txt.lower(), ["novo", "nova", "primeira vez", "criação", "inauguração", "lançamento", "mudança", "alteração"])
        segment.derived_features = {
            "word_count": len(txt.split()),
            "char_count": len(txt),
            "has_voting": bool(segment.voting),
            "has_offsets": segment.segment_start is not None and segment.segment_end is not None,
            "has_reference_news": bool(segment.annotations.reference_title or segment.annotations.reference_lead or segment.annotations.reference_body),
            "has_criteria_annotations": bool(segment.annotations.criteria_scores),
            "is_short_routine_candidate": self._looks_routine(segment),
            "routine_pattern_hits": routine_hits,
            "negative_signal_hits": negative_hits,
            "institutional_signal_hits": institutional_hits,
            "local_signal_hits": local_hits,
            "novelty_signal_hits": novelty_hits,
        }
        return segment

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
        return text.strip()

    def _is_valid(self, segment: Segment) -> bool:
        """Todos os segmentos são mantidos."""
        return True

    @staticmethod
    def _looks_routine(segment: Segment) -> bool:
        txt = f"{segment.item_title} {segment.subject_text} {segment.full_text}".lower()
        patterns = ["aprovação da ata", "ata n.º", "balancete", "não foram agendados assuntos", "ausência de assuntos"]
        return any(p in txt for p in patterns)

    @staticmethod
    def _count_patterns(text: str, patterns: list[str]) -> int:
        return sum(1 for p in patterns if p and p in text)
