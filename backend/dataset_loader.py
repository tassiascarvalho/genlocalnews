from __future__ import annotations

"""Normalização dos segmentos previamente extraídos das atas municipais.

São preservados o texto, os identificadores e os metadados necessários à
classificação, à geração e à verificação editorial da fonte.
"""

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.newsworthiness.iip_calculator import IIPCalculator

logger = logging.getLogger(__name__)


@dataclass
class AnnotationMetadata:
    """Campos opcionais de anotação."""
    annotator_id: str = ""
    annotation_round: str = ""
    annotation_source: str = ""
    adjudicated_label: str = ""
    newsworthiness_label: str = ""
    interest_public_label: str = ""
    reference_title: str = ""
    reference_lead: str = ""
    reference_body: str = ""
    reference_news_id: str = ""
    criteria_scores: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Segment:
    # identificação documental
    segment_id: str
    municipality: str
    minute_id: str
    date: str
    item_id: int | None = None
    item_title: str = ""

    # conteúdo-base
    subject_text: str = ""
    full_text: str = ""
    summary: str = ""
    theme: str = ""
    topics: list[str] = field(default_factory=list)
    voting: list[dict[str, Any]] = field(default_factory=list)

    # offsets / rastreabilidade
    segment_start: int | None = None
    segment_end: int | None = None
    subject_start: int | None = None
    subject_end: int | None = None
    full_minute_text: str = ""
    source_filename: str = ""

    # metadados adicionais do documento
    metadata: dict[str, Any] = field(default_factory=dict)
    personal_info: list[dict[str, Any]] = field(default_factory=list)

    # anotações
    annotations: AnnotationMetadata = field(default_factory=AnnotationMetadata)

    # campos calculados
    iip_scores: dict[str, int] = field(default_factory=dict)
    iip_total: float = 0.0
    iip_level: str = ""
    is_newsworthy: bool = False
    binary_probability: float | None = None
    binary_threshold: float | None = None
    iip_direct: float | None = None
    iip_derived: float | None = None
    priority_rank: int | None = None
    model_version: str = ""
    generation_model_version: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    generated_news: dict[str, Any] = field(default_factory=dict)
    derived_features: dict[str, Any] = field(default_factory=dict)

    @property
    def operational_text(self) -> str:
        if self.subject_text and self.subject_text not in self.full_text:
            return f"{self.subject_text}\n\n{self.full_text}".strip()
        return self.full_text.strip()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


class DatasetLoader:
    """Carrega JSON/JSONL e normaliza a estrutura GenLocalNews."""

    def load(self, path: str | Path) -> list[Segment]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Ficheiro não encontrado: {path}")

        if path.suffix.lower() == ".jsonl":
            return self._load_jsonl(path)
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            return self._parse_citilink(raw, source_filename=path.name)
        if path.suffix.lower() == ".csv":
            return self._load_csv(path)
        raise ValueError(f"Formato não suportado: {path.suffix}. Use .json, .jsonl ou .csv")


    def _load_csv(self, path: Path) -> list[Segment]:
        """Carrega segmentos a partir de CSV, com ou sem colunas de anotação."""
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        segments = self._segments_from_csv_like_rows(rows, source_filename=path.name)
        logger.info("CSV carregado: %d segmentos", len(segments))
        return segments

    def _segments_from_csv_like_rows(self, rows: list[dict[str, Any]], *, source_filename: str) -> list[Segment]:
        segments: list[Segment] = []
        for i, row in enumerate(rows, start=1):
            text = (
                row.get("text")
                or row.get("texto_segmento")
                or row.get("segment_text")
                or row.get("full_text")
                or ""
            ).strip()
            row_source_file = (row.get("source_file") or source_filename).strip()
            begin = self._safe_int(row.get("begin"))
            end = self._safe_int(row.get("end"))
            criteria_scores = self._extract_csv_criteria(row)
            iip_result = IIPCalculator.compute(criteria_scores, require_all=True)

            explicit_has_values = self._parse_nullable_bool(row.get("has_newsworthiness_values"))
            has_values = bool(explicit_has_values) if explicit_has_values is not None else iip_result.has_all_criteria

            # Decisão binária anotada (independente do IIP).
            label_source = "none"
            label_value = self._parse_nullable_bool(
                row.get("noticioso_manual")
                or row.get("label_manual")
                or row.get("manual_label")
                or row.get("is_newsworthy_manual")
            )
            if label_value is not None:
                label_source = "manual_editorial_annotation"
            else:
                legacy_label = self._parse_nullable_bool(row.get("label"))
                if legacy_label is None:
                    legacy_label = self._parse_nullable_bool(row.get("label_28"))
                if legacy_label is not None:
                    label_value = legacy_label
                    label_source = "legacy_or_threshold_label"
                else:
                    label_value = False

            ann = AnnotationMetadata(
                annotation_source=str(row.get("annotation_type", "") or ""),
                newsworthiness_label=str(row.get("label", row.get("label_28", "")) or ""),
                criteria_scores=criteria_scores,
                extra={
                    k: v for k, v in row.items()
                    if k not in {
                        "source_file", "municipality", "minute_id", "date", "begin", "end", "text",
                        "annotation_type", "annotation_begin", "annotation_end", "annotation_text", "tema",
                        "relevancia", "proximidade", "novidade", "atualidade", "continuidade",
                        "notoriedade", "negatividade", "iip_total", "iip_raw_sum",
                        "iip_weighted_0_1", "label", "label_28", "noticioso_manual",
                        "label_manual", "manual_label", "is_newsworthy_manual", "has_newsworthiness_values"
                    }
                },
            )

            # IIP só é calculado com os sete valores-notícia; caso contrário fica None.
            iip_total = iip_result.iip_total
            iip_level = iip_result.iip_level

            segment = Segment(
                segment_id=(row.get("segment_id") or self._csv_segment_id(row_source_file, begin, end, i)).strip(),
                municipality=(row.get("municipality") or "Desconhecido").strip(),
                minute_id=(row.get("minute_id") or self._extract_minute_id_from_source(row_source_file)).strip(),
                date=(row.get("date") or self._extract_date_from_source(row_source_file)).strip(),
                subject_text=(row.get("tema") or row.get("subject_text") or "").strip(),
                full_text=text,
                segment_start=begin,
                segment_end=end,
                source_filename=row_source_file,
                annotations=ann,
                iip_scores=criteria_scores,
                iip_total=iip_total if iip_total is not None else 0.0,
                iip_level=iip_level,
                is_newsworthy=bool(label_value),
                metadata={
                    "source_file": row_source_file,
                    "annotation_type": row.get("annotation_type", ""),
                    "has_newsworthiness_values": has_values,
                    "label_source": label_source,
                    "has_manual_label": label_source == "manual_editorial_annotation",
                    "annotation_text": row.get("annotation_text", ""),
                    "iip_raw_sum": row.get("iip_raw_sum", ""),
                    "presidente_camara": row.get("presidente_camara", ""),
                    "vice_presidente_camara": row.get("vice_presidente_camara", ""),
                    "presidente_reuniao": row.get("presidente_reuniao", ""),
                },
            )
            segments.append(segment)
        return segments

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            if value is None or str(value).strip() == "":
                return None
            return int(float(str(value).strip()))
        except Exception:
            return None

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            if value is None or str(value).strip() == "":
                return 0.0
            return float(str(value).replace(",", ".").strip())
        except Exception:
            return 0.0

    @staticmethod
    def _parse_bool_label(value: Any) -> bool:
        parsed = DatasetLoader._parse_nullable_bool(value)
        return bool(parsed) if parsed is not None else False

    @staticmethod
    def _parse_nullable_bool(value: Any) -> bool | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text == "":
            return None
        text = text.replace("ã", "a").replace("á", "a").replace("ç", "c")
        positives = {"1", "true", "sim", "yes", "y", "noticioso", "newsworthy", "positivo"}
        negatives = {"0", "false", "nao", "não", "no", "n", "rotineiro", "nao noticioso", "não noticioso", "negativo"}
        if text in positives:
            return True
        if text in negatives:
            return False
        return None

    @staticmethod
    def _extract_csv_criteria(row: dict[str, Any]) -> dict[str, int]:
        scores: dict[str, int] = {}
        for key in ("relevancia", "proximidade", "novidade", "atualidade", "continuidade", "notoriedade", "negatividade"):
            try:
                raw = row.get(key)
                if raw is not None and str(raw).strip() != "":
                    scores[key] = int(float(str(raw).replace(",", ".").strip()))
            except Exception:
                pass
        return scores

    @staticmethod
    def _csv_segment_id(source_file: str, begin: int | None, end: int | None, row_number: int) -> str:
        base = re.sub(r"[^A-Za-z0-9_-]+", "_", source_file).strip("_")[:120]
        if begin is not None and end is not None:
            return f"{base}_{begin}_{end}"
        return f"{base}_row_{row_number}"

    @staticmethod
    def _extract_date_from_source(source_file: str) -> str:
        match = re.search(r"(20\d{2}-\d{2}-\d{2})", source_file or "")
        return match.group(1) if match else ""

    @staticmethod
    def _extract_minute_id_from_source(source_file: str) -> str:
        text = source_file or ""
        # Ex.: curation/Alandroal_cm_001_2024-01-03.json/CURATION_VALUE/...
        match = re.search(r"([^/\\]+\.json)", text)
        if match:
            return match.group(1).replace(".json", "")
        return Path(text).stem if text else ""

    def _load_jsonl(self, path: Path) -> list[Segment]:
        segments: list[Segment] = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ann = self._extract_annotations(obj)
            seg = Segment(
                segment_id=obj.get("segment_id", f"jsonl_{i}"),
                municipality=obj.get("municipality", "Desconhecido"),
                minute_id=obj.get("minute_id", ""),
                date=obj.get("date", ""),
                item_id=obj.get("item_id"),
                item_title=obj.get("item_title", ""),
                subject_text=obj.get("subject") or obj.get("subject_text", ""),
                full_text=obj.get("text") or obj.get("full_text", ""),
                summary=obj.get("summary", ""),
                topics=obj.get("topics", []),
                theme=obj.get("theme", ""),
                voting=obj.get("voting", []),
                segment_start=obj.get("start"),
                segment_end=obj.get("end"),
                subject_start=obj.get("subject_start"),
                subject_end=obj.get("subject_end"),
                source_filename=path.name,
                metadata={
                    **(obj.get("metadata", {}) if isinstance(obj.get("metadata", {}), dict) else {}),
                    "presidente_camara": obj.get("presidente_camara") or obj.get("presidente_da_camara") or obj.get("presidente"),
                    "vice_presidente_camara": obj.get("vice_presidente_camara") or obj.get("vice_presidente_da_camara") or obj.get("vice_presidente"),
                },
                annotations=ann,
            )
            segments.append(seg)
        logger.info("JSONL carregado: %d segmentos", len(segments))
        return segments

    def _parse_citilink(self, raw: dict | list, source_filename: str) -> list[Segment]:
        segments: list[Segment] = []
        municipalities = self._normalize_root(raw, source_filename)

        for mun_obj in municipalities:
            municipality = str(mun_obj.get("municipality", Path(source_filename).stem))
            for minute in mun_obj.get("minutes", []):
                minute_id = str(minute.get("minute_id", ""))
                date = self._extract_date(minute)
                minute_metadata = minute.get("metadata", {}) if isinstance(minute.get("metadata", {}), dict) else {}
                minute_personal_info = minute.get("personal_info", []) if isinstance(minute.get("personal_info", []), list) else []
                full_minute_text = minute.get("full_text", "")

                for item in minute.get("agenda_items", []):
                    item_id = item.get("item_id")
                    item_title = item.get("item_title", "")
                    for subj in item.get("subjects", []):
                        segments.append(
                            self._build_segment(
                                subj=subj,
                                municipality=municipality,
                                minute_id=minute_id,
                                date=date,
                                item_id=item_id,
                                item_title=item_title,
                                minute_metadata=minute_metadata,
                                minute_personal_info=minute_personal_info,
                                full_minute_text=full_minute_text,
                                source_filename=source_filename,
                            )
                        )

        logger.info("JSON GenLocalNews carregado: %d segmentos", len(segments))
        return segments

    def _normalize_root(self, raw: dict | list, source_filename: str) -> list[dict[str, Any]]:
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict) and "municipalities" in raw:
            return raw["municipalities"]
        if isinstance(raw, dict) and "municipality" in raw and "minutes" in raw:
            return [raw]
        logger.warning("Estrutura raiz não reconhecida em '%s'; a encapsular de forma defensiva.", source_filename)
        return [{"municipality": Path(source_filename).stem, "minutes": raw.get("minutes", []) if isinstance(raw, dict) else []}]

    def _build_segment(
        self,
        subj: dict[str, Any],
        municipality: str,
        minute_id: str,
        date: str,
        item_id: int | None,
        item_title: str,
        minute_metadata: dict[str, Any],
        minute_personal_info: list[dict[str, Any]],
        full_minute_text: str,
        source_filename: str,
    ) -> Segment:
        subject_info = subj.get("subject", {})
        subject_text = subject_info.get("text", "") if isinstance(subject_info, dict) else str(subject_info)
        ann = self._extract_annotations(subj)
        return Segment(
            segment_id=subj.get("subject_id", ""),
            municipality=municipality,
            minute_id=minute_id,
            date=date,
            item_id=item_id,
            item_title=item_title,
            subject_text=subject_text,
            full_text=subj.get("text", ""),
            summary=subj.get("summary", ""),
            topics=subj.get("topics", []) if isinstance(subj.get("topics", []), list) else [],
            theme=subj.get("theme", ""),
            voting=subj.get("voting", []) if isinstance(subj.get("voting", []), list) else [],
            segment_start=subj.get("start"),
            segment_end=subj.get("end"),
            subject_start=subject_info.get("start") if isinstance(subject_info, dict) else None,
            subject_end=subject_info.get("end") if isinstance(subject_info, dict) else None,
            full_minute_text=full_minute_text,
            source_filename=source_filename,
            metadata=minute_metadata,
            personal_info=minute_personal_info,
            annotations=ann,
        )

    def _extract_annotations(self, obj: dict[str, Any]) -> AnnotationMetadata:
        # Aceita os critérios agrupados num objeto ou como colunas separadas.
        ann_obj = obj.get("annotations", {}) if isinstance(obj.get("annotations", {}), dict) else {}

        criteria = {
            "relevancia", "proximidade", "novidade", "atualidade",
            "continuidade", "notoriedade", "negatividade"
        }
        criteria_scores: dict[str, int] = {}

        for key in criteria:
            if key in ann_obj:
                criteria_scores[key] = int(ann_obj[key])
            elif key in obj:
                try:
                    criteria_scores[key] = int(obj[key])
                except Exception:
                    pass

        known_keys = {
            "annotator_id", "annotation_round", "annotation_source", "adjudicated_label",
            "newsworthiness_label", "interest_public_label", "reference_title", "reference_lead",
            "reference_body", "reference_news_id"
        }

        payload = {}
        for key in known_keys:
            if key in ann_obj:
                payload[key] = ann_obj.get(key, "")
            elif key in obj:
                payload[key] = obj.get(key, "")

        extra = {}
        for source in (ann_obj, obj):
            for k, v in source.items():
                if k not in known_keys and k not in criteria and k not in {
                    "annotations", "subject", "subject_id", "text", "start", "end", "topics", "theme",
                    "summary", "voting"
                }:
                    extra.setdefault(k, v)

        return AnnotationMetadata(
            annotator_id=str(payload.get("annotator_id", "") or ""),
            annotation_round=str(payload.get("annotation_round", "") or ""),
            annotation_source=str(payload.get("annotation_source", "") or ""),
            adjudicated_label=str(payload.get("adjudicated_label", "") or ""),
            newsworthiness_label=str(payload.get("newsworthiness_label", "") or ""),
            interest_public_label=str(payload.get("interest_public_label", "") or ""),
            reference_title=str(payload.get("reference_title", "") or ""),
            reference_lead=str(payload.get("reference_lead", "") or ""),
            reference_body=str(payload.get("reference_body", "") or ""),
            reference_news_id=str(payload.get("reference_news_id", "") or ""),
            criteria_scores=criteria_scores,
            extra=extra,
        )

    def _extract_date(self, minute: dict[str, Any]) -> str:
        meta = minute.get("metadata", {}) if isinstance(minute.get("metadata", {}), dict) else {}
        for key in ("date", "session_date", "data"):
            if key in meta:
                return str(meta[key])
        mid = str(minute.get("minute_id", ""))
        for part in mid.split("_"):
            if len(part) == 10 and part[4] == "-" and part[7] == "-":
                return part
        return ""
