"""API: carregamento, inferência, revisão e exportação."""
from __future__ import annotations

import copy
import json
import logging
import tempfile
from datetime import datetime
from html import escape
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, send_file

from api.config import OUTPUT_DIR
from backend.dataset_loader import DatasetLoader, Segment
from backend.human_review import HumanReviewRepository, RUBRIC_FIELDS, VALID_STATUSES
from backend.integration.classification_service import SelectedClassificationService
from backend.integration.generation_service import SelectedGenerationService
from backend.integration.online_generation_service import OnlineFallbackGenerationService
from backend.integration.registry import ModelConfigurationError, SelectedModelRegistry
from backend.preprocessor import Preprocessor
from backend.utils import export_results, generate_run_id, setup_logging

setup_logging()
logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)
_runs: dict[str, dict] = {}
review_repo = HumanReviewRepository()
selected_model_registry = SelectedModelRegistry()
selected_generation_service = SelectedGenerationService(selected_model_registry)
online_generation_service = OnlineFallbackGenerationService()


def _result_path(run_id: str) -> Path:
    return OUTPUT_DIR / f"results_{run_id}.json"


def _dicts_to_segments(rows: list[dict]) -> list[Segment]:
    segments = []
    for row in rows:
        segment = Segment(
            segment_id=str(row.get("segment_id", "")),
            municipality=str(row.get("municipality", "")),
            minute_id=str(row.get("minute_id", "")),
            date=str(row.get("date", "")),
            item_id=row.get("item_id"),
            item_title=str(row.get("item_title", "")),
            subject_text=str(row.get("subject_text", "")),
            full_text=str(row.get("full_text", "")),
            summary=str(row.get("summary", "")),
            theme=str(row.get("theme", "")),
            topics=list(row.get("topics") or []),
            voting=list(row.get("voting") or []),
            segment_start=row.get("segment_start"),
            segment_end=row.get("segment_end"),
            source_filename=str(row.get("source_filename", "")),
            metadata=dict(row.get("metadata") or {}),
        )
        for name in ("iip_total", "binary_probability", "binary_threshold", "iip_direct", "iip_derived"):
            setattr(segment, name, row.get(name) if row.get(name) is not None else (0.0 if name == "iip_total" else None))
        segment.iip_scores = dict(row.get("iip_scores") or {})
        segment.iip_level = str(row.get("iip_level", ""))
        segment.is_newsworthy = bool(row.get("is_newsworthy", False))
        segment.priority_rank = row.get("priority_rank")
        segment.model_version = str(row.get("model_version", ""))
        segment.generation_model_version = str(row.get("generation_model_version", ""))
        segment.provenance = dict(row.get("provenance") or {})
        segment.generated_news = dict(row.get("generated_news") or {})
        segments.append(segment)
    return segments


def _segments_summary(segments: list[Segment]) -> list[dict]:
    return [{
        "ranking_position": s.priority_rank or index + 1,
        "segment_id": s.segment_id,
        "municipality": s.municipality,
        "date": s.date,
        "minute_id": s.minute_id,
        "item_title": s.item_title,
        "subject": s.subject_text or s.item_title,
        "full_text": s.full_text,
        "source_filename": s.source_filename,
        "is_newsworthy": s.is_newsworthy,
        "binary_probability": s.binary_probability,
        "binary_threshold": s.binary_threshold,
        "iip_total": s.iip_total,
        "iip_level": s.iip_level,
        "iip_scores": s.iip_scores,
        "iip_direct": s.iip_direct,
        "iip_derived": s.iip_derived,
        "priority_rank": s.priority_rank,
        "model_version": s.model_version,
        "generation_model_version": s.generation_model_version,
        "provenance": s.provenance,
        "generated_news": s.generated_news,
    } for index, s in enumerate(segments)]


def _load_run(run_id: str) -> dict | None:
    if run_id in _runs:
        return _runs[run_id]
    path = _result_path(run_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = _dicts_to_segments(payload.get("segments") or [])
        results = {key: value for key, value in payload.items() if key != "segments"}
        results["segments_summary"] = _segments_summary(segments)
        _runs[run_id] = {"type": "run", "results": results, "segments": segments}
        return _runs[run_id]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Falha ao recuperar o run %s: %s", run_id, exc)
        return None


@api_bp.get("/config")
def get_config():
    return jsonify({"selected_models": selected_model_registry.status()})


@api_bp.post("/upload")
def upload_file():
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"error": "Selecione um ficheiro com segmentos."}), 400
    uploaded = request.files["file"]
    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in {".json", ".jsonl", ".csv"}:
        return jsonify({"error": "Formato não suportado. Use JSON, JSONL ou CSV com segmentos já preparados."}), 400
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            uploaded.save(handle.name)
            tmp_path = Path(handle.name)
        processor = Preprocessor()
        segments = processor.process(DatasetLoader().load(tmp_path))
        if not segments:
            return jsonify({"error": "O ficheiro não contém segmentos válidos."}), 400
        upload_id = generate_run_id()
        stats = processor.get_stats(segments)
        _runs[upload_id] = {"type": "upload", "segments": segments, "stats": stats, "filename": uploaded.filename}
        return jsonify({"status": "ok", "upload_id": upload_id, "filename": uploaded.filename, "stats": stats})
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.exception("Falha no upload")
        return jsonify({"error": str(exc)}), 422
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)


@api_bp.get("/integration/status")
def selected_models_status():
    try:
        return jsonify({"status": "ok", "components": selected_model_registry.status()})
    except ModelConfigurationError as exc:
        return jsonify({"error": str(exc)}), 500


@api_bp.get("/integration/generation/health")
def generation_remote_health():
    try:
        return jsonify({"status": "ok", **selected_generation_service.remote_health()})
    except (ModelConfigurationError, RuntimeError) as exc:
        return jsonify({"status": "error", "reachable": False, "detail": str(exc)}), 200


@api_bp.get("/integration/generation/options")
def generation_options():
    status = selected_model_registry.status().get("generation", {})
    primary = {
        "id": "selected_remote",
        "label": "Llama 3.1 8B + LoRA — modelo selecionado",
        "provider": "deucalion",
        "role": "selected_model",
        "configured": bool(status.get("ready")),
    }
    return jsonify({
        "primary": primary,
        "fallbacks": online_generation_service.available_models(),
        "note": "Os modelos online são alternativas operacionais para testar o fluxo de revisão; não substituem o modelo selecionado na avaliação da dissertação.",
    })


@api_bp.post("/integration/classify")
def classify_with_selected_model():
    body = request.get_json(silent=True) or {}
    cached = _runs.get(str(body.get("upload_id", "")))
    if not cached or cached.get("type") != "upload":
        return jsonify({"error": "upload_id inválido. Carregue os segmentos antes da classificação."}), 400
    segments = copy.deepcopy(cached["segments"])
    try:
        service = SelectedClassificationService(selected_model_registry)
        segments = service.apply(segments, service.predict(segments))
        run_id = generate_run_id()
        newsworthy = sum(s.is_newsworthy for s in segments)
        model_version = segments[0].model_version if segments else ""
        direct_values = [s.iip_direct for s in segments if s.iip_direct is not None]
        classification = {
            "total_segments": len(segments),
            "newsworthy": newsworthy,
            "routine": len(segments) - newsworthy,
            "newsworthy_pct": round(100 * newsworthy / len(segments), 1),
            "iip_direct_mean": round(sum(direct_values) / len(direct_values), 4) if direct_values else None,
            "model_version": model_version,
            "decision_source": "binary_head",
            "ranking_source": "iip_direct_descending_within_minute",
        }
        results = {
            "run_id": run_id, "timestamp": datetime.now().isoformat(),
            "filename": cached.get("filename", ""), "model": model_version,
            "strategy": "selected_classification_model", "classification": classification,
            "generation": {"total_generated": 0}, "segments_summary": _segments_summary(segments),
        }
        export_results(segments, OUTPUT_DIR, run_id=run_id, filename=results["filename"], model=model_version,
                       strategy=results["strategy"], classification=classification, generation=results["generation"])
        _runs[run_id] = {"type": "run", "results": results, "segments": segments}
        return jsonify({"status": "ok", **results})
    except (ModelConfigurationError, ValueError, RuntimeError, OSError) as exc:
        logger.exception("Falha na classificação")
        return jsonify({"error": str(exc)}), 422


@api_bp.post("/integration/generate")
def generate_with_selected_model():
    body = request.get_json(silent=True) or {}
    selected_ids = {str(value) for value in body.get("segment_ids") or [] if str(value)}
    if not selected_ids:
        return jsonify({"error": "Indique pelo menos um segment_id para geração."}), 400
    parent_id = str(body.get("classification_run_id", ""))
    parent = _load_run(parent_id)
    if not parent:
        return jsonify({"error": "classification_run_id inválido."}), 400
    available_ids = {segment.segment_id for segment in parent["segments"]}
    unknown_ids = sorted(selected_ids - available_ids)
    if unknown_ids:
        return jsonify({"error": f"segment_id desconhecido: {', '.join(unknown_ids)}"}), 400
    try:
        overrides = {key: body[key] for key in ("temperature", "max_new_tokens") if body.get(key) is not None}
        generator_id = str(body.get("generator_id") or "selected_remote").strip()
        working_segments = copy.deepcopy(parent["segments"])

        if generator_id == "selected_remote":
            segments = selected_generation_service.generate_selected(working_segments, selected_ids, overrides)
            status = selected_model_registry.status()["generation"]
            model_key = status.get("model_key", "")
            model_version = status.get("model_version", "")
            generation_role = "selected_model"
            strategy = "selected_lora_model"
        else:
            if not online_generation_service.is_available(generator_id):
                return jsonify({"error": "Gerador online não disponível ou sem credenciais configuradas."}), 400
            segments = online_generation_service.generate_selected(working_segments, selected_ids, generator_id, overrides)
            model_key = generator_id
            model_version = generator_id
            generation_role = "operational_fallback"
            strategy = "operational_online_fallback"

        run_id = generate_run_id()
        generation = {
            "total_selected": len(selected_ids),
            "total_generated": sum(bool(s.generated_news.get("titulo")) for s in segments),
            "model_key": model_key,
            "model_version": model_version,
            "generation_role": generation_role,
            "selection_source": "explicit_human_selection",
            "experimental_status": "selected_thesis_model" if generation_role == "selected_model" else "not_selected_thesis_model",
        }
        classification = parent["results"].get("classification", {})
        results = {
            "run_id": run_id, "timestamp": datetime.now().isoformat(),
            "filename": parent["results"].get("filename", ""), "model": generation["model_key"],
            "strategy": strategy, "classification": classification,
            "generation": generation, "parent_classification_run_id": parent_id,
            "segments_summary": _segments_summary(segments),
        }
        export_results(segments, OUTPUT_DIR, run_id=run_id, filename=results["filename"], model=results["model"],
                       strategy=results["strategy"], classification=classification, generation=generation,
                       parent_classification_run_id=parent_id)
        _runs[run_id] = {"type": "run", "results": results, "segments": segments}
        return jsonify({"status": "ok", **results, "review_url": f"/revisao.html?run_id={run_id}"})
    except (ModelConfigurationError, ValueError, RuntimeError, OSError) as exc:
        logger.exception("Falha na geração")
        return jsonify({"error": str(exc)}), 422


@api_bp.get("/runs/summary")
def list_runs_summary():
    for path in OUTPUT_DIR.glob("results_*.json"):
        _load_run(path.stem.removeprefix("results_"))
    rows = []
    for run_id, entry in _runs.items():
        if entry.get("type") == "run":
            result = entry["results"]
            rows.append({"run_id": run_id, **{key: result.get(key, {}) for key in
                        ("timestamp", "filename", "model", "strategy", "classification", "generation")}})
    return jsonify({"runs": sorted(rows, key=lambda row: row["timestamp"], reverse=True)})


@api_bp.get("/results/<run_id>")
def get_results(run_id: str):
    run = _load_run(run_id)
    return jsonify(run["results"]) if run else (jsonify({"error": "run_id não encontrado."}), 404)


@api_bp.get("/download/<run_id>")
def download_results(run_id: str):
    path = _result_path(run_id)
    return send_file(path, as_attachment=True) if path.exists() else (jsonify({"error": "run_id não encontrado."}), 404)


@api_bp.post("/review/bootstrap")
def bootstrap_review_batch():
    body = request.get_json(silent=True) or {}
    run_id = str(body.get("run_id", ""))
    run = _load_run(run_id)
    if not run:
        return jsonify({"error": "run_id inválido."}), 400
    batch_id = str(body.get("batch_id") or f"review_{run_id}")
    result = review_repo.bootstrap_batch(batch_id, run_id, run["segments"], filename=run["results"].get("filename", ""))
    return jsonify({"status": "ok", **result, "summary": review_repo.get_batch_summary(batch_id)})


@api_bp.get("/review/<batch_id>")
def get_review_batch(batch_id: str):
    return jsonify({"batch_id": batch_id, "summary": review_repo.get_batch_summary(batch_id),
                    "items": review_repo.list_items(batch_id), "allowed_statuses": sorted(VALID_STATUSES),
                    "rubric_fields": RUBRIC_FIELDS})


@api_bp.get("/review/item/<int:item_id>")
def get_review_item(item_id: int):
    try:
        return jsonify(review_repo.get_item(item_id))
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404


@api_bp.post("/review/item/<int:item_id>")
def update_review_item(item_id: int):
    body = request.get_json(silent=True) or {}
    try:
        item = review_repo.update_item(item_id, **{key: body.get(key) for key in (
            "review_status", "reviewer_name", "reviewer_comment", "manual_antetitle", "manual_title",
            "manual_superlead", "manual_lead", "manual_body", "rubric")})
        return jsonify({"status": "ok", "item": item})
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@api_bp.get("/review/<batch_id>/export")
def export_review_batch(batch_id: str):
    return send_file(review_repo.export_review_dataset(batch_id, OUTPUT_DIR / f"review_{batch_id}.json"), as_attachment=True)


def _approved_news(batch_id: str) -> list[dict]:
    result = []
    for item in review_repo.list_items(batch_id):
        if item.get("review_status") not in {"aprovado", "aprovado_com_edicao"}:
            continue
        generated, corrected = item["generated"], item["corrected"]
        result.append({
            "segment_id": item["segment_id"], "municipality": item["municipality"] or "Município",
            "iip_total": item["iip_total"],
            "antetitulo": corrected.get("antetitulo") or generated.get("antetitulo") or "",
            "titulo": corrected.get("title") or generated.get("title") or "",
            "superlead": corrected.get("superlead") or generated.get("superlead") or "",
            "lead": corrected.get("lead") or generated.get("lead") or "",
            "corpo": corrected.get("body") or generated.get("body") or "",
        })
    return result


def _newsletter_html(items: list[dict]) -> str:
    municipality = items[0]["municipality"] if items else "Município"
    articles = "".join(
        f'<article><p class="kicker">{escape(str(x["antetitulo"]))}</p><h2>{escape(str(x["titulo"]))}</h2>'
        f'<p class="superlead">{escape(str(x["superlead"]))}</p><p class="lead">{escape(str(x["lead"]))}</p>'
        f'<p>{escape(str(x["corpo"]))}</p></article>'
        for x in items)
    safe_municipality = escape(str(municipality))
    return f'''<!doctype html><html lang="pt-PT"><head><meta charset="utf-8"><title>GenLocalNews — notícias aprovadas — {safe_municipality}</title>
<style>body{{max-width:820px;margin:2rem auto;font:16px/1.6 Georgia,serif;color:#171717}}header{{border-bottom:3px solid}}article{{padding:1.5rem 0;border-bottom:1px solid #ccc}}.kicker{{text-transform:uppercase;color:#9f3d28;font:12px monospace}}.superlead{{font-style:italic}}.lead{{font-weight:bold}}footer{{color:#666;font-size:12px}}</style></head>
<body><header><h1>GenLocalNews — notícias aprovadas</h1><p>{safe_municipality} · {datetime.now():%d/%m/%Y}</p></header>{articles}</body></html>'''


@api_bp.get("/review/<batch_id>/publish")
def publish_review_batch(batch_id: str):
    fmt = (request.args.get("format") or "html").lower()
    items = _approved_news(batch_id)
    if fmt == "json":
        return jsonify({"batch_id": batch_id, "total": len(items), "items": items})
    if fmt not in {"html", "iframe"}:
        return jsonify({"error": "Formato inválido. Use html, iframe ou json."}), 400
    return Response(_newsletter_html(items), mimetype="text/html",
                    headers={"Content-Disposition": f"attachment; filename=noticias_aprovadas_{batch_id}.html"})
