from __future__ import annotations

import io
import json

from backend.dataset_loader import Segment
from backend.human_review import HumanReviewRepository


def test_upload_accepts_segmented_jsonl_and_rejects_raw_zip(tmp_path):
    import backend.routes as routes
    from app import create_app

    routes._runs.clear()
    client = create_app().test_client()
    client.post("/login", data={"username": "ubi", "password": "ubi", "next": "/"})
    record = {
        "segment_id": "porto-1",
        "minute_id": "porto-minute-1",
        "municipality": "Porto",
        "date": "2026-08-31",
        "text": "O executivo aprovou a medida por unanimidade.",
    }
    response = client.post(
        "/api/upload",
        data={"file": (io.BytesIO((json.dumps(record) + "\n").encode()), "segments.jsonl")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.get_json()["stats"]["total_segments"] == 1

    rejected = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"not-a-zip"), "raw-minutes.zip")},
        content_type="multipart/form-data",
    )
    assert rejected.status_code == 400
    assert "segmentos já preparados" in rejected.get_json()["error"]


def test_editorial_review_preserves_models_source_and_human_decision(tmp_path):
    repository = HumanReviewRepository(tmp_path / "review.db")
    segment = Segment(
        segment_id="porto-1",
        minute_id="porto-minute-1",
        municipality="Porto",
        date="2026-08-31",
        full_text="O executivo aprovou a medida por unanimidade.",
    )
    segment.is_newsworthy = True
    segment.iip_total = 0.78
    segment.iip_direct = 0.78
    segment.iip_derived = 0.71
    segment.iip_scores = {
        "relevancia": 5, "proximidade": 4, "novidade": 3, "atualidade": 4,
        "continuidade": 2, "notoriedade": 3, "negatividade": 1,
    }
    segment.binary_probability = 0.91
    segment.binary_threshold = 0.57
    segment.priority_rank = 1
    segment.model_version = "classifier-v1"
    segment.generation_model_version = "generator-v1"
    segment.provenance = {"classification": {"decision_source": "binary_head"}}
    segment.generated_news = {
        "antetitulo": "Município", "titulo": "Executivo aprova medida",
        "superlead": "", "lead": "A decisão foi aprovada por unanimidade.",
        "corpo": "O executivo municipal aprovou a medida.",
        "strategy": "selected_lora_model",
    }

    result = repository.bootstrap_batch("review-1", "run-1", [segment], "minute.json")
    assert result["inserted"] == 1
    item = repository.list_items("review-1")[0]
    assert item["classification_model_version"] == "classifier-v1"
    assert item["generation_model_version"] == "generator-v1"
    assert item["full_text"] == segment.full_text

    updated = repository.update_item(
        item["item_id"], review_status="aprovado_com_edicao", manual_title="Título revisto"
    )
    assert updated["review_status"] == "aprovado_com_edicao"
    assert updated["corrected"]["title"] == "Título revisto"
    assert updated["history"][-1]["action"] == "update"
