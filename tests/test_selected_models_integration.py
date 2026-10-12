from __future__ import annotations

from backend.dataset_loader import Segment
from backend.integration.classification_service import SelectedClassificationService


def _segment() -> Segment:
    return Segment(
        segment_id="Porto_cm_001_s1",
        minute_id="Porto_cm_001.json",
        municipality="Porto",
        date="2026-08-31",
        full_text="O executivo aprovou a medida por unanimidade.",
    )


def test_integrated_classification_and_generation_routes(monkeypatch, tmp_path):
    import backend.routes as routes
    from app import create_app

    routes._runs.clear()
    routes._runs["upload-1"] = {
        "type": "upload",
        "segments": [_segment()],
        "filename": "minute.json",
    }
    monkeypatch.setattr(routes, "OUTPUT_DIR", tmp_path)

    prediction = {
        "segment_id": "Porto_cm_001_s1",
        "minute_id": "Porto_cm_001.json",
        "newsworthy": True,
        "binary_probability": 0.91,
        "binary_threshold": 0.57,
        "news_values": {
            "relevancia": 5, "proximidade": 4, "novidade": 3,
            "atualidade": 4, "continuidade": 2, "notoriedade": 3,
            "negatividade": 1,
        },
        "iip_direct": 0.78,
        "iip_derived": 0.71,
        "priority_rank": 1,
        "model_version": "classifier-test-v1",
    }
    monkeypatch.setattr(SelectedClassificationService, "predict", lambda self, segments: [prediction])

    def fake_generate(segments, selected_ids, overrides):
        assert selected_ids == {"Porto_cm_001_s1"}
        segment = segments[0]
        segment.generated_news = {
            "antetitulo": "Município", "titulo": "Executivo aprova medida",
            "superlead": "", "lead": "A decisão foi aprovada por unanimidade.",
            "corpo": "O executivo municipal aprovou a medida.",
            "model_key": "generator-test", "model_version": "generator-test-v1",
            "strategy": "selected_lora_model",
        }
        segment.generation_model_version = "generator-test-v1"
        segment.provenance["generation"] = {"source_segment_id": segment.segment_id}
        return segments

    monkeypatch.setattr(routes.selected_generation_service, "generate_selected", fake_generate)
    original_status = routes.selected_model_registry.status

    def fake_status():
        status = original_status()
        status["generation"].update({"model_key": "generator-test", "model_version": "generator-test-v1"})
        return status

    monkeypatch.setattr(routes.selected_model_registry, "status", fake_status)

    client = create_app().test_client()
    client.post("/login", data={"username": "ubi", "password": "ubi", "next": "/"})
    classified = client.post("/api/integration/classify", json={"upload_id": "upload-1"})
    assert classified.status_code == 200
    classified_data = classified.get_json()
    assert classified_data["segments_summary"][0]["binary_threshold"] == 0.57
    assert classified_data["segments_summary"][0]["iip_derived"] == 0.71
    assert classified_data["segments_summary"][0]["iip_scores"]["relevancia"] == 5
    assert classified_data["segments_summary"][0]["full_text"] == "O executivo aprovou a medida por unanimidade."

    generated = client.post(
        "/api/integration/generate",
        json={
            "classification_run_id": classified_data["run_id"],
            "segment_ids": ["Porto_cm_001_s1"],
        },
    )
    assert generated.status_code == 200
    generated_data = generated.get_json()
    assert generated_data["generation"]["total_generated"] == 1
    assert generated_data["segments_summary"][0]["generated_news"]["antetitulo"] == "Município"
    assert generated_data["review_url"] == f"/revisao.html?run_id={generated_data["run_id"]}"


def test_generation_requires_explicit_selection():
    from app import create_app

    client = create_app().test_client()
    client.post("/login", data={"username": "ubi", "password": "ubi", "next": "/"})
    response = client.post(
        "/api/integration/generate",
        json={"classification_run_id": "any", "segment_ids": []},
    )
    assert response.status_code == 400
    assert "segment_id" in response.get_json()["error"]


def test_remote_generation_configuration_does_not_require_local_adapter(tmp_path):
    import json
    from backend.integration.registry import SelectedModelRegistry

    config = tmp_path / "models.json"
    config.write_text(json.dumps({
        "classification": {
            "enabled": False,
            "bundle_path": str(tmp_path / "classification-missing"),
        },
        "generation": {
            "enabled": True,
            "backend": "remote",
            "remote_url": "http://127.0.0.1:8000",
            "model_key": "llama31_8b_4096_lora",
            "model_version": "llama31_8b_4096_lora",
            "base_model_id": "meta-llama/Llama-3.1-8B-Instruct",
        },
    }), encoding="utf-8")

    status = SelectedModelRegistry(config).status()["generation"]
    assert status["ready"] is True
    assert status["backend"] == "remote"
    assert status["remote_url"] == "http://127.0.0.1:8000"
    assert status["missing"] == []
