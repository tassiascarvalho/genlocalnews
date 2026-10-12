from __future__ import annotations

from typing import Any

from .registry import SelectedModelRegistry


EXPECTED_FIELDS = ("antetitulo", "titulo", "superlead", "lead", "corpo")


class SelectedGenerationService:
    """Cliente do serviço de inferência do Llama 3.1 8B + LoRA.

    O modelo generativo não é carregado pela aplicação. O endereço do serviço
    é definido em ``generation.remote_url``.
    """

    def __init__(self, registry: SelectedModelRegistry | None = None):
        self.registry = registry or SelectedModelRegistry()

    @staticmethod
    def _requests():
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Instale requests para utilizar o backend remoto de geração") from exc
        return requests

    def remote_health(self, timeout: float = 2.0) -> dict[str, Any]:
        cfg = self.registry.require_ready("generation")
        backend = str(cfg.get("backend", "remote") or "remote").strip().lower()
        if backend != "remote":
            return {
                "backend": backend,
                "reachable": False,
                "detail": "Esta distribuição operacional utiliza geração remota.",
            }

        requests = self._requests()
        remote_url = str(cfg.get("remote_url") or "").rstrip("/")
        try:
            response = requests.get(f"{remote_url}/health", timeout=timeout)
            try:
                payload = response.json() if response.content else {}
            except ValueError:
                payload = {"raw": response.text[:1000]}
            return {
                "backend": "remote",
                "remote_url": remote_url,
                "reachable": bool(response.ok),
                "http_status": response.status_code,
                "detail": payload,
            }
        except requests.RequestException as exc:
            return {
                "backend": "remote",
                "remote_url": remote_url,
                "reachable": False,
                "detail": str(exc),
            }

    def generate(
        self,
        segment: Any,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = self.registry.require_ready("generation")
        backend = str(cfg.get("backend", "remote") or "remote").strip().lower()
        if backend != "remote":
            raise RuntimeError(
                "Backend generativo não suportado nesta distribuição. "
                "Configure generation.backend='remote'."
            )

        requests = self._requests()
        options = {**cfg, **(overrides or {})}
        remote_url = str(options.get("remote_url") or "").rstrip("/")
        timeout = int(options.get("remote_timeout_seconds", 600))
        payload = {
            "segment_id": segment.segment_id,
            "text": segment.operational_text,
            "max_seq_length": int(options.get("max_seq_length", 4096)),
            "max_new_tokens": int(options.get("max_new_tokens", 450)),
            "temperature": float(options.get("temperature", 0.0)),
        }

        try:
            response = requests.post(f"{remote_url}/generate", json=payload, timeout=timeout)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Não foi possível contactar o serviço de geração em {remote_url}: {exc}"
            ) from exc

        if not response.ok:
            raise RuntimeError(
                f"Serviço remoto devolveu HTTP {response.status_code}: {response.text[:2000]}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError("O serviço remoto não devolveu JSON válido") from exc

        news = data.get("result")
        if not isinstance(news, dict):
            raise ValueError("Resposta inválida do serviço remoto: campo result ausente")

        missing = [field for field in EXPECTED_FIELDS if field not in news]
        extra = [field for field in news if field not in EXPECTED_FIELDS]
        if missing or extra:
            raise ValueError(
                f"Schema generativo remoto inválido; missing={missing}, extra={extra}"
            )

        clean = {field: str(news.get(field, "")).strip() for field in EXPECTED_FIELDS}
        if not clean["titulo"] or not clean["lead"] or not clean["corpo"]:
            raise ValueError("A saída deve conter titulo, lead e corpo")

        clean.update({
            "strategy": "selected_lora_model_remote",
            "model_key": data.get("model_key", cfg.get("model_key", "")),
            "model_version": data.get("model_version", cfg.get("model_version", "")),
            "source_segment_id": segment.segment_id,
            "parse_status": "json",
        })
        return clean

    def generate_selected(
        self,
        segments: list[Any],
        selected_ids: set[str],
        overrides: dict[str, Any] | None = None,
    ) -> list[Any]:
        for segment in segments:
            if segment.segment_id not in selected_ids:
                continue
            news = self.generate(segment, overrides=overrides)
            segment.generated_news = news
            segment.generation_model_version = str(news.get("model_version", ""))
            segment.provenance["generation"] = {
                "source_segment_id": segment.segment_id,
                "model_key": news.get("model_key", ""),
                "model_version": news.get("model_version", ""),
                "strategy": news.get("strategy", ""),
                "selection_source": "explicit_human_selection",
            }
        return segments
