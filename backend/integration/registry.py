from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


class ModelConfigurationError(RuntimeError):
    """Configuração inválida ou incompleta dos modelos selecionados."""


class SelectedModelRegistry:
    """Lê e valida a configuração dos modelos (sem os carregar)."""

    def __init__(self, config_path: str | Path | None = None):
        self.project_root = Path(__file__).resolve().parents[2]
        configured = config_path or os.getenv(
            "GENLOCALNEWS_MODEL_CONFIG",
            self.project_root / "config" / "model_selection.json",
        )
        self.config_path = Path(configured).expanduser().resolve()

    def load(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise ModelConfigurationError(f"Configuração inexistente: {self.config_path}")
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelConfigurationError(f"Não foi possível ler {self.config_path}: {exc}") from exc
        if not isinstance(payload.get("classification"), dict) or not isinstance(payload.get("generation"), dict):
            raise ModelConfigurationError("A configuração deve conter classification e generation")
        return payload

    def resolve_path(self, value: str | Path | None) -> Path | None:
        if not value:
            return None
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def component(self, name: str) -> dict[str, Any]:
        if name not in {"classification", "generation"}:
            raise KeyError(name)
        cfg = deepcopy(self.load()[name])
        if name == "classification" and "bundle_path" in cfg:
            cfg["bundle_path"] = self.resolve_path(cfg.get("bundle_path"))
        return cfg

    def status(self) -> dict[str, Any]:
        classification = self.component("classification")
        generation = self.component("generation")

        bundle = classification.get("bundle_path")
        checkpoint = bundle / "checkpoint" if bundle else None
        registry_path = bundle / "registry.json" if bundle else None
        class_required = ["model_metadata.json", "model_state.pt"]
        class_missing = [
            name for name in class_required
            if not checkpoint or not (checkpoint / name).exists()
        ]
        classification_base_model = ""
        metadata_path = checkpoint / "model_metadata.json" if checkpoint else None
        if metadata_path and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                classification_base_model = str(metadata.get("model_config", {}).get("model_id", ""))
                if classification_base_model:
                    candidate = Path(classification_base_model).expanduser()
                    if candidate.is_absolute() and not candidate.exists():
                        class_missing.append("classification_base_model")
            except (OSError, json.JSONDecodeError, TypeError):
                class_missing.append("valid_model_metadata.json")

        class_version = ""
        if registry_path and registry_path.exists():
            try:
                class_version = str(
                    json.loads(registry_path.read_text(encoding="utf-8")).get("bundle_version", "")
                )
            except (OSError, json.JSONDecodeError, TypeError):
                class_version = ""

        generation_backend = str(generation.get("backend", "remote") or "remote").strip().lower()
        remote_url = str(generation.get("remote_url") or "").strip()
        generation_missing: list[str] = []
        if generation_backend != "remote":
            generation_missing.append("backend=remote")
        if not remote_url:
            generation_missing.append("remote_url")

        return {
            "configuration_file": str(self.config_path),
            "classification": {
                "enabled": bool(classification.get("enabled", False)),
                "ready": bool(classification.get("enabled", False) and not class_missing),
                "backend": "transformer",
                "bundle_path": str(bundle) if bundle else "",
                "model_version": class_version,
                "base_model": classification_base_model,
                "missing": class_missing,
            },
            "generation": {
                "enabled": bool(generation.get("enabled", False)),
                "ready": bool(generation.get("enabled", False) and not generation_missing),
                "backend": generation_backend,
                "remote_url": remote_url,
                "model_key": generation.get("model_key", ""),
                "model_version": generation.get("model_version", ""),
                "base_model_id": generation.get("base_model_id", ""),
                "missing": generation_missing,
            },
        }

    def require_ready(self, component: str) -> dict[str, Any]:
        status = self.status()[component]
        if not status["enabled"]:
            raise ModelConfigurationError(f"O componente {component} não está ativado")
        if not status["ready"]:
            raise ModelConfigurationError(
                f"O componente {component} não está pronto; em falta: {', '.join(status['missing'])}"
            )
        return self.component(component)
