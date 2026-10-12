"""Inferência do checkpoint Transformer multitarefa selecionado."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import NEWS_VALUES
from .iip import compute_iip_from_predictions


def _prepare_input(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"segment_id", "text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas de inferência em falta: {sorted(missing)}")
    work = frame.copy()
    if "minute_id" not in work:
        work["minute_id"] = "unassigned"
    return work


def _finalize(frame: pd.DataFrame, threshold: float, model_version: str) -> pd.DataFrame:
    frame["newsworthy"] = frame["binary_probability"] >= threshold
    frame["binary_threshold"] = threshold
    frame["priority_rank"] = frame.groupby("minute_id")["iip_direct"].rank(method="first", ascending=False).astype(int)
    frame["model_version"] = model_version
    frame["news_values"] = frame.apply(
        lambda row: {name: int(row[name]) for name in NEWS_VALUES}, axis=1
    )
    columns = [
        "segment_id", "minute_id", "newsworthy", "binary_probability", "binary_threshold",
        *NEWS_VALUES, "news_values", "iip_direct", "iip_derived", "priority_rank", "model_version",
    ]
    return frame[columns]


def _load_checkpoint_state(model, path: Path, checkpoint_format: str) -> None:
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # compatibilidade com versões anteriores do PyTorch
        state = torch.load(path, map_location="cpu")
    strict = checkpoint_format == "full_state_dict"
    incompatible = model.load_state_dict(state, strict=strict)
    if not strict:
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        missing = sorted(trainable.intersection(incompatible.missing_keys))
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint parcial incompatível: parâmetros em falta={missing}; "
                f"inesperados={list(incompatible.unexpected_keys)}"
            )


def predict_transformer(
    frame: pd.DataFrame,
    checkpoint_dir: str | Path,
    *,
    batch_size: int = 8,
    device: str | None = None,
) -> pd.DataFrame:
    work = _prepare_input(frame)
    checkpoint_dir = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_dir / "model_metadata.json").read_text(encoding="utf-8"))
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("PyTorch/Transformers não instalados") from exc
    from .multitask_model import build_model, ordinal_logits_to_scores
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer", trust_remote_code=True)
    model = build_model(
        metadata["model_config"],
        metadata["experiment_config"],
        metadata.get("binary_pos_weight", 1.0),
        metadata.get("ordinal_pos_weights"),
        metadata.get("ordinal_initial_biases"),
    )
    _load_checkpoint_state(
        model,
        checkpoint_dir / "model_state.pt",
        metadata.get("checkpoint_format", "full_state_dict"),
    )
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(selected_device).eval()
    records = []
    max_length = int(metadata["model_config"].get("max_length", 512))
    with torch.no_grad():
        for start in range(0, len(work), batch_size):
            part = work.iloc[start : start + batch_size]
            encoded = tokenizer(
                part["text"].fillna("").astype(str).tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(selected_device)
            outputs = model(**encoded)
            probabilities = outputs["binary_logits"].sigmoid().cpu().numpy()
            direct = outputs["iip_prediction"].cpu().numpy()
            values = {
                name: ordinal_logits_to_scores(outputs["ordinal_logits"][name]).cpu().numpy()
                for name in NEWS_VALUES
            }
            for offset, (_, source) in enumerate(part.iterrows()):
                predicted_values = {name: int(values[name][offset]) for name in NEWS_VALUES}
                records.append(
                    {
                        "segment_id": source["segment_id"],
                        "minute_id": source["minute_id"],
                        "binary_probability": float(probabilities[offset]),
                        **predicted_values,
                        "iip_direct": float(direct[offset]),
                        "iip_derived": compute_iip_from_predictions(predicted_values),
                    }
                )
    return _finalize(
        pd.DataFrame(records),
        float(metadata.get("binary_threshold", 0.5)),
        f"{metadata['model_config'].get('model_key', 'transformer')}:{metadata['experiment_config'].get('setting', 'full')}",
    )
