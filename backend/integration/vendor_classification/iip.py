"""Cálculo do Índice de Interesse Público (IIP)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .constants import NEWS_VALUES

IIP_WEIGHTS = {
    "relevancia": 0.30,
    "proximidade": 0.20,
    "novidade": 0.15,
    "atualidade": 0.10,
    "continuidade": 0.10,
    "notoriedade": 0.10,
    "negatividade": 0.05,
}


def compute_iip(values: Mapping[str, Any], *, strict: bool = True) -> float:
    """IIP em [0, 1] a partir dos sete valores-notícia (escala 1 a 5)."""
    scores: dict[str, float] = {}
    for name in NEWS_VALUES:
        raw = values.get(name)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            if strict:
                raise ValueError(f"Valor-notícia em falta: {name}")
            return float("nan")
        score = float(raw)
        if not 1 <= score <= 5:
            raise ValueError(f"{name}={score} fora da escala ordinal [1, 5]")
        scores[name] = score

    if not np.isclose(sum(IIP_WEIGHTS.values()), 1.0):
        raise RuntimeError("Os pesos do IIP não somam 1")
    return float(sum(IIP_WEIGHTS[n] * ((scores[n] - 1.0) / 4.0) for n in NEWS_VALUES))


def compute_iip_from_predictions(values: Mapping[str, Any]) -> float:
    """IIP calculado a partir dos valores-notícia previstos."""
    return compute_iip(values, strict=True)
