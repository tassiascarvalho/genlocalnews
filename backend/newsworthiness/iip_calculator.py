"""Cálculo do Índice de Interesse Público (IIP).

Os valores-notícia (escala 1 a 5) são normalizados para [0, 1] e combinados
por média ponderada. Se faltar algum valor, o IIP não é calculado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CRITERIA = [
    "relevancia",
    "proximidade",
    "novidade",
    "atualidade",
    "continuidade",
    "notoriedade",
    "negatividade",
]

WEIGHTS = {
    "relevancia": 0.30,
    "proximidade": 0.20,
    "novidade": 0.15,
    "atualidade": 0.10,
    "continuidade": 0.10,
    "notoriedade": 0.10,
    "negatividade": 0.05,
}

LEVEL_THRESHOLDS = {
    "baixo": (0.00, 0.44),
    "medio": (0.45, 0.61),
    "alto": (0.62, 1.00),
}


@dataclass(frozen=True)
class IIPResult:
    raw_scores: dict[str, int]
    normalised_scores: dict[str, float]
    iip_total: float | None
    iip_level: str
    has_all_criteria: bool


class IIPCalculator:
    criteria = CRITERIA
    weights = WEIGHTS

    @staticmethod
    def normalise_criterion(value: Any) -> float | None:
        """Converte critério 1..5 ou 0..1 para 0..1.

        Valores vazios devolvem None. Valores fora da escala são truncados.
        """
        if value is None:
            return None
        text = str(value).strip().replace(",", ".")
        if text == "":
            return None
        try:
            val = float(text)
        except Exception:
            return None
        if val > 1.0:
            val = (val - 1.0) / 4.0
        return round(max(0.0, min(1.0, val)), 4)

    @staticmethod
    def denormalise_to_1_5(value: Any) -> int | None:
        norm = IIPCalculator.normalise_criterion(value)
        if norm is None:
            return None
        return int(round(norm * 4 + 1))

    @classmethod
    def has_all_criteria(cls, scores: dict[str, Any]) -> bool:
        return all(cls.normalise_criterion(scores.get(c)) is not None for c in cls.criteria)

    @classmethod
    def compute(cls, scores: dict[str, Any], *, require_all: bool = True) -> IIPResult:
        normalised: dict[str, float] = {}
        raw: dict[str, int] = {}
        missing = []
        for criterion in cls.criteria:
            norm = cls.normalise_criterion(scores.get(criterion))
            if norm is None:
                missing.append(criterion)
                if not require_all:
                    norm = 0.0
                else:
                    continue
            normalised[criterion] = norm
            raw[criterion] = int(round(norm * 4 + 1))

        has_all = not missing
        if require_all and not has_all:
            return IIPResult(raw, normalised, None, "", False)

        total = sum(normalised.get(c, 0.0) * cls.weights[c] for c in cls.criteria)
        total = round(max(0.0, min(1.0, total)), 4)
        return IIPResult(raw, normalised, total, cls.level(total), has_all)

    @staticmethod
    def level(iip_total: float | None) -> str:
        if iip_total is None:
            return ""
        val = max(0.0, min(1.0, float(iip_total)))
        for name, (lo, hi) in LEVEL_THRESHOLDS.items():
            if lo <= val <= hi:
                return name
        return "baixo"
