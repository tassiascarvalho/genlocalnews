"""Constantes partilhadas."""

NEWS_VALUES = (
    "relevancia",
    "proximidade",
    "novidade",
    "atualidade",
    "continuidade",
    "notoriedade",
    "negatividade",
)

LABEL_ALIASES = (
    "noticavel",
    "noticiavel",
    "noticiabilidade",
    "newsworthy",
    "label_binario",
    "classificacao_binaria",
    "noticiavel_s_n",
)

POSITIVE_LABELS = {"1", "s", "sim", "yes", "y", "true", "verdadeiro", "noticiavel"}
NEGATIVE_LABELS = {"0", "n", "nao", "não", "no", "false", "falso", "nao noticiavel", "não noticiavel"}
