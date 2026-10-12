"""Encoder partilhado com saídas binária, ordinais (sete valores-notícia) e IIP."""

from __future__ import annotations

import math
from typing import Any

from .constants import NEWS_VALUES


def require_transformer_stack():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyTorch/Transformers não instalados. Execute pip install -r requirements.txt") from exc
    return torch, AutoModel, AutoTokenizer


def cumulative_ordinal_targets(labels, levels: int = 5):
    """Converte pontuações 1..levels em alvos cumulativos CORAL (N, levels-1)."""
    torch, _, _ = require_transformer_stack()
    thresholds = torch.arange(1, levels, device=labels.device).unsqueeze(0)
    return (labels.unsqueeze(1) > thresholds).float()


def ordinal_logits_to_scores(logits):
    """Pontuação = 1 + número de fronteiras com probabilidade > 0,5."""
    return 1 + (logits.sigmoid() > 0.5).sum(dim=-1)


def build_model(
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    binary_pos_weight: float = 1.0,
    ordinal_pos_weights: dict[str, list[float]] | None = None,
    ordinal_initial_biases: dict[str, list[float]] | None = None,
):
    torch, AutoModel, _ = require_transformer_stack()
    nn = torch.nn
    F = torch.nn.functional

    class RankConsistentOrdinalHead(nn.Module):
        """Saída ordinal cumulativa: pesos partilhados e termos independentes ordenados."""

        def __init__(self, hidden_size: int, initial_biases: list[float] | None = None):
            super().__init__()
            self.score = nn.Linear(hidden_size, 1, bias=False)
            biases = initial_biases or [1.5, 0.5, -0.5, -1.5]
            if len(biases) != 4:
                raise ValueError("Cada cabeça ordinal requer quatro biases cumulativos")
            ordered = sorted((float(value) for value in biases), reverse=True)
            self.first_bias = nn.Parameter(torch.tensor(ordered[0], dtype=torch.float32))
            gaps = [max(ordered[idx] - ordered[idx + 1], 1e-4) for idx in range(3)]
            raw_gaps = [math.log(math.expm1(gap)) if gap < 20 else gap for gap in gaps]
            self.raw_gaps = nn.Parameter(torch.tensor(raw_gaps, dtype=torch.float32))

        def forward(self, hidden):
            positive_gaps = F.softplus(self.raw_gaps)
            offsets = torch.cat(
                [torch.zeros(1, device=hidden.device, dtype=hidden.dtype), positive_gaps.to(hidden.dtype)]
            )
            biases = self.first_bias.to(hidden.dtype) - torch.cumsum(offsets, dim=0)
            return self.score(hidden) + biases.unsqueeze(0)

    class MultiTaskNewsworthinessModel(nn.Module):
        def __init__(self):
            super().__init__()
            trust_remote_code = bool(model_config.get("trust_remote_code", False))
            kwargs = {
                "trust_remote_code": trust_remote_code,
                "revision": model_config.get("revision", "main"),
            }
            if model_config.get("dtype") in {"float16", "bfloat16", "float32"}:
                kwargs["torch_dtype"] = getattr(torch, model_config["dtype"])
            self.backbone = AutoModel.from_pretrained(model_config["model_id"], **kwargs)
            if getattr(self.backbone.config, "pad_token_id", None) is None:
                self.backbone.config.pad_token_id = getattr(self.backbone.config, "eos_token_id", None)
            if training_config.get("gradient_checkpointing", False):
                if not hasattr(self.backbone, "gradient_checkpointing_enable"):
                    raise ValueError("O backbone selecionado não suporta gradient checkpointing")
                self.backbone.gradient_checkpointing_enable()
                if hasattr(self.backbone.config, "use_cache"):
                    self.backbone.config.use_cache = False
            hidden_size = getattr(self.backbone.config, "hidden_size", None) or getattr(
                self.backbone.config, "d_model", None
            )
            if hidden_size is None:
                raise ValueError("Não foi possível determinar hidden_size/d_model")

            setting = training_config.get("setting", "full")
            if setting == "head_only":
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False
            elif setting == "lora":
                try:
                    from peft import LoraConfig, TaskType, get_peft_model
                except ImportError as exc:
                    raise RuntimeError("PEFT em falta para setting=lora") from exc
                lora = training_config.get("lora", {})
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=int(lora.get("r", 16)),
                    lora_alpha=int(lora.get("alpha", 32)),
                    lora_dropout=float(lora.get("dropout", 0.05)),
                    target_modules=model_config.get("lora_target_modules", lora.get("target_modules")),
                    bias="none",
                )
                self.backbone = get_peft_model(self.backbone, peft_config)
                if training_config.get("gradient_checkpointing", False) and hasattr(
                    self.backbone, "enable_input_require_grads"
                ):
                    self.backbone.enable_input_require_grads()
            elif setting != "full":
                raise ValueError("setting deve ser head_only, full ou lora")

            dropout = float(training_config.get("dropout", 0.1))
            self.dropout = nn.Dropout(dropout)
            self.binary_head = nn.Linear(hidden_size, 1)
            ordinal_head_type = training_config.get("ordinal_head_type", "independent_thresholds")
            if ordinal_head_type == "rank_consistent":
                self.ordinal_heads = nn.ModuleDict(
                    {
                        name: RankConsistentOrdinalHead(
                            hidden_size,
                            (ordinal_initial_biases or {}).get(name),
                        )
                        for name in NEWS_VALUES
                    }
                )
            elif ordinal_head_type == "independent_thresholds":
                self.ordinal_heads = nn.ModuleDict({name: nn.Linear(hidden_size, 4) for name in NEWS_VALUES})
                if ordinal_initial_biases:
                    with torch.no_grad():
                        for name, head in self.ordinal_heads.items():
                            if name in ordinal_initial_biases:
                                head.bias.copy_(torch.tensor(ordinal_initial_biases[name], dtype=head.bias.dtype))
            else:
                raise ValueError("ordinal_head_type deve ser independent_thresholds ou rank_consistent")
            self.iip_head = nn.Linear(hidden_size, 1)
            self.register_buffer("binary_pos_weight", torch.tensor(float(binary_pos_weight)))
            for name in NEWS_VALUES:
                weights = (ordinal_pos_weights or {}).get(name, [1.0, 1.0, 1.0, 1.0])
                if len(weights) != 4:
                    raise ValueError(f"ordinal_pos_weights[{name}] deve conter quatro valores")
                self.register_buffer(
                    f"ordinal_pos_weight_{name}",
                    torch.tensor([float(value) for value in weights], dtype=torch.float32),
                    # Só é guardado no checkpoint quando os pesos vêm dos metadados.
                    persistent=ordinal_pos_weights is not None,
                )

        @staticmethod
        def _masked_mean(hidden, attention_mask):
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        @staticmethod
        def _masked_average(losses, mask):
            return (losses * mask).sum() / mask.sum().clamp(min=1.0)

        def forward(self, input_ids, attention_mask, **labels):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = self.dropout(self._masked_mean(outputs.last_hidden_state, attention_mask))
            binary_logits = self.binary_head(pooled).squeeze(-1)
            ordinal_logits = {name: self.ordinal_heads[name](pooled) for name in NEWS_VALUES}
            iip_prediction = self.iip_head(pooled).squeeze(-1).sigmoid()
            result = {
                "binary_logits": binary_logits,
                "ordinal_logits": ordinal_logits,
                "iip_prediction": iip_prediction,
            }
            losses = {}
            if "binary_label" in labels:
                raw = F.binary_cross_entropy_with_logits(
                    binary_logits,
                    labels["binary_label"].float(),
                    reduction="none",
                    pos_weight=self.binary_pos_weight,
                )
                losses["binary"] = self._masked_average(raw, labels["mask_binary"].float())
            for name in NEWS_VALUES:
                key = f"label_{name}"
                if key in labels:
                    targets = cumulative_ordinal_targets(labels[key])
                    pos_weight = getattr(self, f"ordinal_pos_weight_{name}")
                    raw = F.binary_cross_entropy_with_logits(
                        ordinal_logits[name],
                        targets,
                        reduction="none",
                        pos_weight=pos_weight,
                    ).mean(1)
                    losses[name] = self._masked_average(raw, labels[f"mask_{name}"].float())
            if "iip_label" in labels:
                raw = F.mse_loss(iip_prediction, labels["iip_label"].float(), reduction="none")
                losses["iip"] = self._masked_average(raw, labels["mask_iip"].float())
            if losses:
                weights = training_config.get("loss_weights", {})
                ordinal_weight = float(weights.get("ordinal_each", 1.0 / len(NEWS_VALUES)))
                total = float(weights.get("binary", 1.0)) * losses["binary"]
                total = total + sum(ordinal_weight * losses[name] for name in NEWS_VALUES)
                total = total + float(weights.get("iip", 1.0)) * losses["iip"]
                result["loss"] = total
                result["losses"] = losses
            return result

    return MultiTaskNewsworthinessModel()


def load_tokenizer(model_config: dict[str, Any]):
    _, _, AutoTokenizer = require_transformer_stack()
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"],
        revision=model_config.get("revision", "main"),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        use_fast=model_config.get("use_fast", True),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
