from __future__ import annotations

import json
import os
import uuid
from typing import Any

from .generation_prompts import SYSTEM_PROMPT, build_user_prompt

EXPECTED_FIELDS = ("antetitulo", "titulo", "superlead", "lead", "corpo")

# Fornecedores alternativos; cada um só fica disponível se as variáveis de ambiente estiverem definidas.
ONLINE_MODELS = {
    "iaedu": {
        "provider": "iaedu",
        "label": "IA Edu",
        "env": ("IAEDU_API_ENDPOINT", "IAEDU_API_KEY", "IAEDU_CHANNEL_ID"),
    },
    "gemini-2.5-pro": {
        "provider": "google",
        "label": "Gemini 2.5 Pro",
        "env": ("GOOGLE_API_KEY",),
    },
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {
        "provider": "together",
        "label": "Llama 3.3 70B — Together",
        "env": ("TOGETHER_API_KEY",),
    },
    "llama-3.3-70b-versatile": {
        "provider": "groq",
        "label": "Llama 3.3 70B — Groq",
        "env": ("GROQ_API_KEY",),
    },
}


class OnlineFallbackGenerationService:
    """Gerador alternativo, usado apenas quando o serviço do Llama 3.1 8B + LoRA
    não está disponível. Devolve os mesmos cinco campos, para que o fluxo de
    revisão possa ser testado. As saídas ficam marcadas como não pertencendo ao
    modelo selecionado.
    """

    @staticmethod
    def available_models() -> list[dict[str, Any]]:
        rows = []
        for model_id, spec in ONLINE_MODELS.items():
            configured = all(bool(os.getenv(name, "").strip()) for name in spec["env"])
            if configured:
                rows.append({
                    "id": model_id,
                    "provider": spec["provider"],
                    "label": spec["label"],
                    "role": "operational_fallback",
                })
        return rows

    @classmethod
    def is_available(cls, model_id: str) -> bool:
        return any(row["id"] == model_id for row in cls.available_models())

    @staticmethod
    def _requests():
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Instale requests para utilizar o gerador online temporário") from exc
        return requests

    @staticmethod
    def _extract_first_json_object(text: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "", 1).replace("```", "").strip()
        for idx, char in enumerate(cleaned):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(cleaned[idx:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        raise ValueError("O modelo online não devolveu um objeto JSON reconhecível")

    @staticmethod
    def _validate_news(news: dict[str, Any]) -> dict[str, str]:
        missing = [field for field in EXPECTED_FIELDS if field not in news]
        extra = [field for field in news if field not in EXPECTED_FIELDS]
        if missing or extra:
            raise ValueError(f"Schema generativo inválido; missing={missing}, extra={extra}")
        clean = {field: str(news.get(field, "")).strip() for field in EXPECTED_FIELDS}
        if not clean["titulo"] or not clean["lead"] or not clean["corpo"]:
            raise ValueError("A saída deve conter titulo, lead e corpo")
        return clean

    def generate(self, segment: Any, model_id: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        if model_id not in ONLINE_MODELS:
            raise ValueError(f"Gerador online desconhecido: {model_id}")
        if not self.is_available(model_id):
            raise RuntimeError(f"O gerador {model_id} não tem credenciais configuradas")

        options = overrides or {}
        temperature = float(options.get("temperature", 0.0))
        max_tokens = int(options.get("max_new_tokens", 450))
        user_prompt = build_user_prompt(segment.operational_text)
        provider = ONLINE_MODELS[model_id]["provider"]

        if provider == "google":
            raw = self._google(model_id, SYSTEM_PROMPT, user_prompt, max_tokens, temperature)
        elif provider == "together":
            raw = self._openai_compatible(
                base_url="https://api.together.xyz/v1",
                api_key=os.getenv("TOGETHER_API_KEY", ""),
                model=model_id,
                system=SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif provider == "groq":
            raw = self._groq(model_id, SYSTEM_PROMPT, user_prompt, max_tokens, temperature)
        elif provider == "iaedu":
            raw = self._iaedu(SYSTEM_PROMPT, user_prompt, max_tokens, temperature)
        else:
            raise RuntimeError(f"Provider não suportado: {provider}")

        news = self._validate_news(self._extract_first_json_object(raw))
        news.update({
            "strategy": "operational_online_fallback",
            "model_key": model_id,
            "model_version": model_id,
            "source_segment_id": segment.segment_id,
            "parse_status": "json",
            "generation_role": "operational_fallback",
        })
        return news

    def generate_selected(self, segments: list[Any], selected_ids: set[str], model_id: str,
                          overrides: dict[str, Any] | None = None) -> list[Any]:
        for segment in segments:
            if segment.segment_id not in selected_ids:
                continue
            news = self.generate(segment, model_id, overrides=overrides)
            segment.generated_news = news
            segment.generation_model_version = str(news.get("model_version", ""))
            segment.provenance["generation"] = {
                "source_segment_id": segment.segment_id,
                "model_key": news.get("model_key", ""),
                "model_version": news.get("model_version", ""),
                "strategy": "operational_online_fallback",
                "generation_role": "operational_fallback",
                "selection_source": "explicit_human_selection",
                "experimental_status": "not_selected_thesis_model",
            }
        return segments

    @staticmethod
    def _google(model_id: str, system: str, user: str, max_tokens: int, temperature: float) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError("Instale google-generativeai para utilizar Gemini") from exc
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))
        model = genai.GenerativeModel(
            model_name=model_id,
            system_instruction=system,
            generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
        )
        return model.generate_content(user).text

    @staticmethod
    def _openai_compatible(*, base_url: str, api_key: str, model: str, system: str,
                           user: str, max_tokens: int, temperature: float) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Instale openai para utilizar Together") from exc
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _groq(model_id: str, system: str, user: str, max_tokens: int, temperature: float) -> str:
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("Instale groq para utilizar o fallback Groq") from exc
        client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""

    def _iaedu(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        requests = self._requests()
        endpoint = os.getenv("IAEDU_API_ENDPOINT", "").strip()
        api_key = os.getenv("IAEDU_API_KEY", "").strip()
        channel_id = os.getenv("IAEDU_CHANNEL_ID", "").strip()
        timeout = int(os.getenv("IAEDU_TIMEOUT", "120"))
        prompt = (
            f"{system.strip()}\n\nPARÂMETROS DE GERAÇÃO:\n"
            f"- max_tokens: {max_tokens}\n- temperature: {temperature}\n\n{user.strip()}"
        )
        files = {
            "channel_id": (None, channel_id),
            "thread_id": (None, str(uuid.uuid4())),
            "user_info": (None, os.getenv("IAEDU_USER_INFO", "{}") or "{}"),
            "user_context": (None, os.getenv("IAEDU_USER_CONTEXT", "{}") or "{}"),
            "message": (None, prompt),
        }
        response = requests.post(
            endpoint,
            headers={"x-api-key": api_key},
            files=files,
            stream=True,
            timeout=timeout,
        )
        response.raise_for_status()
        chunks: list[str] = []
        last_line = ""
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8", errors="replace").strip()
            if decoded.startswith("data:"):
                decoded = decoded[5:].strip()
            if not decoded or decoded == "[DONE]":
                continue
            try:
                data = json.loads(decoded)
            except json.JSONDecodeError:
                last_line = decoded[:300]
                continue
            if data.get("type") == "error":
                raise RuntimeError(json.dumps(data, ensure_ascii=False))
            if data.get("type") == "message" and "content" in data:
                msg = data.get("content")
                if isinstance(msg, dict) and msg.get("type") == "ai" and msg.get("content"):
                    chunks.append(str(msg["content"]))
                elif isinstance(msg, str) and msg.strip():
                    chunks.append(msg.strip())
            for key in ("output", "response", "text"):
                if data.get(key):
                    chunks.append(str(data[key]).strip())
        text = "".join(chunks).strip()
        if text:
            return text
        raise RuntimeError(f"IA Edu terminou sem conteúdo gerado. {last_line}".strip())
