from backend.integration.online_generation_service import OnlineFallbackGenerationService, ONLINE_MODELS


def test_online_model_catalog_is_restricted():
    assert set(ONLINE_MODELS) == {
        "iaedu",
        "gemini-2.5-pro",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "llama-3.3-70b-versatile",
    }


def test_json_extraction_and_contract():
    raw = 'texto antes {"antetitulo":"","titulo":"Título","superlead":"","lead":"Lead","corpo":"Corpo"} depois'
    data = OnlineFallbackGenerationService._extract_first_json_object(raw)
    clean = OnlineFallbackGenerationService._validate_news(data)
    assert clean["titulo"] == "Título"
    assert set(clean) == {"antetitulo", "titulo", "superlead", "lead", "corpo"}
