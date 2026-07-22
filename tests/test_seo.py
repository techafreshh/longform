import pytest
from unittest.mock import MagicMock, patch
import json

from src.seo import _remove_emojis, generate_seo


def test_remove_emojis():
    raw_text = "🔥 Amazing Video Title 🎬 [Must Watch!] 💡"
    clean_text = _remove_emojis(raw_text)
    assert clean_text == "Amazing Video Title [Must Watch!]"
    assert "🔥" not in clean_text
    assert "🎬" not in clean_text
    assert "💡" not in clean_text


@patch("src.seo.OPENROUTER_API_KEY", "fake_key")
@patch("src.seo.OpenAI")
def test_generate_seo_strips_emojis_and_includes_research(mock_openai_cls, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "title": "🚀 Top 5 Predator Secrets 🦁",
        "description": "Learn how predators attack. 📌 Sources & References:\n- Journal of Zoology 2023 🔬",
        "tags": ["predators 🐅", "wildlife 🌿", "biology"],
        "category_id": "27"
    })
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    seo_data = generate_seo(
        topic="Predator Attacks",
        niche="wildlife",
        script_text="Script text content...",
        style="stickman",
        research_text="Deep research info: Journal of Zoology study on ambush predators.",
        output_path=tmp_path / "seo.json",
        verbose=False,
    )

    assert seo_data["title"] == "Top 5 Predator Secrets"
    assert "🚀" not in seo_data["title"]
    assert "🦁" not in seo_data["title"]
    assert "🚀" not in seo_data["description"]
    assert "🔬" not in seo_data["description"]
    assert seo_data["tags"] == ["predators", "wildlife", "biology"]

    # Verify research content was passed into openrouter prompt
    call_args = mock_client.chat.completions.create.call_args[1]
    user_content = call_args["messages"][1]["content"]
    assert "Research Material & Sources:" in user_content
    assert "Journal of Zoology study" in user_content
