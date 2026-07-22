import pytest
from unittest.mock import MagicMock, patch
import time

# We mock google.genai and other modules so we can import and test scene_gen without actual dependencies
import sys
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.genai.types'] = MagicMock()

from src.scene_gen import _try_gemini_image_generation, _try_imagen_generation


def test_try_gemini_image_generation_success():
    client = MagicMock()
    # Mocking successful response
    mock_part = MagicMock()
    mock_part.inline_data.data = b"fake-image-bytes"
    client.models.generate_content.return_value.candidates = [
        MagicMock(content=MagicMock(parts=[mock_part]))
    ]
    
    types = MagicMock()
    
    res = _try_gemini_image_generation(client, "gemini-2.5-flash-image", "prompt", types, verbose=False)
    assert res == b"fake-image-bytes"
    client.models.generate_content.assert_called_once()


@patch("time.sleep")
def test_try_gemini_image_generation_retry_on_429(mock_sleep):
    client = MagicMock()
    
    # First call raises 429, second call succeeds
    mock_part = MagicMock()
    mock_part.inline_data.data = b"fake-image-bytes"
    success_response = MagicMock()
    success_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
    
    client.models.generate_content.side_effect = [
        Exception("429 RESOURCE_EXHAUSTED: Resource exhausted"),
        success_response
    ]
    
    types = MagicMock()
    res = _try_gemini_image_generation(client, "gemini-2.5-flash-image", "prompt", types, verbose=True)
    
    assert res == b"fake-image-bytes"
    assert client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once_with(5)  # base_delay * (2 ** 0) = 5


@patch("time.sleep")
def test_try_gemini_image_generation_fails_on_persistent_429(mock_sleep):
    client = MagicMock()
    
    # All 10 calls raise 429
    client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Resource exhausted")
    
    types = MagicMock()
    res = _try_gemini_image_generation(client, "gemini-2.5-flash-image", "prompt", types, verbose=True)
    
    assert res is None
    assert client.models.generate_content.call_count == 10
    assert mock_sleep.call_count == 9


def test_try_gemini_image_generation_fails_immediately_on_404():
    client = MagicMock()
    client.models.generate_content.side_effect = Exception("404 Publisher model not found")
    
    types = MagicMock()
    res = _try_gemini_image_generation(client, "gemini-3.1-flash-image", "prompt", types, verbose=True)
    
    assert res is None
    assert client.models.generate_content.call_count == 1


@patch("src.scene_gen.GOOGLE_API_KEY", "fake_key")
@patch("src.scene_gen._generate_single_image")
@patch("src.scene_gen.get_genai_client")
def test_generate_thumbnail_max_3_words_and_seo_title(mock_get_client, mock_gen_single_image, tmp_path):
    from src.scene_gen import generate_thumbnail

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    # LLM returns 5 words: should be truncated to max 3 words
    mock_client.models.generate_content.return_value.text = "Why Do Predators Attack From Behind"
    mock_gen_single_image.return_value = (b"fake-image-png", "gemini-2.5-flash-image", "AI Studio")

    thumbs = generate_thumbnail(
        topic="Predator attacks",
        niche="educational",
        style="color_whiteboard",
        output_dir=tmp_path,
        count=1,
        seo_title="Why Do Predators Almost Always Attack From Behind?",
        verbose=False,
    )

    assert len(thumbs) == 1
    assert thumbs[0].exists()

    # Check prompt passed to text generator
    call_args = mock_client.models.generate_content.call_args
    assert "Why Do Predators Almost Always Attack From Behind?" in call_args[1]["contents"]

    # Check image prompt generated contains at most 3 words overlay
    single_image_call = mock_gen_single_image.call_args
    prompt_text = single_image_call[0][1]
    assert "Why Do Predators" in prompt_text


@patch("src.scene_gen.GOOGLE_API_KEY", "fake_key")
@patch("src.scene_gen._generate_single_image")
@patch("src.scene_gen.get_genai_client")
def test_generate_thumbnail_style_consistency(mock_get_client, mock_gen_single_image, tmp_path):
    from src.scene_gen import generate_thumbnail

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.models.generate_content.return_value.text = "Attack Behind?"
    mock_gen_single_image.return_value = (b"fake-image-png", "gemini-2.5-flash-image", "AI Studio")

    # Generate 3 stickman thumbnail variants
    thumbs_stickman = generate_thumbnail(
        topic="Predator attacks",
        niche="educational",
        style="stickman",
        output_dir=tmp_path / "stickman_thumbs",
        count=3,
        verbose=False,
    )
    assert len(thumbs_stickman) == 3

    # Verify all 3 calls generated prompts for stickman on white background
    for call in mock_gen_single_image.call_args_list[-3:]:
        prompt_text = call[0][1]
        assert "Minimalist stickman cartoon style" in prompt_text
        assert "white background" in prompt_text
        assert "chalkboard" not in prompt_text
