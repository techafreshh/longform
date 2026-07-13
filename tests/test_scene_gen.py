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
    
    # All 5 calls raise 429
    client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Resource exhausted")
    
    types = MagicMock()
    res = _try_gemini_image_generation(client, "gemini-2.5-flash-image", "prompt", types, verbose=True)
    
    assert res is None
    assert client.models.generate_content.call_count == 5
    assert mock_sleep.call_count == 4


def test_try_gemini_image_generation_fails_immediately_on_404():
    client = MagicMock()
    client.models.generate_content.side_effect = Exception("404 Publisher model not found")
    
    types = MagicMock()
    res = _try_gemini_image_generation(client, "gemini-3.1-flash-image", "prompt", types, verbose=True)
    
    assert res is None
    assert client.models.generate_content.call_count == 1
