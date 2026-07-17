import sys
from unittest.mock import MagicMock, patch
import pytest
from pathlib import Path
import json

# Setup sys.modules mocks before importing youtube_clip
sys.modules['yt_dlp'] = MagicMock()

from src.youtube_clip import download_reference_clip_optional, analyze_reference_clip_with_transcript

@patch("yt_dlp.YoutubeDL")
def test_download_reference_clip_success(mock_ydl, tmp_path):
    # Setup mocks
    instance = mock_ydl.return_value.__enter__.return_value
    
    # We want to simulate successful download
    dest_file = tmp_path / "reference_clips" / "test_clip.mp4"
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write dummy file so it exists
    dest_file.write_bytes(b"dummy mp4 data" * 100)
    
    res = download_reference_clip_optional("abc_123", dest_file, duration_seconds=30, verbose=False)
    
    assert res == dest_file
    mock_ydl.assert_called_once()
    instance.download.assert_called_once_with(["https://www.youtube.com/watch?v=abc_123"])

@patch.dict(sys.modules, {'yt_dlp': None})
def test_download_reference_clip_no_ytdlp(tmp_path):
    # Unimport to trigger ImportError
    with patch("builtins.__import__", side_effect=ImportError("No module named 'yt_dlp'")):
        dest_file = tmp_path / "reference_clips" / "test_clip.mp4"
        res = download_reference_clip_optional("abc_123", dest_file, duration_seconds=30, verbose=False)
        assert res is None

def test_analyze_reference_clip_success():
    mock_client = MagicMock()
    mock_file = MagicMock()
    mock_file.name = "files/test_file_id"
    mock_file.state.name = "ACTIVE"
    mock_client.files.upload.return_value = mock_file
    mock_client.files.get.return_value = mock_file

    mock_response = MagicMock()
    mock_response.text = """
    {
      "background_style": "Green chalkboard with heavy chalk dust and wooden board outline.",
      "drawing_style": "Sloppy hand-drawn yellow and white chalk sketches.",
      "pacing_style": "Fast cuts every 10 seconds with sudden camera slide transitions.",
      "narrative_pacing": "Rapid-fire speech with short 100ms breaks between points.",
      "reproduction_prompt_prefix": "Green chalkboard animation style, heavy chalk lines, simple hand drawing outlines..."
    }
    """
    mock_client.models.generate_content.return_value = mock_response

    clip_path = Path("reference_clips/fake_clip.mp4")
    analysis = analyze_reference_clip_with_transcript(
        client=mock_client,
        clip_path=clip_path,
        transcript_text="Dummy transcript text of the reference video.",
        verbose=False
    )

    assert analysis is not None
    assert analysis["background_style"] == "Green chalkboard with heavy chalk dust and wooden board outline."
    assert analysis["reproduction_prompt_prefix"] == "Green chalkboard animation style, heavy chalk lines, simple hand drawing outlines..."
    mock_client.files.upload.assert_called_once_with(file=clip_path)
    mock_client.models.generate_content.assert_called_once()
    mock_client.files.delete.assert_called_once_with(name="files/test_file_id")
