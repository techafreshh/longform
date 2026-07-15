import sys
from unittest.mock import MagicMock, patch
import pytest

# Mock optional dependency module so the test file can be imported/run without it
sys.modules['youtube_transcript_api'] = MagicMock()

from src.youtube_reference import search_youtube_videos, download_youtube_transcript


@patch("requests.get")
def test_search_youtube_videos_success(mock_get):
    mock_response = MagicMock()
    mock_response.text = """
    <html>
    <body>
    <script>
    var ytInitialData = {
        "contents": {
            "twoColumnSearchResultsRenderer": {
                "primaryContents": {
                    "sectionListRenderer": {
                        "contents": [
                            {
                                "itemSectionRenderer": {
                                    "contents": [
                                        {
                                            "videoRenderer": {
                                                "videoId": "abc_1234567",
                                                "title": {
                                                    "runs": [{"text": "Hannibal's Battle at Trebbia"}]
                                                }
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                }
            }
        }
    };
    </script>
    </body>
    </html>
    """
    mock_get.return_value = mock_response

    results = search_youtube_videos("Hannibal Trebbia", count=1)
    assert len(results) == 1
    assert results[0]["video_id"] == "abc_1234567"
    assert results[0]["title"] == "Hannibal's Battle at Trebbia"
    mock_get.assert_called_once()


@patch("requests.get")
def test_search_youtube_videos_fallback(mock_get):
    mock_response = MagicMock()
    mock_response.text = """
    <a href="/watch?v=abc12345678">Video 1</a>
    <a href="/watch?v=def12345678">Video 2</a>
    """
    mock_get.return_value = mock_response

    results = search_youtube_videos("Fallback Search", count=2)
    assert len(results) == 2
    assert results[0]["video_id"] == "abc12345678"
    assert results[1]["video_id"] == "def12345678"


@patch("youtube_transcript_api.YouTubeTranscriptApi.list_transcripts")
def test_download_youtube_transcript_success(mock_list_transcripts, tmp_path):
    mock_transcript = MagicMock()
    mock_transcript.fetch.return_value = [
        {"text": "Hello, welcome to this video.", "start": 0.0, "duration": 2.0},
        {"text": "Today we are talking about Rome&#39;s strategy.", "start": 2.0, "duration": 3.0}
    ]
    
    mock_transcript_list = MagicMock()
    mock_transcript_list.find_transcript.return_value = mock_transcript
    mock_list_transcripts.return_value = mock_transcript_list

    dest_dir = tmp_path / "reference_scripts"
    dest_path = download_youtube_transcript("abc1234567", "Roman History Video", dest_dir, verbose=False)
    
    assert dest_path is not None
    assert dest_path.exists()
    content = dest_path.read_text(encoding="utf-8")
    
    assert "Rome's strategy" in content
    assert "Hello, welcome to this video." in content
    assert "Today we are talking about Rome's strategy." in content
