import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# We mock config before importing assembler to avoid errors
sys.modules['src.config'] = MagicMock()

from src.assembler import _detect_h264_encoder, SceneTiming

def test_scene_timing_dataclass():
    timing = SceneTiming(
        index=1,
        image_path=Path("scene_01.png"),
        start_time=0.0,
        duration=5.0,
        narration="Hello world",
    )
    assert timing.index == 1
    assert timing.duration == 5.0

@patch("subprocess.run")
def test_detect_h264_encoder_nvenc(mock_run):
    mock_response = MagicMock()
    mock_response.stdout = "V..... h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)"
    mock_run.return_value = mock_response
    
    encoder = _detect_h264_encoder()
    assert encoder == "h264_nvenc"
    mock_run.assert_called_once()

@patch("subprocess.run")
def test_detect_h264_encoder_fallback(mock_run):
    mock_response = MagicMock()
    mock_response.stdout = "V..... libx264              libx264 H.264 encoder"
    mock_run.return_value = mock_response
    
    encoder = _detect_h264_encoder()
    assert encoder == "libx264"
