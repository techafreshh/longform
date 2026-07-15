import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

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


@patch("src.assembler._render_scene_clip")
@patch("src.assembler._concat_with_transitions")
@patch("src.assembler._finalize_video")
@patch("src.assembler._get_duration")
def test_assemble_video_skips_cached_clip(
    mock_duration, mock_finalize, mock_concat, mock_render, tmp_path
):
    from src.assembler import assemble_video, SceneTiming
    
    # Setup paths
    output_path = tmp_path / "output" / "video.mp4"
    image_path = tmp_path / "scene_01.png"
    image_path.write_bytes(b"mock image")
    
    # Create timings list
    timings = [
        SceneTiming(
            index=1,
            image_path=image_path,
            start_time=0.0,
            duration=3.0,
            narration="Test scene",
        )
    ]
    
    # Pre-create cached clip in clips_cache folder
    clips_cache_dir = tmp_path / "clips_cache"
    clips_cache_dir.mkdir(parents=True, exist_ok=True)
    
    clip_path = clips_cache_dir / f"clip_01_dur_3.00.mp4"
    clip_path.write_bytes(b"mock clip content" * 100) # > 1KB
    
    # Ensure cache clip modified time is greater than image modified time
    import time
    # Set image file back in time
    past_time = time.time() - 100
    import os
    os.utime(image_path, (past_time, past_time))
    # Set clip file to current time
    now_time = time.time()
    os.utime(clip_path, (now_time, now_time))
    
    mock_duration.return_value = 3.0
    
    res = assemble_video(
        timings=timings,
        voiceover_path=tmp_path / "voice.wav",
        output_path=output_path,
        transition_type="none",
        verbose=True,
    )
    
    assert res == output_path
    # _render_scene_clip should NOT be called since it is cached and valid!
    mock_render.assert_not_called()


def test_word_level_srt():
    from src.assembler import _word_level_srt
    timestamps = [
        {"word": "This", "start": 0.0, "end": 0.5},
        {"word": "is", "start": 0.5, "end": 1.0},
        {"word": "a", "start": 1.0, "end": 1.5},
        {"word": "test", "start": 1.5, "end": 2.0},
    ]
    srt = _word_level_srt(timestamps, words_per_group=4, highlight_color="#FFCC00")
    
    # We expect 4 entries since there are 4 words in the group
    # Each entry should highlight exactly one word in sequence
    assert "1\n00:00:00,000 --> 00:00:00,500\n<font color=\"#FFCC00\">THIS</font> IS A TEST" in srt
    assert "2\n00:00:00,500 --> 00:00:01,000\nTHIS <font color=\"#FFCC00\">IS</font> A TEST" in srt
    assert "3\n00:00:01,000 --> 00:00:01,500\nTHIS IS <font color=\"#FFCC00\">A</font> TEST" in srt
    assert "4\n00:00:01,500 --> 00:00:02,000\nTHIS IS A <font color=\"#FFCC00\">TEST</font>" in srt


