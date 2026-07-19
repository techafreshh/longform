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
@patch("src.assembler._concat_and_finalize")
@patch("src.assembler._get_duration")
def test_assemble_video_skips_cached_clip(
    mock_duration, mock_concat_finalize, mock_render, tmp_path
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
    # Words 1-3 extend to the next word's start (which equals their end here — no gap)
    assert '1\n00:00:00,000 --> 00:00:00,500\n<font color="#FFCC00">THIS</font> IS A TEST' in srt
    assert '2\n00:00:00,500 --> 00:00:01,000\nTHIS <font color="#FFCC00">IS</font> A TEST' in srt
    assert '3\n00:00:01,000 --> 00:00:01,500\nTHIS IS <font color="#FFCC00">A</font> TEST' in srt
    # Last word in last group uses its own end time
    assert '4\n00:00:01,500 --> 00:00:02,000\nTHIS IS A <font color="#FFCC00">TEST</font>' in srt


def test_word_level_srt_gap_free():
    """Verify that gaps between words are bridged — each cue extends to the next word's start."""
    from src.assembler import _word_level_srt
    timestamps = [
        {"word": "Hello", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.8, "end": 1.2},   # 400ms gap after "Hello"
        {"word": "this", "start": 1.5, "end": 1.9},     # 300ms gap after "world"
    ]
    srt = _word_level_srt(timestamps, words_per_group=3, highlight_color="#FFCC00")

    # Word 1 ("Hello") should extend to word 2's start (0.8), NOT end at 0.4
    assert '00:00:00,000 --> 00:00:00,800' in srt
    # Word 2 ("world") should extend to word 3's start (1.5), NOT end at 1.2
    assert '00:00:00,800 --> 00:00:01,500' in srt
    # Word 3 ("this") is last in last group — uses its own end (1.9)
    assert '00:00:01,500 --> 00:00:01,899' in srt


def test_word_level_srt_multi_group_overlap():
    """Verify overlap buffer between groups prevents inter-group flashing."""
    from src.assembler import _word_level_srt
    timestamps = [
        {"word": "First", "start": 0.0, "end": 0.5},
        {"word": "group", "start": 0.5, "end": 1.0},
        # Group boundary here
        {"word": "Second", "start": 1.5, "end": 2.0},
        {"word": "group", "start": 2.0, "end": 2.5},
    ]
    srt = _word_level_srt(timestamps, words_per_group=2, highlight_color="#FFCC00")

    # Last word of first group ("group") should extend toward next group start (1.5)
    # with overlap buffer of 0.05, so end = max(1.0, 1.5 - 0.05) = 1.45
    assert '00:00:00,500 --> 00:00:01,449' in srt


def test_word_level_srt_strips_asterisks():
    """Verify that markdown emphasis markers are stripped from subtitle text."""
    from src.assembler import _word_level_srt
    timestamps = [
        {"word": "what", "start": 0.0, "end": 0.5},
        {"word": "*didn't*", "start": 0.5, "end": 1.0},
        {"word": "happen", "start": 1.0, "end": 1.5},
    ]
    srt = _word_level_srt(timestamps, words_per_group=3, highlight_color="#FFCC00")

    # The asterisks should be stripped
    assert "DIDN'T" in srt
    assert "*" not in srt



@patch("src.assembler._render_scene_clip")
@patch("src.assembler._concat_and_finalize")
@patch("src.assembler._get_duration")
def test_assemble_video_renders_parallel(
    mock_duration, mock_concat_finalize, mock_render, tmp_path
):
    from src.assembler import assemble_video, SceneTiming
    
    # Setup paths
    output_path = tmp_path / "output" / "video.mp4"
    image_path1 = tmp_path / "scene_01.png"
    image_path1.write_bytes(b"mock image 1")
    image_path2 = tmp_path / "scene_02.png"
    image_path2.write_bytes(b"mock image 2")
    
    # Create timings list
    timings = [
        SceneTiming(index=1, image_path=image_path1, start_time=0.0, duration=3.0, narration="Test scene 1"),
        SceneTiming(index=2, image_path=image_path2, start_time=3.0, duration=2.5, narration="Test scene 2"),
    ]
    
    mock_duration.return_value = 5.5
    
    res = assemble_video(
        timings=timings,
        voiceover_path=tmp_path / "voice.wav",
        output_path=output_path,
        transition_type="none",
        max_workers=2,
        verbose=True,
    )
    
    assert res == output_path
    # _render_scene_clip should be called twice since we have 2 uncached scenes
    assert mock_render.call_count == 2


def test_build_scene_timings_with_transitions(tmp_path):
    from src.assembler import build_scene_timings, _scene_level_srt
    
    # Create dummy images
    image_path1 = tmp_path / "scene_01.png"
    image_path1.write_bytes(b"mock image 1")
    image_path2 = tmp_path / "scene_02.png"
    image_path2.write_bytes(b"mock image 2")
    
    scenes = [
        {"index": 1, "description": "Scene 1", "narration": "First scene narration text."},
        {"index": 2, "description": "Scene 2", "narration": "Second scene narration text."},
    ]
    
    voice_segments = [
        {"index": 1, "duration": 5.0},
        {"index": 2, "duration": 4.0},
    ]
    
    # Case 1: transition_type = "none"
    timings_none = build_scene_timings(
        scenes=scenes,
        voice_segments=voice_segments,
        image_dir=tmp_path,
        transition_type="none",
        transition_duration=0.5
    )
    
    assert len(timings_none) == 2
    assert timings_none[0].duration == 5.0
    assert timings_none[0].voice_duration == 5.0
    assert timings_none[1].duration == 4.0
    assert timings_none[1].voice_duration == 4.0
    
    # Case 2: transition_type = "fade", transition_duration = 0.5
    timings_fade = build_scene_timings(
        scenes=scenes,
        voice_segments=voice_segments,
        image_dir=tmp_path,
        transition_type="fade",
        transition_duration=0.5
    )
    
    assert len(timings_fade) == 2
    # First scene duration should be extended by transition_duration (5.0 + 0.5 = 5.5)
    assert timings_fade[0].duration == 5.5
    assert timings_fade[0].voice_duration == 5.0
    # Last scene duration should NOT be extended (remains 4.0)
    assert timings_fade[1].duration == 4.0
    assert timings_fade[1].voice_duration == 4.0
    
    # Verify fallback scene-level subtitles use voice_duration instead of clip duration
    srt_content = _scene_level_srt(timings_fade)
    # Timing 1 starts at 0.0, has voice_duration = 5.0, so subtitle should end at 5.0
    # First cue starts around 0.0 and splits narration into chunks
    # Verify we don't bleed into 5.5s
    assert "00:00:05,000" in srt_content
    # The subtitle of scene 2 starts at 5.0 and ends at 9.0 (duration 4.0)
    assert "00:00:05,000 -->" in srt_content
    assert "00:00:09,000" in srt_content


def test_build_ffmpeg_cmd(tmp_path):
    from src.assembler import _build_ffmpeg_cmd
    
    clips = [tmp_path / "clip1.mp4", tmp_path / "clip2.mp4"]
    voiceover_path = tmp_path / "voiceover.wav"
    output_path = tmp_path / "output.mp4"
    bgm_path = tmp_path / "bgm.mp3"
    bgm_path.write_bytes(b"bgm content")
    subtitle_path = tmp_path / "subtitles.srt"
    subtitle_path.write_bytes(b"sub content")
    
    # 1. Test standard transition + subtitles + BGM
    cmd = _build_ffmpeg_cmd(
        clips=clips,
        voiceover_path=voiceover_path,
        output_path=output_path,
        transition_type="fade",
        transition_duration=0.5,
        subtitle_path=subtitle_path,
        bgm_path=bgm_path,
        bgm_volume=0.15,
        subtitle_style={"subtitle_color": "#FFCC00"},
        encoder="libx264"
    )
    
    # Verify inputs are mapped
    assert "-i" in cmd
    assert str(clips[0]) in cmd
    assert str(clips[1]) in cmd
    assert str(voiceover_path) in cmd
    assert str(bgm_path) in cmd
    
    # Verify filter graph
    assert "-filter_complex" in cmd
    # Should contain xfade, subtitles, and amix
    filter_graph = cmd[cmd.index("-filter_complex") + 1]
    assert "xfade=transition=fade" in filter_graph
    assert "subtitles=" in filter_graph
    assert "amix=inputs=2" in filter_graph
    
    # Verify mapping
    assert "-map" in cmd
    # Should map the final video stream (which should be [vfinal] after subtitles filter)
    assert "[vfinal]" in cmd
    # Should map the audio track [aout]
    assert "[aout]" in cmd
    
    # Verify encoder parameters
    assert "-c:v" in cmd
    assert "libx264" in cmd
    assert "-preset" in cmd
    assert "medium" in cmd
    
    # 2. Test single clip + no transitions + no BGM
    cmd_single = _build_ffmpeg_cmd(
        clips=[clips[0]],
        voiceover_path=voiceover_path,
        output_path=output_path,
        transition_type="none",
        transition_duration=0.0,
        subtitle_path=None,
        bgm_path=None,
        encoder="h264_nvenc"
    )
    
    # No filter graph since 1 clip, no subtitles, no BGM
    assert "-filter_complex" not in cmd_single
    assert "-map" in cmd_single
    assert "0:v" in cmd_single
    assert "1:a" in cmd_single
    assert "h264_nvenc" in cmd_single


def test_word_level_srt_with_delay():
    from src.assembler import _word_level_srt
    timestamps = [
        {"word": "First", "start": 10.0, "end": 10.5},
        {"word": "second", "start": 20.0, "end": 20.5},
        {"word": "third", "start": 65.0, "end": 65.5},
        {"word": "fourth", "start": 70.0, "end": 70.5},
    ]
    # Delay for 60 seconds (1 minute)
    srt = _word_level_srt(timestamps, words_per_group=2, highlight_color="#FFCC00", start_time_delay=60.0)
    
    # "First" (10.0) and "second" (20.0) should be skipped
    assert "FIRST" not in srt
    assert "SECOND" not in srt
    # "third" (65.0) and "fourth" (70.0) should be included
    assert "1\n00:01:05,000" in srt
    assert "THIRD" in srt
    assert "2\n00:01:10,000" in srt
    assert "FOURTH" in srt


def test_scene_level_srt_with_delay(tmp_path):
    from src.assembler import build_scene_timings, _scene_level_srt
    
    image_path1 = tmp_path / "scene_01.png"
    image_path1.write_bytes(b"mock image 1")
    image_path2 = tmp_path / "scene_02.png"
    image_path2.write_bytes(b"mock image 2")
    
    scenes = [
        {"index": 1, "description": "Scene 1", "narration": "First scene narration text."},
        {"index": 2, "description": "Scene 2", "narration": "Second scene narration text."},
    ]
    
    # Total duration = 70.0s for scene 1 (so chunks start from 0 to 70) and 20.0s for scene 2 (from 70 to 90)
    voice_segments = [
        {"index": 1, "duration": 70.0},
        {"index": 2, "duration": 20.0},
    ]
    
    timings = build_scene_timings(
        scenes=scenes,
        voice_segments=voice_segments,
        image_dir=tmp_path,
        transition_type="none"
    )
    
    # Delay for 60 seconds (1 minute)
    srt = _scene_level_srt(timings, start_time_delay=60.0)
    
    # The first scene has narration "First scene narration text." (4 words).
    # Chunks are split into size 10, so 1 chunk of duration 70.0s.
    # The chunk starts at 0.0, which is < 60.0, so it is skipped.
    assert "FIRST SCENE" not in srt
    
    # Second scene starts at 70.0s, which is >= 60.0s, so it should be included.
    assert "SECOND SCENE" in srt






