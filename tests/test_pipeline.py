import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import sys

# Mock google.genai and other modules before imports
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.genai.types'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['google.auth'] = MagicMock()
sys.modules['googleapiclient'] = MagicMock()
sys.modules['googleapiclient.discovery'] = MagicMock()
sys.modules['googleapiclient.http'] = MagicMock()

from src.config import ProjectPaths
from src.pipeline import (
    run_stage_research,
    run_stage_script,
    run_stage_voice,
    run_stage_scenes,
    run_stage_assembly,
    run_stage_thumbnails,
    run_stage_seo,
)
from src.scriptwriter import Script, Scene
from src.voice import VoiceResult, VoiceSegment


@pytest.fixture
def temp_project_paths(tmp_path):
    paths = ProjectPaths(
        base_dir=tmp_path,
        niche="test-niche",
        topic_slug="test-topic",
    )
    paths.ensure_dirs()
    return paths


@patch("src.pipeline.research_topic")
def test_run_stage_research_skips_when_exists(mock_research_topic, temp_project_paths):
    # Setup existing research file
    research_content = "Existing research content"
    temp_project_paths.research_file.write_text(research_content, encoding="utf-8")

    # Run stage
    res = run_stage_research(
        paths=temp_project_paths,
        topic="test-topic",
        niche="test-niche",
        force=False,
    )

    # Verification
    assert res == research_content
    mock_research_topic.assert_not_called()


@patch("src.pipeline.research_topic")
def test_run_stage_research_generates_when_missing_or_forced(mock_research_topic, temp_project_paths):
    mock_research_topic.return_value = "Generated research"

    # Run stage when missing
    res = run_stage_research(
        paths=temp_project_paths,
        topic="test-topic",
        niche="test-niche",
        force=False,
    )

    assert res == "Generated research"
    mock_research_topic.assert_called_once()

    # Reset mock and run stage when forced even if exists
    mock_research_topic.reset_mock()
    temp_project_paths.research_file.write_text("Old research", encoding="utf-8")

    res_forced = run_stage_research(
        paths=temp_project_paths,
        topic="test-topic",
        niche="test-niche",
        force=True,
    )

    assert res_forced == "Generated research"
    mock_research_topic.assert_called_once()


@patch("src.pipeline.generate_script")
def test_run_stage_script_skips_when_exists(mock_generate_script, temp_project_paths):
    # Setup existing script & scenes.json
    script_text = "## Scene 1\n[Illustration: None]\nNarration: hello"
    temp_project_paths.script_file.write_text(script_text, encoding="utf-8")
    
    scenes_json = temp_project_paths.project_dir / "scenes.json"
    scenes_json.write_text("[]", encoding="utf-8")

    script_obj = run_stage_script(
        paths=temp_project_paths,
        topic="test-topic",
        niche="test-niche",
        research_text="some research",
        force=False,
    )

    assert isinstance(script_obj, Script)
    assert script_obj.raw_text == script_text
    mock_generate_script.assert_not_called()


@patch("src.pipeline.generate_script")
def test_run_stage_script_generates_when_missing(mock_generate_script, temp_project_paths):
    dummy_script = Script(
        raw_text="dummy script text",
        scenes=[Scene(index=1, description="desc", narration="narr")],
        topic="test-topic",
        niche="test-niche",
        style="color_whiteboard"
    )
    mock_generate_script.return_value = dummy_script

    script_obj = run_stage_script(
        paths=temp_project_paths,
        topic="test-topic",
        niche="test-niche",
        research_text="some research",
        force=False,
    )

    assert script_obj.raw_text == "dummy script text"
    assert (temp_project_paths.project_dir / "scenes.json").exists()
    mock_generate_script.assert_called_once()


@patch("src.pipeline.generate_voice_fish")
def test_run_stage_voice_skips_when_exists(mock_generate_voice_fish, temp_project_paths):
    # Write existing voiceover file and timestamps JSON
    temp_project_paths.voiceover_file.write_text("audio-bytes", encoding="utf-8")
    
    timestamps_data = {
        "total_duration": 12.5,
        "segments": [
            {"index": 1, "audio_path": str(temp_project_paths.audio_dir / "scene_01.wav"), "duration": 12.5}
        ],
        "word_timestamps": []
    }
    with open(temp_project_paths.timestamps_file, "w", encoding="utf-8") as f:
        json.dump(timestamps_data, f)

    voice_res = run_stage_voice(
        paths=temp_project_paths,
        scenes_data=[{"index": 1}],
        tts_engine="fish",
        force=False,
    )

    assert voice_res.total_duration == 12.5
    assert len(voice_res.segments) == 1
    mock_generate_voice_fish.assert_not_called()


@patch("src.pipeline.generate_scenes")
def test_run_stage_scenes_skips_when_all_exist(mock_generate_scenes, temp_project_paths):
    # Setup scene image files
    img1 = temp_project_paths.scenes_dir / "scene_01.png"
    img2 = temp_project_paths.scenes_dir / "scene_02.png"
    img1.write_text("data", encoding="utf-8")
    img2.write_text("data", encoding="utf-8")

    scenes_data = [{"index": 1}, {"index": 2}]

    res = run_stage_scenes(
        paths=temp_project_paths,
        scenes_data=scenes_data,
        style="color_whiteboard",
        force=False,
    )

    assert len(res) == 2
    assert res[0] == img1
    assert res[1] == img2
    mock_generate_scenes.assert_not_called()


@patch("src.pipeline.assemble_video")
@patch("src.pipeline.generate_subtitles")
def test_run_stage_assembly_skips_when_exists(mock_gen_sub, mock_assemble, temp_project_paths):
    temp_project_paths.final_video.write_text("mp4-data", encoding="utf-8")

    voice_result = VoiceResult(
        segments=[VoiceSegment(index=1, text="hello", audio_path=Path("dummy"), duration=5.0)],
        combined_audio=Path("dummy"),
        total_duration=5.0,
        timestamps=[]
    )

    res = run_stage_assembly(
        paths=temp_project_paths,
        scenes_data=[{"index": 1}],
        voice_result=voice_result,
        force=False,
    )

    assert res == temp_project_paths.final_video
    mock_gen_sub.assert_not_called()
    mock_assemble.assert_not_called()


@patch("src.pipeline.generate_thumbnail")
def test_run_stage_thumbnails_skips_when_exists(mock_gen_thumb, temp_project_paths):
    # Write dummy thumbnail
    thumb = temp_project_paths.thumbnail_dir / "test-topic_thumbnail_01.png"
    thumb.write_text("png-data", encoding="utf-8")

    res = run_stage_thumbnails(
        paths=temp_project_paths,
        topic="test-topic",
        niche="test-niche",
        style="color_whiteboard",
        force=False,
    )

    assert len(res) == 1
    assert res[0] == thumb
    mock_gen_thumb.assert_not_called()


@patch("src.pipeline.generate_seo")
def test_run_stage_seo_skips_when_exists(mock_generate_seo, temp_project_paths):
    seo_data = {"title": "Test Title", "description": "desc", "tags": []}
    with open(temp_project_paths.seo_file, "w", encoding="utf-8") as f:
        json.dump(seo_data, f)

    res = run_stage_seo(
        paths=temp_project_paths,
        topic="test-topic",
        niche="test-niche",
        script_text="script",
        style="color_whiteboard",
        force=False,
    )

    assert res == seo_data
    mock_generate_seo.assert_not_called()


@patch("src.pipeline.generate_scenes")
def test_run_stage_scenes_respects_resume_from_scene(mock_generate_scenes, temp_project_paths):
    scenes_data = [{"index": 1}, {"index": 2}]
    
    # Only scene 2 exists
    img2 = temp_project_paths.scenes_dir / "scene_02.png"
    img2.write_text("data", encoding="utf-8")
    
    # We resume from 2, so scene 1 is ignored and scene 2 exists, so we should skip generation
    res = run_stage_scenes(
        paths=temp_project_paths,
        scenes_data=scenes_data,
        style="color_whiteboard",
        force=False,
        resume_from_scene=2,
    )
    
    assert len(res) == 2
    mock_generate_scenes.assert_not_called()


@patch("src.pipeline.assemble_video")
@patch("src.pipeline.generate_subtitles")
@patch("src.pipeline.build_scene_timings")
def test_run_stage_assembly_generates_subtitles(mock_build_timings, mock_gen_sub, mock_assemble, temp_project_paths):
    # Setup dummy voiceover file and timings
    voice_result = VoiceResult(
        segments=[VoiceSegment(index=1, text="hello", audio_path=Path("dummy"), duration=5.0)],
        combined_audio=Path("dummy"),
        total_duration=5.0,
        timestamps=[]
    )
    
    mock_build_timings.return_value = []
    
    # We create a scene image so that assembler can check it
    scene_img = temp_project_paths.scenes_dir / "scene_01.png"
    scene_img.write_text("data", encoding="utf-8")

    run_stage_assembly(
        paths=temp_project_paths,
        scenes_data=[{"index": 1}],
        voice_result=voice_result,
        force=True,
        skip_subtitles=False,
    )

    mock_gen_sub.assert_called_once()
    mock_assemble.assert_called_once()
    # Check that subtitle_path is not None in the assemble call
    kwargs = mock_assemble.call_args[1]
    assert kwargs["subtitle_path"] is not None


@patch("src.pipeline.assemble_video")
@patch("src.pipeline.generate_subtitles")
@patch("src.pipeline.build_scene_timings")
def test_run_stage_assembly_skips_subtitles_when_requested(mock_build_timings, mock_gen_sub, mock_assemble, temp_project_paths):
    # Setup dummy voiceover file and timings
    voice_result = VoiceResult(
        segments=[VoiceSegment(index=1, text="hello", audio_path=Path("dummy"), duration=5.0)],
        combined_audio=Path("dummy"),
        total_duration=5.0,
        timestamps=[]
    )
    
    mock_build_timings.return_value = []
    
    # We create a scene image so that assembler can check it
    scene_img = temp_project_paths.scenes_dir / "scene_01.png"
    scene_img.write_text("data", encoding="utf-8")

    run_stage_assembly(
        paths=temp_project_paths,
        scenes_data=[{"index": 1}],
        voice_result=voice_result,
        force=True,
        skip_subtitles=True,
    )

    mock_gen_sub.assert_not_called()
    mock_assemble.assert_called_once()
    # Check that subtitle_path is None in the assemble call
    kwargs = mock_assemble.call_args[1]
    assert kwargs["subtitle_path"] is None


@patch("src.pipeline.assemble_video")
@patch("src.pipeline.build_scene_timings")
def test_run_stage_assembly_forces_ken_burns_false_for_stickman(mock_build_timings, mock_assemble, temp_project_paths):
    voice_result = VoiceResult(
        segments=[VoiceSegment(index=1, text="hello", audio_path=Path("dummy"), duration=5.0)],
        combined_audio=Path("dummy"),
        total_duration=5.0,
        timestamps=[]
    )
    mock_build_timings.return_value = []

    run_stage_assembly(
        paths=temp_project_paths,
        scenes_data=[{"index": 1}],
        voice_result=voice_result,
        style="stickman",
        ken_burns=True,
        force=True,
    )

    mock_assemble.assert_called_once()
    kwargs = mock_assemble.call_args[1]
    assert kwargs["ken_burns"] is False

