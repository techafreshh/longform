"""Tests for scriptwriter.py scene parsing and narration cleanup."""

from src.scriptwriter import _clean_narration, parse_scenes, _estimate_scene_count


class TestCleanNarration:
    """Tests for _clean_narration markdown stripping."""

    def test_strips_single_asterisk_emphasis(self):
        result = _clean_narration("what *didn't* happen")
        assert result == "what didn't happen"
        assert "*" not in result

    def test_strips_double_asterisk_bold(self):
        result = _clean_narration("the **real** answer")
        assert result == "the real answer"

    def test_strips_triple_asterisk(self):
        result = _clean_narration("this is ***important***")
        assert result == "this is important"

    def test_strips_underscore_emphasis(self):
        result = _clean_narration("the _real_ answer")
        assert result == "the real answer"

    def test_strips_double_underscore_bold(self):
        result = _clean_narration("__very__ bold")
        assert result == "very bold"

    def test_removes_scene_markers(self):
        result = _clean_narration("[SCENE: A drawing of a house] Some narration here.")
        assert "[SCENE:" not in result
        assert "Some narration here." in result

    def test_removes_html_comments(self):
        result = _clean_narration("<!-- 2:30 --> Some text")
        assert "<!--" not in result
        assert "Some text" in result

    def test_removes_markdown_headers(self):
        result = _clean_narration("## Section Title\nSome text")
        assert "##" not in result
        assert "Some text" in result

    def test_combined_real_script(self):
        """Test with actual narration from the sample run."""
        text = """<!-- 1:10 -->
[SCENE: A chalk outline of empty white space]

It's about what *didn't* happen. The comfort that never came."""
        result = _clean_narration(text)
        assert "*" not in result
        assert "didn't" in result
        assert "<!--" not in result
        assert "[SCENE:" not in result


class TestParseScenes:
    """Tests for parse_scenes with markdown emphasis in narration."""

    def test_strips_asterisks_from_parsed_narration(self):
        script = """[SCENE: A drawing]

It's about what *didn't* happen.

[SCENE: Another drawing]

How things *look* always beats how you actually *feel*."""
        scenes = parse_scenes(script)
        for scene in scenes:
            assert "*" not in scene.narration, f"Scene {scene.index} still has asterisks: {scene.narration}"
            assert "didn't" in scenes[0].narration or "look" in scenes[1].narration


class TestEstimateSceneCount:
    """Tests for _estimate_scene_count visual pacing."""

    def test_whiteboard_style_density(self):
        # 10 minutes, default/whiteboard style should yield ~40 scenes
        count = _estimate_scene_count("10 min", "color_whiteboard")
        assert count == 40

    def test_stickman_style_density(self):
        # 10 minutes, stickman style should yield ~180 scenes
        count = _estimate_scene_count("10 min", "stickman")
        assert count == 180


from unittest.mock import MagicMock, patch

@patch("src.scriptwriter._get_client")
def test_generate_script_resume(mock_get_client, tmp_path):
    from src.scriptwriter import generate_script
    
    # 1. Setup mocked OpenRouter client and response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="[SCENE: Continuation] This is the rest of the script."))
    ]
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client
    
    # 2. Write a partial script file
    partial_path = tmp_path / "script.md"
    partial_path.write_text("[SCENE: Introduction] This is the start of the script.", encoding="utf-8")
    
    # 3. Call generate_script with resume_partial=True
    script = generate_script(
        topic="test topic",
        niche="test niche",
        research="mock research content",
        output_path=partial_path,
        resume_partial=True,
        verbose=True
    )
    
    # 4. Verify mock calls and merged content
    mock_client.chat.completions.create.assert_called_once()
    called_messages = mock_client.chat.completions.create.call_args[1]["messages"]
    
    # Check that it constructed the history correctly
    assert len(called_messages) == 4
    assert called_messages[0]["role"] == "system"
    assert called_messages[1]["role"] == "user"
    assert called_messages[2]["role"] == "assistant"
    assert called_messages[2]["content"] == "[SCENE: Introduction] This is the start of the script."
    assert "cut off" in called_messages[3]["content"]
    
    # Check raw_text merged successfully
    assert script.raw_text == "[SCENE: Introduction] This is the start of the script.\n\n[SCENE: Continuation] This is the rest of the script."
    assert len(script.scenes) == 2
    assert script.scenes[0].description == "Introduction"
    assert script.scenes[1].description == "Continuation"


