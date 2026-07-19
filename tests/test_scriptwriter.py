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

