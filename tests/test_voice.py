"""Tests for voice.py TTS text preparation functions."""

from src.voice import _prepare_text_for_tts, _strip_markdown_for_tts


class TestPrepareTextForTts:
    """Tests for _prepare_text_for_tts (Fish Audio S2)."""

    def test_strips_markdown_bold(self):
        assert _prepare_text_for_tts("what *didn't* happen") == "what didn't happen"

    def test_strips_markdown_double_bold(self):
        assert _prepare_text_for_tts("how things **look** always") == "how things look always"

    def test_strips_markdown_italic_underscore(self):
        assert _prepare_text_for_tts("the _real_ answer") == "the real answer"

    def test_converts_ellipsis_to_pause(self):
        result = _prepare_text_for_tts("They just... didn't have the tools.")
        assert "[pause]" in result
        assert "..." not in result

    def test_converts_em_dash_to_pause(self):
        result = _prepare_text_for_tts("Not from pain \u2014 but from fear.")
        assert "[pause]" in result
        assert "\u2014" not in result

    def test_converts_double_newline_to_long_pause(self):
        result = _prepare_text_for_tts("First paragraph.\n\nSecond paragraph.")
        assert "[long pause]" in result
        assert "\n" not in result

    def test_collapses_single_newlines(self):
        result = _prepare_text_for_tts("Line one.\nLine two.")
        assert "\n" not in result
        assert "Line one. Line two." == result

    def test_cleans_multiple_spaces(self):
        result = _prepare_text_for_tts("word   word")
        assert "   " not in result

    def test_strips_markdown_headers(self):
        result = _prepare_text_for_tts("## Header\nSome text")
        assert "##" not in result
        assert "Some text" in result

    def test_strips_list_bullets(self):
        result = _prepare_text_for_tts("- item one\n* item two")
        assert result == "item one item two"

    def test_combined_real_narration(self):
        """Test with actual narration from the sample run."""
        text = "It's about what *didn't* happen. The comfort that never came. The listening that never landed. It's the blank space in the family photo."
        result = _prepare_text_for_tts(text)
        assert "*" not in result
        assert "didn't" in result

    def test_preserves_pause_tags_already_present(self):
        """If the script already contains [pause] tags, they should survive."""
        text = "I thought I was ready. [pause] I wasn't."
        result = _prepare_text_for_tts(text)
        assert "[pause]" in result
        assert "I wasn't." in result


class TestStripMarkdownForTts:
    """Tests for _strip_markdown_for_tts (Qwen — markdown only, keeps punctuation)."""

    def test_strips_asterisks(self):
        assert _strip_markdown_for_tts("what *didn't* happen") == "what didn't happen"

    def test_preserves_ellipsis(self):
        result = _strip_markdown_for_tts("They just... didn't have the tools.")
        assert "..." in result

    def test_preserves_em_dash(self):
        result = _strip_markdown_for_tts("Not from pain \u2014 but from fear.")
        assert "\u2014" in result


def test_pad_wav_with_silence():
    from unittest.mock import MagicMock, patch, mock_open
    from pathlib import Path
    from src.voice import _pad_wav_with_silence
    
    with patch("subprocess.run") as mock_run, \
         patch("src.voice._create_silence") as mock_create_silence, \
         patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.replace") as mock_replace, \
         patch("builtins.open", mock_open()) as mock_file:
         
        mock_run.return_value.returncode = 0
        
        # Test file path
        path = Path("/mock/dir/scene_01.wav")
        
        _pad_wav_with_silence(path, 0.5)
        
        # Verify silence segment creation and FFMPEG concat command execution
        mock_create_silence.assert_called_once_with(Path("/mock/dir/silence_scene_01.wav"), duration=0.5)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "ffmpeg" in args
        assert "-f" in args
        assert "concat" in args
        mock_replace.assert_called_once()

