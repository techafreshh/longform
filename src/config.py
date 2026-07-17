"""Configuration, API keys, style presets, and project paths."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
FISH_API_KEY = os.getenv("FISH_API_KEY", "")
FISH_VOICE_ID = os.getenv("FISH_VOICE_ID", "")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "")

# Vertex AI settings
USE_VERTEX = os.getenv("USE_VERTEX", "false").lower() == "true"
GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
IMAGEN_MODEL = os.getenv("IMAGEN_MODEL", "gemini-3.1-flash-image")

# Pacing & Rendering Settings
PAUSE_BETWEEN_SCENES = float(os.getenv("PAUSE_BETWEEN_SCENES", "0.8"))
RENDER_MAX_WORKERS = int(os.getenv("RENDER_MAX_WORKERS", "20"))
USE_REFERENCE_CLIPS = os.getenv("USE_REFERENCE_CLIPS", "false").lower() == "true"
REFERENCE_CLIP_DURATION = int(os.getenv("REFERENCE_CLIP_DURATION", "60"))



# ---------------------------------------------------------------------------
# Visual Style Presets
# ---------------------------------------------------------------------------

STYLE_PRESETS = {
    "color_whiteboard": {
        "name": "Color Whiteboard",
        "image_prompt_prefix": (
            "Color whiteboard animation style, clean white background, "
            "colorful marker outlines in blue, red, green, and orange, "
            "hand-drawn educational sketch, minimalist, high contrast, "
            "2D illustration, no photorealism, bold outlines"
        ),
        "bg_color": "#FFFFFF",
        "text_color": "#1A1A1A",
        "subtitle_bg": "rgba(0,0,0,0.6)",
        "subtitle_color": "#FFFFFF",
        "subtitle_highlight": "#FFCC00",
        "transition": "fade",
        "transition_duration": 0.5,
    },
    "chalkboard": {
        "name": "Chalkboard",
        "image_prompt_prefix": (
            "Chalkboard animation style, dark green chalkboard background, "
            "white and colored chalk drawings, hand-drawn educational sketch, "
            "chalk dust texture, high contrast, 2D illustration, "
            "no photorealism, chalk-like strokes"
        ),
        "bg_color": "#2D4A2D",
        "text_color": "#F5F5F0",
        "subtitle_bg": "rgba(0,0,0,0.7)",
        "subtitle_color": "#F5F5F0",
        "subtitle_highlight": "#FFCC00",
        "transition": "fade",
        "transition_duration": 0.5,
    },
}


# ---------------------------------------------------------------------------
# Video Defaults
# ---------------------------------------------------------------------------

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
AUDIO_SAMPLE_RATE = 44100


# ---------------------------------------------------------------------------
# Project Structure
# ---------------------------------------------------------------------------

@dataclass
class ProjectPaths:
    """Manages all paths for a single video project."""

    base_dir: Path
    niche: str
    topic_slug: str

    @property
    def project_dir(self) -> Path:
        """Root folder for this video: base_dir / niche / topic_slug."""
        return self.base_dir / self.niche / self.topic_slug

    @property
    def research_file(self) -> Path:
        return self.project_dir / "research.md"

    @property
    def script_file(self) -> Path:
        return self.project_dir / "script.md"

    @property
    def scenes_dir(self) -> Path:
        return self.project_dir / "scenes"

    @property
    def broll_dir(self) -> Path:
        return self.project_dir / "broll"

    @property
    def audio_dir(self) -> Path:
        return self.project_dir / "audio"

    @property
    def voiceover_file(self) -> Path:
        return self.audio_dir / "voiceover.wav"

    @property
    def timestamps_file(self) -> Path:
        return self.audio_dir / "timestamps.json"

    @property
    def output_dir(self) -> Path:
        return self.project_dir / "output"

    @property
    def final_video(self) -> Path:
        return self.output_dir / f"{self.topic_slug}.mp4"

    @property
    def thumbnail_dir(self) -> Path:
        return self.project_dir / "thumbnails"

    @property
    def seo_file(self) -> Path:
        return self.project_dir / "seo.json"

    @property
    def reference_scripts_dir(self) -> Path:
        return self.base_dir / "reference_scripts"

    @property
    def reference_clips_dir(self) -> Path:
        return self.project_dir / "reference_clips"

    def ensure_dirs(self):
        """Create all necessary directories."""
        for d in [
            self.project_dir,
            self.scenes_dir,
            self.broll_dir,
            self.audio_dir,
            self.output_dir,
            self.thumbnail_dir,
            self.reference_scripts_dir,
            self.reference_clips_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Script Writing Prompts
# ---------------------------------------------------------------------------

RESEARCH_SYSTEM_PROMPT = """You are a world-class researcher preparing material for an educational YouTube video.

Given a topic, niche, and any additional context, produce a comprehensive research document that includes:

1. **Key Facts & Concepts** — The core information a viewer needs to understand
2. **Surprising Statistics** — Numbers that will hook viewers and make them share
3. **Narrative Arc Suggestions** — 2-3 possible story structures for the video
4. **Common Misconceptions** — What most people get wrong about this topic
5. **Expert Quotes / Studies** — Credible sources to reference
6. **Visual Scene Ideas** — Suggestions for illustrations that would explain each concept
7. **Hook Ideas** — 3-5 opening lines that would make someone click and stay

Format as clean markdown with headers. Be thorough but concise — this feeds a 8-20 minute video script."""


SCRIPT_SYSTEM_PROMPT = """You are a top-tier YouTube scriptwriter specializing in educational whiteboard animation videos (like After Skool, Kurzgesagt style narration).

Write a script for a {target_length} educational video. Follow these rules:

## Structure
- **Hook (0:00-0:30):** Start with a provocative question, surprising fact, or relatable scenario. Never start with generic AI transitions or cliches like "Picture this," "Imagine a world," "Think about it," "So," "Have you ever wondered," "Hey guys," or "In this video." Start directly with the hook.
- **Setup (0:30-2:00):** Frame the problem or question clearly. Why should the viewer care?
- **Body (2:00-{body_end}):** Break into 3-5 clear sections. Each section = one core idea.
- **Climax ({body_end}-{climax_end}):** The "aha moment" — the most surprising or satisfying insight.
- **Outro ({climax_end}-end):** Brief recap + thought-provoking closer. No begging for likes.

## Scene Markers
Insert `[SCENE: description]` markers throughout. Each scene = one whiteboard illustration.
- A scene should last 10-20 seconds of narration (approx. 25-50 words). Aim for a higher visual pacing/tempo to keep the viewer visually engaged.
- Describe what the illustration should show (characters, diagrams, metaphors)
- Aim for {scene_count} scenes total

## Writing Style
- Conversational, not academic. Write like you're explaining to a smart friend.
- Use analogies and metaphors to make abstract concepts concrete.
- Short sentences. Vary rhythm. Pause for emphasis.
- Build curiosity — plant questions early, answer them later.
- No filler. Every sentence earns its place.
- **TTS-Optimized Formatting**: The narration is read aloud by a neural TTS engine. Write with these specific guidelines to guide vocal delivery:
  - Use `[pause]` inline where there should be a short dramatic or conversational pause.
  - Use `[long pause]` for extended, suspenseful beats between major ideas.
  - Use exclamations (`!`) when introducing high-energy or surprising revelations to raise vocal excitement.
  - Vary the pacing by alternating short, punchy statements with detailed explanations.
  - Insert commas where the narrator should naturally breathe.
  - NEVER use markdown formatting like *bold*, **bold**, or _italic_ in narration text. These characters will be read aloud by the TTS engine.

## Format
Output the script as clean markdown. Each scene marker should be on its own line.
Include approximate timestamps as comments like `<!-- 2:30 -->`.

## Context
Topic: {topic}
Niche: {niche}
Additional direction: {additional_prompt}
Target length: {target_length}
Visual style: {style}"""


def slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    text = re.sub(r'-+', '-', text)
    return text[:80].strip('-')


def get_genai_client():
    """Initialize and return the genai.Client based on configuration (AI Studio or Vertex AI)."""
    from google import genai
    
    if USE_VERTEX:
        if not GCP_PROJECT:
            raise ValueError("GCP_PROJECT is required when USE_VERTEX=True")
        return genai.Client(
            vertexai=True,
            project=GCP_PROJECT,
            location=GCP_LOCATION,
        )
    else:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set.")
        return genai.Client(api_key=GOOGLE_API_KEY)
