"""Script generation module — creates timestamped video scripts with scene markers."""

import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from .config import OPENROUTER_API_KEY, DEFAULT_MODEL, SCRIPT_SYSTEM_PROMPT


@dataclass
class Scene:
    """A single scene parsed from the script."""
    index: int
    description: str
    narration: str
    timestamp_hint: Optional[str] = None  # e.g., "2:30"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "description": self.description,
            "narration": self.narration,
            "timestamp_hint": self.timestamp_hint,
        }


@dataclass
class Script:
    """A complete video script with parsed scenes."""
    raw_text: str
    scenes: list[Scene]
    topic: str
    niche: str
    style: str

    @property
    def total_narration(self) -> str:
        """All narration text concatenated (for TTS)."""
        return "\n\n".join(s.narration for s in self.scenes)

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "niche": self.niche,
            "style": self.style,
            "scene_count": self.scene_count,
            "scenes": [s.to_dict() for s in self.scenes],
        }

    def save_scenes_json(self, path: Path):
        """Save parsed scenes to JSON for downstream pipeline."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def _get_client() -> OpenAI:
    """Get OpenRouter client."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set. Add it to your .env file.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


def _estimate_scene_count(target_length: str) -> int:
    """Estimate number of scenes from target length string."""
    # Parse "10-12 min" or "15 min" etc.
    numbers = re.findall(r'\d+', target_length)
    if numbers:
        avg_minutes = sum(int(n) for n in numbers) / len(numbers)
    else:
        avg_minutes = 10  # default

    # ~4 scenes per minute for whiteboard style (approx 15s per scene)
    return max(8, int(avg_minutes * 4.0))


def _estimate_timestamps(target_length: str) -> tuple[str, str]:
    """Estimate body_end and climax_end timestamps."""
    numbers = re.findall(r'\d+', target_length)
    if numbers:
        total_min = sum(int(n) for n in numbers) / len(numbers)
    else:
        total_min = 10

    body_end = f"{int(total_min * 0.75)}:00"
    climax_end = f"{int(total_min * 0.9)}:00"
    return body_end, climax_end


def generate_script(
    topic: str,
    niche: str,
    research: str,
    style: str = "color_whiteboard",
    target_length: str = "10-12 min",
    additional_prompt: str = "",
    model: Optional[str] = None,
    output_path: Optional[Path] = None,
    reference_scripts_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Script:
    """
    Generate a video script from research material.

    Args:
        topic: Video topic.
        niche: Content niche.
        research: The research document (from researcher.py).
        style: Visual style key (color_whiteboard or chalkboard).
        target_length: Target video length (e.g., "10-12 min").
        additional_prompt: Extra direction.
        model: LLM model to use (defaults to DEFAULT_MODEL).
        output_path: Where to save the raw script.
        reference_scripts_dir: Directory containing reference transcripts to emulate.
        verbose: Print progress.

    Returns:
        A Script object with parsed scenes.
    """
    client = _get_client()
    model = model or DEFAULT_MODEL

    scene_count = _estimate_scene_count(target_length)
    body_end, climax_end = _estimate_timestamps(target_length)

    system_prompt = SCRIPT_SYSTEM_PROMPT.format(
        target_length=target_length,
        body_end=body_end,
        climax_end=climax_end,
        scene_count=scene_count,
        topic=topic,
        niche=niche,
        additional_prompt=additional_prompt or "None",
        style=style,
    )

    # Load style reference scripts if directory exists
    reference_content = ""
    dirs_to_check = []
    if reference_scripts_dir:
        dirs_to_check.append(reference_scripts_dir)
    if output_path:
        dirs_to_check.append(output_path.parent / "reference_scripts")

    ref_parts = []
    seen_files = set()
    for d in dirs_to_check:
        if d and d.exists():
            ref_files = list(d.glob("*.txt")) + list(d.glob("*.md"))
            for rf in ref_files:
                if rf.name in seen_files:
                    continue
                try:
                    content = rf.read_text(encoding="utf-8").strip()
                    if content:
                        ref_parts.append(f"### Reference: {rf.name}\n{content}")
                        seen_files.add(rf.name)
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Failed to read reference script {rf.name}: {e}")
        if ref_parts:
            reference_content = "\n\n".join(ref_parts)
            if verbose:
                print(f"📚 Loaded {len(ref_parts)} reference script(s) for style emulation.")

    user_message = f"""Here is the research material for this video:

---
{research}
---
"""

    if reference_content:
        user_message += f"""

Here are reference transcripts of successful high-performing videos. Emulate their vocabulary, hook styling, transitions, conversational rhythm, and pacing:

---
{reference_content}
---
"""

    user_message += f"""

Now write the complete script with [SCENE: description] markers. 
Remember: {scene_count} scenes, {target_length} target length, {style} visual style."""

    if verbose:
        print(f"📝 Generating script with {model} (~{scene_count} scenes)...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=8000,
    )

    raw_script = response.choices[0].message.content

    # Save raw script
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(raw_script, encoding="utf-8")
        if verbose:
            print(f"✅ Script saved to {output_path}")

    # Parse into scenes
    scenes = parse_scenes(raw_script)
    script = Script(
        raw_text=raw_script,
        scenes=scenes,
        topic=topic,
        niche=niche,
        style=style,
    )

    if verbose:
        print(f"✅ Parsed {script.scene_count} scenes from script")

    return script


def parse_scenes(script_text: str) -> list[Scene]:
    """
    Parse [SCENE: description] markers and their following narration from a script.

    Returns a list of Scene objects with index, description, and narration text.
    """
    # Find all scene markers and their positions
    scene_pattern = r'\[SCENE:\s*(.+?)\]'
    markers = list(re.finditer(scene_pattern, script_text, re.IGNORECASE))

    if not markers:
        # Fallback: treat the entire script as one scene
        return [Scene(
            index=1,
            description="Full video",
            narration=_clean_narration(script_text),
        )]

    scenes = []
    for i, match in enumerate(markers):
        description = match.group(1).strip()

        # Get the text between this marker and the next (or end of script)
        start = match.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(script_text)
        narration = script_text[start:end].strip()

        # Look for timestamp hints (<!-- 2:30 -->)
        timestamp_match = re.search(r'<!--\s*(\d+:\d+)\s*-->', narration)
        timestamp_hint = timestamp_match.group(1) if timestamp_match else None

        # Clean the narration
        narration = _clean_narration(narration)

        if narration:  # Skip empty scenes
            scenes.append(Scene(
                index=i + 1,
                description=description,
                narration=narration,
                timestamp_hint=timestamp_hint,
            ))

    return scenes


def _clean_narration(text: str) -> str:
    """Remove markers, comments, and formatting from narration text."""
    # Remove scene markers
    text = re.sub(r'\[SCENE:.*?\]', '', text, flags=re.IGNORECASE)
    # Remove HTML comments (timestamps)
    text = re.sub(r'<!--.*?-->', '', text)
    # Remove markdown headers (## etc)
    text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
