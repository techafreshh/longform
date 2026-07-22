"""SEO metadata generation — titles, descriptions, tags for YouTube optimization."""

import json
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .config import OPENROUTER_API_KEY, DEFAULT_MODEL


SEO_SYSTEM_PROMPT = """You are a YouTube SEO expert specializing in educational content. 

Given a video script, research material, and metadata, generate optimized YouTube metadata that maximizes discoverability and click-through rate.

Return a JSON object (no markdown fences) with exactly these keys:
{
    "title": "Video title (50-70 chars, include power words, create curiosity gap)",
    "description": "Full description (first 150 chars are critical — hook the viewer). Include:\n- Brief summary (2-3 sentences)\n- Timestamps for key sections\n- Sources & References: Cite key research studies, papers, books, or experts mentioned in the research material and script\n- Relevant links/resources\n- Channel CTA\n- 3-5 relevant hashtags at the end",
    "tags": ["list", "of", "15-20", "relevant", "search", "terms"],
    "category_id": "27"
}

Category IDs: 22=People & Blogs, 27=Education, 28=Science & Tech, 24=Entertainment

Rules:
- STRICT NO EMOJIS: Do NOT include any emojis anywhere in the title, description, or tags. Use plain text only.
- Title: No clickbait that doesn't deliver. Use numbers, questions, or surprising claims.
- Description: Include a dedicated 'Sources & References' section detailing the research details and sources.
- Tags: Mix broad terms + long-tail. Include topic, niche, and related queries.
- Keep the tone professional but engaging."""


def _remove_emojis(text: str) -> str:
    """Remove emoji characters from text."""
    if not isinstance(text, str):
        return text
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF"
        "]+",
        flags=re.UNICODE,
    )
    cleaned = emoji_pattern.sub("", text)
    return re.sub(r"  +", " ", cleaned).strip()


def generate_seo(
    topic: str,
    niche: str,
    script_text: str,
    style: str,
    research_text: Optional[str] = None,
    output_path: Optional[Path] = None,
    model: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Generate YouTube SEO metadata from the video script and research.

    Args:
        topic: Video topic.
        niche: Content niche.
        script_text: The full script text.
        style: Visual style (for description context).
        research_text: Optional full research document text for citing sources.
        output_path: Where to save the SEO JSON.
        model: LLM model to use.
        verbose: Print progress.

    Returns:
        Dict with title, description, tags, and category_id.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    model = model or DEFAULT_MODEL

    if verbose:
        print("📊 Generating SEO metadata...")

    # Truncate script and research for context window efficiency
    script_excerpt = script_text[:3000] if len(script_text) > 3000 else script_text
    research_excerpt = ""
    if research_text:
        research_excerpt = f"\n\nResearch Material & Sources:\n{research_text[:2000]}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SEO_SYSTEM_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\nNiche: {niche}\nStyle: {style}\n\nScript:\n{script_excerpt}{research_excerpt}"},
        ],
        temperature=0.5,
        max_tokens=2000,
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON (handle potential markdown fences)
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]

    try:
        seo_data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: create basic metadata
        seo_data = {
            "title": topic,
            "description": f"Learn about {topic} in this educational video about {niche}.",
            "tags": [niche, topic.lower()],
            "category_id": "27",
        }

    # Post-process: strictly remove all emojis
    seo_data["title"] = _remove_emojis(seo_data.get("title", topic))
    seo_data["description"] = _remove_emojis(seo_data.get("description", ""))
    if "tags" in seo_data and isinstance(seo_data["tags"], list):
        seo_data["tags"] = [_remove_emojis(t) for t in seo_data["tags"] if t]

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(seo_data, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"✅ SEO metadata saved to {output_path}")

    if verbose:
        print(f"   Title: {seo_data.get('title', 'N/A')}")
        print(f"   Tags: {len(seo_data.get('tags', []))} tags")

    return seo_data
