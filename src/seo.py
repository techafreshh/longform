"""SEO metadata generation — titles, descriptions, tags for YouTube optimization."""

import json
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .config import OPENROUTER_API_KEY, DEFAULT_MODEL


SEO_SYSTEM_PROMPT = """You are a YouTube SEO expert specializing in educational content. 

Given a video script and metadata, generate optimized YouTube metadata that maximizes discoverability and click-through rate.

Return a JSON object (no markdown fences) with exactly these keys:
{
    "title": "Video title (50-70 chars, include power words, create curiosity gap)",
    "description": "Full description (first 150 chars are critical — hook the viewer). Include:\n- Brief summary (2-3 sentences)\n- Timestamps for key sections\n- Relevant links/resources\n- Channel CTA\n- 3-5 relevant hashtags at the end",
    "tags": ["list", "of", "15-20", "relevant", "search", "terms"],
    "category_id": "27"
}

Category IDs: 22=People & Blogs, 27=Education, 28=Science & Tech, 24=Entertainment

Rules:
- Title: No clickbait that doesn't deliver. Use numbers, questions, or surprising claims.
- Description: Front-load the most important info. YouTube only shows first ~100 chars in search.
- Tags: Mix broad terms + long-tail. Include topic, niche, and related queries.
- Keep the tone professional but engaging."""


def generate_seo(
    topic: str,
    niche: str,
    script_text: str,
    style: str,
    output_path: Optional[Path] = None,
    model: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Generate YouTube SEO metadata from the video script.

    Args:
        topic: Video topic.
        niche: Content niche.
        script_text: The full script text.
        style: Visual style (for description context).
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

    # Truncate script for context window efficiency
    script_excerpt = script_text[:3000] if len(script_text) > 3000 else script_text

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SEO_SYSTEM_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\nNiche: {niche}\nStyle: {style}\n\nScript:\n{script_excerpt}"},
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
