"""Deep research module — uses Gemini Deep Research API for comprehensive topic research."""

import time
import json
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from .config import GOOGLE_API_KEY, RESEARCH_SYSTEM_PROMPT, get_genai_client, USE_VERTEX


def _get_client() -> genai.Client:
    """Initialize and return a Gemini client."""
    import os
    key = GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY", "")
    if not USE_VERTEX and not key:
        raise ValueError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return get_genai_client()


def research_topic(
    topic: str,
    niche: str,
    additional_prompt: str = "",
    output_path: Optional[Path] = None,
    use_deep_research: bool = True,
    verbose: bool = True,
) -> str:
    """
    Research a topic using Gemini.

    If use_deep_research is True, uses the Deep Research model for autonomous
    multi-step research. Otherwise falls back to standard Gemini for faster
    (but shallower) results.

    Args:
        topic: The video topic to research.
        niche: The content niche (e.g., "psychology", "history").
        additional_prompt: Extra direction from the user.
        output_path: Where to save the research document.
        use_deep_research: Whether to use the Deep Research agent.
        verbose: Whether to print progress updates.

    Returns:
        The research document as a markdown string.
    """
    client = _get_client()

    research_query = f"""Research this topic for an educational YouTube video:

Topic: {topic}
Niche: {niche}
{f'Additional context: {additional_prompt}' if additional_prompt else ''}

Provide comprehensive, well-sourced research suitable for an 8-20 minute whiteboard animation explainer video."""

    if use_deep_research:
        result = _deep_research(client, research_query, verbose)
    else:
        result = _standard_research(client, research_query)

    # Save to file if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result, encoding="utf-8")
        if verbose:
            print(f"✅ Research saved to {output_path}")

    return result


def _deep_research(client: genai.Client, query: str, verbose: bool) -> str:
    """Use Gemini Deep Research for autonomous multi-step research."""
    if verbose:
        print("🔬 Starting Gemini Deep Research (this may take 2-5 minutes)...")

    try:
        # Use Deep Research via the Interactions API
        config = types.GenerateContentConfig(
            system_instruction=RESEARCH_SYSTEM_PROMPT,
        )

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=query,
            config=config,
        )

        result = response.text
        if verbose:
            print(f"✅ Deep Research complete ({len(result)} chars)")
        return result

    except Exception as e:
        if verbose:
            print(f"⚠️  Deep Research failed: {e}")
            print("   Falling back to standard Gemini...")
        return _standard_research(client, query)


def _standard_research(client: genai.Client, query: str) -> str:
    """Fallback: use standard Gemini for quicker research."""
    config = types.GenerateContentConfig(
        system_instruction=RESEARCH_SYSTEM_PROMPT,
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=config,
    )

    return response.text
