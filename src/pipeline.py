"""End-to-end pipeline orchestrator — ties all modules together."""

import json
from pathlib import Path
from typing import Optional

from .config import ProjectPaths, STYLE_PRESETS, slugify
from .researcher import research_topic
from .scriptwriter import generate_script, parse_scenes
from .voice import generate_voice_fish, generate_voice_qwen
from .scene_gen import generate_scenes, generate_thumbnail
from .assembler import build_scene_timings, generate_subtitles, assemble_video
from .seo import generate_seo
from .stock import search_stock_footage


def run_pipeline(
    topic: str,
    niche: str,
    style: str = "color_whiteboard",
    additional_prompt: str = "",
    target_length: str = "10-12 min",
    base_dir: str = "/content/drive/MyDrive/LongformFactory",
    tts_engine: str = "fish",  # "fish", "qwen_1.7b", "qwen_0.6b"
    reference_audio: Optional[str] = None,
    bgm_path: Optional[str] = None,
    bgm_volume: float = 0.15,
    ken_burns: bool = True,
    skip_research: bool = False,
    skip_voice: bool = False,
    skip_scenes: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run the full video production pipeline.

    This is designed to be called from the Colab notebook cell-by-cell,
    but can also run end-to-end for fully automated production.

    Returns a dict with paths to all generated artifacts.
    """
    # Setup project paths
    paths = ProjectPaths(
        base_dir=Path(base_dir),
        niche=slugify(niche),
        topic_slug=slugify(topic),
    )
    paths.ensure_dirs()

    results = {"project_dir": str(paths.project_dir)}

    # --- Stage 1: Research ---
    if not skip_research:
        if verbose:
            print("\n" + "=" * 60)
            print("📚 STAGE 1: Deep Research")
            print("=" * 60)

        research = research_topic(
            topic=topic,
            niche=niche,
            additional_prompt=additional_prompt,
            output_path=paths.research_file,
            verbose=verbose,
        )
        results["research"] = str(paths.research_file)
    else:
        research = paths.research_file.read_text(encoding="utf-8")

    # --- Stage 2: Script ---
    if verbose:
        print("\n" + "=" * 60)
        print("📝 STAGE 2: Script Generation")
        print("=" * 60)

    script = generate_script(
        topic=topic,
        niche=niche,
        research=research,
        style=style,
        target_length=target_length,
        additional_prompt=additional_prompt,
        output_path=paths.script_file,
        verbose=verbose,
    )
    script.save_scenes_json(paths.project_dir / "scenes.json")
    results["script"] = str(paths.script_file)
    results["scene_count"] = script.scene_count

    # Return here for script review (in notebook, user reviews before continuing)
    results["_script_obj"] = script
    results["_paths"] = paths
    results["_research"] = research

    return results


def continue_after_script_review(
    results: dict,
    tts_engine: str = "fish",
    reference_audio: Optional[str] = None,
    bgm_path: Optional[str] = None,
    bgm_volume: float = 0.15,
    ken_burns: bool = True,
    verbose: bool = True,
) -> dict:
    """Continue the pipeline after script review approval."""

    script = results["_script_obj"]
    paths = results["_paths"]
    style = script.style
    scenes_data = [s.to_dict() for s in script.scenes]

    # --- Stage 3: Voice ---
    if verbose:
        print("\n" + "=" * 60)
        print("🎙️ STAGE 3: Voice Generation")
        print("=" * 60)

    if tts_engine == "fish":
        voice_result = generate_voice_fish(
            scenes=scenes_data,
            output_dir=paths.audio_dir,
            verbose=verbose,
        )
    elif tts_engine.startswith("qwen"):
        model_size = "0.6B" if "0.6" in tts_engine else "1.7B"
        voice_result = generate_voice_qwen(
            scenes=scenes_data,
            output_dir=paths.audio_dir,
            reference_audio=reference_audio,
            model_size=model_size,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown TTS engine: {tts_engine}")

    voice_result.save_timestamps(paths.timestamps_file)
    results["voiceover"] = str(voice_result.combined_audio)
    results["total_duration"] = voice_result.total_duration

    # --- Stage 4: Scene Images ---
    if verbose:
        print("\n" + "=" * 60)
        print("🎨 STAGE 4: Scene Image Generation")
        print("=" * 60)

    image_paths = generate_scenes(
        scenes=scenes_data,
        style=style,
        output_dir=paths.scenes_dir,
        verbose=verbose,
    )
    results["scenes"] = [str(p) for p in image_paths]

    # --- Stage 5: Assembly ---
    if verbose:
        print("\n" + "=" * 60)
        print("🎬 STAGE 5: Video Assembly")
        print("=" * 60)

    # Build timing map
    voice_segments = [
        {"index": s.index, "duration": s.duration}
        for s in voice_result.segments
    ]
    timings = build_scene_timings(scenes_data, voice_segments, paths.scenes_dir)

    # Generate subtitles
    srt_path = paths.output_dir / "subtitles.srt"
    generate_subtitles(timings, voice_result.timestamps, srt_path, style)

    # Assemble final video
    assemble_video(
        timings=timings,
        voiceover_path=voice_result.combined_audio,
        output_path=paths.final_video,
        style=style,
        subtitle_path=srt_path,
        bgm_path=Path(bgm_path) if bgm_path else None,
        bgm_volume=bgm_volume,
        ken_burns=ken_burns,
        verbose=verbose,
    )
    results["video"] = str(paths.final_video)

    # --- Stage 6: Thumbnails ---
    if verbose:
        print("\n" + "=" * 60)
        print("🖼️ STAGE 6: Thumbnail Generation")
        print("=" * 60)

    thumbnails = generate_thumbnail(
        topic=script.topic,
        niche=script.niche,
        style=style,
        output_dir=paths.thumbnail_dir,
        verbose=verbose,
    )
    results["thumbnails"] = [str(p) for p in thumbnails]

    # --- Stage 7: SEO ---
    if verbose:
        print("\n" + "=" * 60)
        print("📊 STAGE 7: SEO Metadata")
        print("=" * 60)

    seo = generate_seo(
        topic=script.topic,
        niche=script.niche,
        script_text=script.raw_text,
        style=style,
        output_path=paths.seo_file,
        verbose=verbose,
    )
    results["seo"] = seo

    if verbose:
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"  Video: {paths.final_video}")
        print(f"  Duration: {results['total_duration']:.1f}s")
        print(f"  Thumbnails: {len(thumbnails)} variants")
        print(f"  Title: {seo.get('title', 'N/A')}")

    return results
