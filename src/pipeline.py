"""End-to-end pipeline orchestrator — ties all modules together."""

import json
from pathlib import Path
from typing import Optional

from .config import ProjectPaths, STYLE_PRESETS, slugify, USE_REFERENCE_CLIPS, REFERENCE_CLIP_DURATION
from .researcher import research_topic
from .scriptwriter import generate_script, parse_scenes, Script
from .voice import generate_voice_fish, generate_voice_qwen, VoiceResult, VoiceSegment
from .scene_gen import generate_scenes, generate_thumbnail
from .assembler import build_scene_timings, generate_subtitles, assemble_video
from .seo import generate_seo
from .stock import search_stock_footage
from .youtube_reference import search_youtube_videos, download_youtube_transcript
from .youtube_clip import download_reference_clip_optional, analyze_reference_clip_with_transcript



def _upload_if_gdrive_active(gdrive_folder_id: Optional[str], file_path: Path, subfolder: Optional[str] = None):
    """Utility to upload a file to Google Drive if folder ID is provided, without raising exceptions."""
    if not gdrive_folder_id:
        return
    try:
        from .gdrive import get_gdrive_service, upload_file_to_drive_folder
        service = get_gdrive_service()
        if service:
            upload_file_to_drive_folder(service, gdrive_folder_id, file_path, subfolder)
    except Exception as e:
        print(f"⚠️ Failed to upload {file_path.name} to Google Drive: {e}")


def run_stage_research(
    paths: ProjectPaths,
    topic: str,
    niche: str,
    additional_prompt: str = "",
    force: bool = False,
    gdrive_folder_id: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Run research stage, skipping if research file already exists."""
    if paths.research_file.exists() and not force:
        if verbose:
            print(f"ℹ️ Research file already exists at: {paths.research_file}")
            print("   Skipping research stage to save credits. Set force=True to regenerate.")
        return paths.research_file.read_text(encoding="utf-8")

    res_text = research_topic(
        topic=topic,
        niche=niche,
        additional_prompt=additional_prompt,
        output_path=paths.research_file,
        verbose=verbose,
    )
    _upload_if_gdrive_active(gdrive_folder_id, paths.research_file)
    return res_text


def run_stage_script(
    paths: ProjectPaths,
    topic: str,
    niche: str,
    research_text: str,
    style: str = "color_whiteboard",
    target_length: str = "10-12 min",
    additional_prompt: str = "",
    model: Optional[str] = None,
    force: bool = False,
    gdrive_folder_id: Optional[str] = None,
    resume_partial: bool = False,
    verbose: bool = True,
) -> Script:
    """Run script generation stage, skipping if script and scenes JSON already exist."""
    scenes_json_path = paths.project_dir / "scenes.json"
    
    # Auto-resume if script file exists but scenes.json was never completed successfully
    if not force and paths.script_file.exists() and not scenes_json_path.exists():
        if verbose:
            print("💡 Script file exists but scenes.json metadata is missing (aborted/incomplete run). Auto-enabling RESUME_PARTIAL fallback.")
        resume_partial = True

    if paths.script_file.exists() and scenes_json_path.exists() and not force and not resume_partial:
        if verbose:
            print(f"ℹ️ Script and scenes list already exist at: {paths.script_file}")
            print("   Skipping scriptwriting stage to save credits. Set force=True to regenerate.")
        raw_text = paths.script_file.read_text(encoding="utf-8")
        scenes = parse_scenes(raw_text)
        return Script(
            raw_text=raw_text,
            scenes=scenes,
            topic=topic,
            niche=niche,
            style=style,
        )

    # Inject reference style guidelines if they exist
    analysis_file = paths.project_dir / "reference_style_analysis.json"
    if analysis_file.exists():
        try:
            with open(analysis_file, "r", encoding="utf-8") as f:
                style_analysis = json.load(f)
                style_directions = f"""
Visual and Pacing style instructions (emulate this in scene transitions, layouts, and pauses):
- Background style: {style_analysis.get('background_style', 'N/A')}
- Drawing style: {style_analysis.get('drawing_style', 'N/A')}
- Visual metaphors: {style_analysis.get('visual_metaphors', 'N/A')}
- Narrative Pacing: {style_analysis.get('narrative_pacing', 'N/A')}
"""
                additional_prompt = (additional_prompt + "\n" + style_directions).strip()
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to load style guidelines for script generation: {e}")

    script = generate_script(
        topic=topic,
        niche=niche,
        research=research_text,
        style=style,
        target_length=target_length,
        additional_prompt=additional_prompt,
        model=model,
        output_path=paths.script_file,
        reference_scripts_dir=paths.reference_scripts_dir,
        resume_partial=resume_partial,
        verbose=verbose,
    )
    script.save_scenes_json(scenes_json_path)
    
    _upload_if_gdrive_active(gdrive_folder_id, paths.script_file)
    _upload_if_gdrive_active(gdrive_folder_id, scenes_json_path)
    return script


def run_stage_voice(
    paths: ProjectPaths,
    scenes_data: list[dict],
    tts_engine: str = "fish",
    voice_model: str = "s2.1-pro-free",
    reference_audio: Optional[str] = None,
    force: bool = False,
    gdrive_folder_id: Optional[str] = None,
    verbose: bool = True,
) -> VoiceResult:
    """Run voice generation stage, skipping if voiceover and timestamps already exist."""
    if paths.voiceover_file.exists() and paths.timestamps_file.exists() and not force:
        if verbose:
            print(f"ℹ️ Voiceover audio already exists at: {paths.voiceover_file}")
            print("   Skipping voice generation stage to save credits. Set force=True to regenerate.")
        with open(paths.timestamps_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        segments = [
            VoiceSegment(
                index=s["index"],
                text="",
                audio_path=Path(s["audio_path"]),
                duration=s["duration"]
            )
            for s in data["segments"]
        ]
        return VoiceResult(
            segments=segments,
            combined_audio=paths.voiceover_file,
            total_duration=data["total_duration"],
            timestamps=data.get("word_timestamps", [])
        )

    if tts_engine == "fish":
        voice_result = generate_voice_fish(
            scenes=scenes_data,
            output_dir=paths.audio_dir,
            model=voice_model,
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
    
    # Upload voice files immediately
    _upload_if_gdrive_active(gdrive_folder_id, voice_result.combined_audio)
    _upload_if_gdrive_active(gdrive_folder_id, paths.timestamps_file)
    for seg in voice_result.segments:
        _upload_if_gdrive_active(gdrive_folder_id, seg.audio_path, "voice")

    return voice_result


def run_stage_scenes(
    paths: ProjectPaths,
    scenes_data: list[dict],
    style: str,
    force: bool = False,
    force_scenes: Optional[list[int]] = None,
    gdrive_folder_id: Optional[str] = None,
    resume_from_scene: Optional[int] = None,
    verbose: bool = True,
) -> list[Path]:
    """Run scene image generation stage, skipping if all expected scene images exist."""
    expected_paths = []
    for scene in scenes_data:
        idx = scene["index"]
        img_path = paths.scenes_dir / f"scene_{idx:02d}.png"
        expected_paths.append(img_path)
    
    # If not forcing everything and no specific scenes are forced, check if all exist
    if not force and not force_scenes:
        all_exist = True
        for scene in scenes_data:
            idx = scene["index"]
            if resume_from_scene is not None and idx < resume_from_scene:
                continue
            img_path = paths.scenes_dir / f"scene_{idx:02d}.png"
            if not img_path.exists():
                all_exist = False

        if all_exist:
            if verbose:
                print(f"ℹ️ All expected scene images already exist in: {paths.scenes_dir}")
                print("   Skipping image generation to save credits. Set force=True to regenerate.")
            return expected_paths

    return generate_scenes(
        scenes=scenes_data,
        style=style,
        output_dir=paths.scenes_dir,
        force=force,
        force_scenes=force_scenes,
        gdrive_folder_id=gdrive_folder_id,
        resume_from_scene=resume_from_scene,
        verbose=verbose,
    )


def run_stage_assembly(
    paths: ProjectPaths,
    scenes_data: list[dict],
    voice_result: VoiceResult,
    style: str = "color_whiteboard",
    bgm_path: Optional[str] = None,
    bgm_volume: float = 0.15,
    ken_burns: bool = True,
    force: bool = False,
    gdrive_folder_id: Optional[str] = None,
    resume_from_scene: Optional[int] = None,
    max_workers: Optional[int] = None,
    skip_subtitles: bool = False,
    verbose: bool = True,
) -> Path:
    """Run video assembly stage, skipping if final video already exists."""
    if style == "stickman":
        ken_burns = False

    if paths.final_video.exists() and not force:
        if verbose:
            print(f"ℹ️ Final video already exists at: {paths.final_video}")
            print("   Skipping video assembly stage. Set force=True to re-assemble.")
        return paths.final_video

    # Build timing map
    voice_segments = [
        {"index": s.index, "duration": s.duration}
        for s in voice_result.segments
    ]
    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["color_whiteboard"])
    transition_type = preset.get("transition", "fade")
    transition_duration = preset.get("transition_duration", 0.5)

    min_scene_duration = 1.0 if style == "stickman" else 3.0
    timings = build_scene_timings(
        scenes_data,
        voice_segments,
        paths.scenes_dir,
        transition_type=transition_type,
        transition_duration=transition_duration,
        min_scene_duration=min_scene_duration,
    )

    srt_path = None
    if not skip_subtitles:
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
        transition_type=transition_type,
        transition_duration=transition_duration,
        gdrive_folder_id=gdrive_folder_id,
        resume_from_scene=resume_from_scene,
        max_workers=max_workers,
        verbose=verbose,
    )
    return paths.final_video


def run_stage_thumbnails(
    paths: ProjectPaths,
    topic: str,
    niche: str,
    style: str,
    force: bool = False,
    gdrive_folder_id: Optional[str] = None,
    verbose: bool = True,
) -> list[Path]:
    """Run thumbnail generation stage, skipping if thumbnails already exist."""
    topic_slug = slugify(topic)
    existing_thumbs = list(paths.thumbnail_dir.glob(f"{topic_slug}_thumbnail_*.png"))
    if existing_thumbs and not force:
        if verbose:
            print(f"ℹ️ Thumbnail variants already exist in: {paths.thumbnail_dir}")
            print("   Skipping thumbnail generation to save credits. Set force=True to regenerate.")
        return existing_thumbs

    thumbs = generate_thumbnail(
        topic=topic,
        niche=niche,
        style=style,
        output_dir=paths.thumbnail_dir,
        verbose=verbose,
    )
    
    # Upload thumbnails immediately
    for tp in thumbs:
        _upload_if_gdrive_active(gdrive_folder_id, tp, "thumbnails")
        
    return thumbs


def run_stage_seo(
    paths: ProjectPaths,
    topic: str,
    niche: str,
    script_text: str,
    style: str,
    force: bool = False,
    gdrive_folder_id: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Run SEO generation stage, skipping if SEO file already exists."""
    if paths.seo_file.exists() and not force:
        if verbose:
            print(f"ℹ️ SEO metadata file already exists at: {paths.seo_file}")
            print("   Skipping SEO generation to save credits. Set force=True to regenerate.")
        with open(paths.seo_file, "r", encoding="utf-8") as f:
            return json.load(f)

    seo = generate_seo(
        topic=topic,
        niche=niche,
        script_text=script_text,
        style=style,
        output_path=paths.seo_file,
        verbose=verbose,
    )
    _upload_if_gdrive_active(gdrive_folder_id, paths.seo_file)
    return seo


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
    skip_reference_fetch: bool = False,
    reference_count: int = 3,
    model: Optional[str] = None,
    force: bool = False,
    force_scenes: Optional[list[int]] = None,
    gdrive_folder_id: Optional[str] = None,
    resume_from_scene: Optional[int] = None,
    resume_partial: bool = False,
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
    if verbose:
        print("\n" + "=" * 60)
        print("📚 STAGE 1: Deep Research")
        print("=" * 60)

    if skip_research:
        research = paths.research_file.read_text(encoding="utf-8")
        results["research"] = str(paths.research_file)
    else:
        research = run_stage_research(
            paths=paths,
            topic=topic,
            niche=niche,
            additional_prompt=additional_prompt,
            force=force,
            gdrive_folder_id=gdrive_folder_id,
            verbose=verbose,
        )
        results["research"] = str(paths.research_file)

    # --- Stage 1.5: Reference Scripts ---
    if not skip_reference_fetch:
        if verbose:
            print("\n" + "=" * 60)
            print("🎥 STAGE 1.5: Fetching Reference Scripts from YouTube")
            print("=" * 60)
        run_stage_reference_scripts(
            paths=paths,
            topic=topic,
            count=reference_count,
            force=force,
            gdrive_folder_id=gdrive_folder_id,
            verbose=verbose,
        )

    # --- Stage 1.7: Reference Clips ---
    if USE_REFERENCE_CLIPS:
        if verbose:
            print("\n" + "=" * 60)
            print("📹 STAGE 1.7: Downloading & Analyzing Reference Clips")
            print("=" * 60)
        run_stage_reference_clips(
            paths=paths,
            topic=topic,
            force=force,
            gdrive_folder_id=gdrive_folder_id,
            verbose=verbose,
        )

    # --- Stage 2: Script ---
    if verbose:
        print("\n" + "=" * 60)
        print("📝 STAGE 2: Script Generation")
        print("=" * 60)

    script = run_stage_script(
        paths=paths,
        topic=topic,
        niche=niche,
        research_text=research,
        style=style,
        target_length=target_length,
        additional_prompt=additional_prompt,
        model=model,
        force=force,
        gdrive_folder_id=gdrive_folder_id,
        resume_partial=resume_partial,
        verbose=verbose,
    )
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
    voice_model: str = "s2.1-pro-free",
    reference_audio: Optional[str] = None,
    bgm_path: Optional[str] = None,
    bgm_volume: float = 0.15,
    ken_burns: bool = True,
    force: bool = False,
    force_scenes: Optional[list[int]] = None,
    gdrive_folder_id: Optional[str] = None,
    resume_from_scene: Optional[int] = None,
    max_workers: Optional[int] = None,
    skip_subtitles: bool = False,
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

    voice_result = run_stage_voice(
        paths=paths,
        scenes_data=scenes_data,
        tts_engine=tts_engine,
        voice_model=voice_model,
        reference_audio=reference_audio,
        force=force,
        gdrive_folder_id=gdrive_folder_id,
        verbose=verbose,
    )
    results["voiceover"] = str(voice_result.combined_audio)
    results["total_duration"] = voice_result.total_duration

    # --- Stage 4: Scene Images ---
    if verbose:
        print("\n" + "=" * 60)
        print("🎨 STAGE 4: Scene Image Generation")
        print("=" * 60)

    image_paths = run_stage_scenes(
        paths=paths,
        scenes_data=scenes_data,
        style=style,
        force=force,
        force_scenes=force_scenes,
        gdrive_folder_id=gdrive_folder_id,
        resume_from_scene=resume_from_scene,
        verbose=verbose,
    )
    results["scenes"] = [str(p) for p in image_paths]

    # --- Stage 5: Assembly ---
    if verbose:
        print("\n" + "=" * 60)
        print("🎬 STAGE 5: Video Assembly")
        print("=" * 60)

    video_path = run_stage_assembly(
        paths=paths,
        scenes_data=scenes_data,
        voice_result=voice_result,
        style=style,
        bgm_path=bgm_path,
        bgm_volume=bgm_volume,
        ken_burns=ken_burns,
        force=force,
        gdrive_folder_id=gdrive_folder_id,
        resume_from_scene=resume_from_scene,
        max_workers=max_workers,
        skip_subtitles=skip_subtitles,
        verbose=verbose,
    )
    results["video"] = str(video_path)

    # --- Stage 6: Thumbnails ---
    if verbose:
        print("\n" + "=" * 60)
        print("🖼️ STAGE 6: Thumbnail Generation")
        print("=" * 60)

    thumbnails = run_stage_thumbnails(
        paths=paths,
        topic=script.topic,
        niche=script.niche,
        style=style,
        force=force,
        gdrive_folder_id=gdrive_folder_id,
        verbose=verbose,
    )
    results["thumbnails"] = [str(p) for p in thumbnails]

    # --- Stage 7: SEO ---
    if verbose:
        print("\n" + "=" * 60)
        print("📊 STAGE 7: SEO Metadata")
        print("=" * 60)

    seo = run_stage_seo(
        paths=paths,
        topic=script.topic,
        niche=script.niche,
        script_text=script.raw_text,
        style=style,
        force=force,
        gdrive_folder_id=gdrive_folder_id,
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


def run_stage_reference_scripts(
    paths: ProjectPaths,
    topic: str,
    count: int = 3,
    force: bool = False,
    gdrive_folder_id: Optional[str] = None,
    verbose: bool = True,
) -> list[Path]:
    """Search and download YouTube transcripts for reference scripts in project_dir."""
    project_ref_dir = paths.project_dir / "reference_scripts"

    if project_ref_dir.exists() and not force:
        existing = list(project_ref_dir.glob("*.txt"))
        if existing:
            if verbose:
                print(f"ℹ️ Reference transcripts already exist in: {project_ref_dir}")
                print("   Skipping download to save bandwidth. Set force=True to download new ones.")
            return existing

    project_ref_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"🔍 Searching YouTube for references related to: '{topic}'...")

    videos = search_youtube_videos(topic, count=count)
    if not videos:
        if verbose:
            print("⚠️ No reference videos found on YouTube.")
        return []

    downloaded = []
    for v in videos:
        v_id = v["video_id"]
        title = v["title"]
        file_path = download_youtube_transcript(v_id, title, project_ref_dir, verbose=verbose)
        if file_path:
            downloaded.append(file_path)
            # Upload immediately if gdrive is configured
            _upload_if_gdrive_active(gdrive_folder_id, file_path, "reference_scripts")

    return downloaded


def run_stage_reference_clips(
    paths: ProjectPaths,
    topic: str,
    force: bool = False,
    gdrive_folder_id: Optional[str] = None,
    verbose: bool = True,
) -> Optional[dict]:
    """Search, download (optional), and analyze reference video clips from YouTube."""
    if not USE_REFERENCE_CLIPS:
        return None

    analysis_file = paths.project_dir / "reference_style_analysis.json"
    if analysis_file.exists() and not force:
        if verbose:
            print(f"ℹ️ Reference style analysis already exists at: {analysis_file}")
            print("   Skipping style analysis. Set force=True to regenerate.")
        try:
            with open(analysis_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # 1. Scan reference_clips directory for existing files
    clip_path = None
    existing_clips = []
    if paths.reference_clips_dir.exists():
        existing_clips = list(paths.reference_clips_dir.glob("*.mp4")) + \
                         list(paths.reference_clips_dir.glob("*.webm")) + \
                         list(paths.reference_clips_dir.glob("*.mkv"))

    if existing_clips:
        clip_path = existing_clips[0]
        if verbose:
            print(f"📁 Found existing local reference clip: {clip_path.name}")
    else:
        # Check if we have dynamic yt-dlp downloading capability
        if verbose:
            print(f"🔍 Searching YouTube for references related to: '{topic}' to download a clip...")
        from .youtube_reference import search_youtube_videos
        videos = search_youtube_videos(topic, count=1)
        if not videos:
            if verbose:
                print("⚠️ No reference videos found to download.")
            return None
        
        video_id = videos[0]["video_id"]
        title = videos[0]["title"]
        safe_title = slugify(title)[:50]
        clip_path = paths.reference_clips_dir / f"{safe_title}_{video_id}.mp4"
        
        # Pull reference clip (checks if yt-dlp is available)
        download_reference_clip_optional(video_id, clip_path, duration_seconds=REFERENCE_CLIP_DURATION, verbose=verbose)

    if not clip_path or not clip_path.exists() or clip_path.stat().st_size < 1024:
        if verbose:
            print("⚠️ No valid reference clip available for analysis.")
        return None

    # 2. Get transcript text for this reference
    transcript_text = ""
    video_id = None
    stem = clip_path.stem
    if "_" in stem:
        video_id = stem.split("_")[-1]
    
    project_ref_dir = paths.project_dir / "reference_scripts"
    existing_transcripts = list(project_ref_dir.glob(f"*{video_id}*.txt")) if (video_id and project_ref_dir.exists()) else []
    if existing_transcripts:
        try:
            transcript_text = existing_transcripts[0].read_text(encoding="utf-8")
        except Exception:
            pass
    
    if not transcript_text and video_id:
        from .youtube_reference import download_youtube_transcript
        ref_title = stem.rsplit("_", 1)[0]
        tx_path = download_youtube_transcript(video_id, ref_title, project_ref_dir, verbose=verbose)
        if tx_path and tx_path.exists():
            try:
                transcript_text = tx_path.read_text(encoding="utf-8")
            except Exception:
                pass

    if not transcript_text:
        if verbose:
            print("⚠️ No transcript text available for this reference. Performing video-only analysis.")
        transcript_text = "[No transcript available. Analyze visual pacing and editing from the video frames alone.]"

    # 3. Perform Gemini multimodal analysis
    from .config import get_genai_client
    try:
        client = get_genai_client()
        analysis = analyze_reference_clip_with_transcript(client, clip_path, transcript_text, verbose=verbose)
        if analysis:
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            _upload_if_gdrive_active(gdrive_folder_id, analysis_file)
            return analysis
    except Exception as e:
        if verbose:
            print(f"⚠️ Failed to analyze reference clip: {e}")
    
    return None

