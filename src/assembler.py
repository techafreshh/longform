"""Video assembly module — FFmpeg-based slideshow with transitions, Ken Burns, and subtitles."""

import json
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from .config import VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS, STYLE_PRESETS, RENDER_MAX_WORKERS, SUBTITLE_DELAY


@dataclass
class SceneTiming:
    """Timing info for a single scene in the video."""
    index: int
    image_path: Path
    start_time: float  # seconds
    duration: float    # seconds
    narration: str
    voice_duration: float = 0.0



def build_scene_timings(
    scenes: list[dict],
    voice_segments: list[dict],
    image_dir: Path,
    transition_type: str = "fade",
    transition_duration: float = 0.5,
    min_scene_duration: float = 3.0,
) -> list[SceneTiming]:
    """
    Build timing map: which image shows at which time, driven by voice durations.

    Args:
        scenes: Parsed scene list with 'index', 'description', 'narration'.
        voice_segments: Voice segment list with 'index', 'duration'.
        image_dir: Directory containing scene_XX.png images.
        transition_type: Transition type (e.g. 'fade').
        transition_duration: Transition duration in seconds.
        min_scene_duration: Minimum duration for each scene.

    Returns:
        List of SceneTiming objects in order.
    """
    # Build a duration map from voice segments
    duration_map = {}
    for seg in voice_segments:
        duration_map[seg["index"]] = seg["duration"]

    timings = []
    current_time = 0.0

    for scene in scenes:
        idx = scene["index"]
        # Use voice duration if available, otherwise estimate from word count
        v_dur = duration_map.get(idx, len(scene.get("narration", "").split()) * 0.4)

        # Minimum duration constraint
        v_dur = max(min_scene_duration, v_dur)

        image_path = image_dir / f"scene_{idx:02d}.png"
        if not image_path.exists():
            print(f"⚠️ Missing image for scene {idx}: {image_path}")
            continue

        timings.append(SceneTiming(
            index=idx,
            image_path=image_path,
            start_time=current_time,
            duration=v_dur,
            narration=scene.get("narration", ""),
            voice_duration=v_dur,
        ))

        current_time += v_dur

    # Adjust durations to account for transitions (only for non-last scenes)
    if transition_type != "none" and transition_duration > 0.0 and len(timings) > 1:
        for i in range(len(timings) - 1):
            timings[i].duration += transition_duration

    return timings



def generate_subtitles(
    timings: list[SceneTiming],
    word_timestamps: list[dict],
    output_path: Path,
    style: str = "color_whiteboard",
    subtitle_delay: float = SUBTITLE_DELAY,
) -> Path:
    """
    Generate SRT subtitle file from scene timings and word timestamps.

    If word-level timestamps are available (from Whisper), uses those
    for more precise subtitle timing. Otherwise, creates sentence-level
    subtitles from scene narrations.

    Args:
        timings: Scene timing list.
        word_timestamps: Word-level timestamps from Whisper.
        output_path: Where to save the .srt file.
        style: Visual style (affects subtitle formatting).
        subtitle_delay: Seconds to delay subtitles before displaying.

    Returns:
        Path to the generated SRT file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if word_timestamps:
        preset = STYLE_PRESETS.get(style, STYLE_PRESETS["color_whiteboard"])
        highlight_color = preset.get("subtitle_highlight", "#FFCC00")
        srt_content = _word_level_srt(word_timestamps, highlight_color=highlight_color, start_time_delay=subtitle_delay)
    else:
        srt_content = _scene_level_srt(timings, start_time_delay=subtitle_delay)

    output_path.write_text(srt_content, encoding="utf-8")
    return output_path


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _word_level_srt(
    timestamps: list[dict],
    words_per_group: int = 6,
    highlight_color: str = "#FFCC00",
    start_time_delay: float = 0.0,
) -> str:
    """Create SRT from word-level timestamps, grouping words and highlighting the active word.

    Each cue extends to the next word's start time so the subtitle stays on
    screen continuously — no flashing between words or groups.
    """
    import re

    srt_entries = []
    idx = 1
    OVERLAP_BUFFER = 0.05  # 50ms overlap to prevent inter-group flash

    for i in range(0, len(timestamps), words_per_group):
        group = timestamps[i:i + words_per_group]
        if not group:
            continue

        # Check if there's a next group
        next_group_start = (timestamps[i + words_per_group]["start"]
                           if (i + words_per_group) < len(timestamps) else None)

        for word_index, active_word_info in enumerate(group):
            start = active_word_info["start"]
            if start < start_time_delay:
                continue

            # Extend to next word's start (no gap) or group boundary
            if word_index < len(group) - 1:
                end = group[word_index + 1]["start"]
            elif next_group_start is not None:
                # Last word in group: extend slightly past to overlap with next group
                end = max(active_word_info["end"], next_group_start - OVERLAP_BUFFER)
            else:
                end = active_word_info["end"]

            # Format words, wrapping the active word in color tags
            words_formatted = []
            for j, w in enumerate(group):
                # Strip any residual markdown characters (*, _)
                word_text = re.sub(r'[*_]', '', w["word"]).upper()
                if j == word_index:
                    words_formatted.append(f'<font color="{highlight_color}">{word_text}</font>')
                else:
                    words_formatted.append(word_text)

            text = " ".join(words_formatted)

            srt_entries.append(
                f"{idx}\n"
                f"{_format_srt_time(start)} --> {_format_srt_time(end)}\n"
                f"{text}\n"
            )
            idx += 1

    return "\n".join(srt_entries)


def _scene_level_srt(timings: list[SceneTiming], start_time_delay: float = 0.0) -> str:
    """Create SRT from scene-level timings (fallback when no word timestamps)."""
    import re

    srt_entries = []
    idx = 1

    for timing in timings:
        # Use voice_duration if available and valid (> 0), fallback to duration
        dur = timing.voice_duration if getattr(timing, "voice_duration", 0.0) > 0.0 else timing.duration

        # Split narration into ~10-word chunks
        words = timing.narration.split()
        chunk_size = 10
        chunks = [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

        if not chunks:
            continue

        chunk_duration = dur / len(chunks)

        for j, chunk in enumerate(chunks):
            start = timing.start_time + j * chunk_duration
            if start < start_time_delay:
                continue
            end = start + chunk_duration

            srt_entries.append(
                f"{idx}\n"
                f"{_format_srt_time(start)} --> {_format_srt_time(end)}\n"
                f"{chunk.upper()}\n"
            )
            idx += 1

    return "\n".join(srt_entries)


def _detect_h264_encoder() -> str:
    """Detect the best available H.264 encoder (h264_nvenc for GPU or libx264 for CPU)."""
    try:
        res = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, check=True)
        if "h264_nvenc" in res.stdout:
            return "h264_nvenc"
    except Exception:
        pass
    return "libx264"


def assemble_video(
    timings: list[SceneTiming],
    voiceover_path: Path,
    output_path: Path,
    style: str = "color_whiteboard",
    subtitle_path: Optional[Path] = None,
    bgm_path: Optional[Path] = None,
    bgm_volume: float = 0.15,
    ken_burns: bool = True,
    transition_type: str = "fade",
    transition_duration: float = 0.5,
    gdrive_folder_id: Optional[str] = None,
    resume_from_scene: Optional[int] = None,
    max_workers: Optional[int] = None,
    verbose: bool = True,
) -> Path:
    """
    Assemble the final video from images + audio using FFmpeg.

    This is the core assembly function. It:
    1. Creates a slideshow from timestamped images
    2. Applies Ken Burns (zoom/pan) effect per scene
    3. Adds crossfade transitions between scenes
    4. Mixes voiceover + optional background music
    5. Burns in subtitles

    Args:
        timings: Scene timings (images + durations).
        voiceover_path: Path to the voiceover WAV.
        output_path: Where to save the final MP4.
        style: Visual style key.
        subtitle_path: Path to SRT file for subtitles.
        bgm_path: Optional background music file.
        bgm_volume: Background music volume (0-1).
        ken_burns: Apply Ken Burns zoom/pan effect.
        transition_type: Type of transition between scenes.
        transition_duration: Duration of transitions in seconds.
        gdrive_folder_id: Optional Google Drive folder ID to upload clips/video immediately.
        verbose: Print progress.

    Returns:
        Path to the final video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["color_whiteboard"])

    if verbose:
        total_dur = sum(t.duration for t in timings)
        print(f"🎬 Assembling video: {len(timings)} scenes, {total_dur:.1f}s total")

    # Strategy: render each scene as a short clip, then concatenate with transitions
    # Use a persistent cache directory to allow resuming video assembly
    clips_cache_dir = output_path.parent.parent / "clips_cache"
    clips_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect available encoder
    encoder = _detect_h264_encoder()
    if verbose:
        print(f"⚙️ Video encoder: {encoder}")

    try:
        # Step 1: Generate per-scene clips with Ken Burns
        if verbose:
            print(f"🎬 Step 1/3: Rendering {len(timings)} scene clips with Ken Burns...")
        
        workers = max_workers if max_workers is not None else RENDER_MAX_WORKERS
        scene_clips = [None] * len(timings)
        total_scenes = len(timings)
        render_tasks = []

        for idx, timing in enumerate(timings):
            clip_filename = f"clip_{timing.index:02d}_dur_{timing.duration:.2f}.mp4"
            clip_path = clips_cache_dir / clip_filename
            scene_clips[idx] = clip_path
            
            if resume_from_scene is not None and timing.index < resume_from_scene:
                if verbose:
                    print(f"   [Scene {idx+1}/{total_scenes}] Scene {timing.index} is before resume index {resume_from_scene}. Skipping...")
                continue

            # Clean up any stale duration clips for this scene index
            for stale_file in clips_cache_dir.glob(f"clip_{timing.index:02d}_dur_*.mp4"):
                if stale_file.name != clip_filename:
                    try:
                        stale_file.unlink()
                    except Exception:
                        pass

            image_mtime = timing.image_path.stat().st_mtime
            clip_exists = clip_path.exists()
            clip_mtime = clip_path.stat().st_mtime if clip_exists else 0
            
            if clip_exists and clip_path.stat().st_size > 1024 and clip_mtime > image_mtime:
                if verbose:
                    print(f"   [Scene {idx+1}/{total_scenes}] Clip for scene {timing.index} already exists (cached). Skipping...")
            else:
                is_stale = clip_exists
                render_tasks.append((idx, timing, clip_path, is_stale))

        if render_tasks:
            if verbose:
                print(f"   🚀 Rendering {len(render_tasks)} clips in parallel with {workers} workers...")
                
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def render_worker(task):
                task_idx, t, out_path, is_stale = task
                if verbose:
                    status_str = "stale (regenerating)" if is_stale else "rendering"
                    print(f"   [Scene {task_idx+1}/{total_scenes}] Started {status_str} clip for scene {t.index} ({t.duration:.1f}s)...")
                
                _render_scene_clip(
                    image_path=t.image_path,
                    output_path=out_path,
                    duration=t.duration,
                    ken_burns=ken_burns,
                    scene_index=t.index,
                    encoder=encoder,
                    verbose=False,
                )
                
                # Upload scene clip immediately if gdrive is configured
                if gdrive_folder_id:
                    try:
                        from .gdrive import get_gdrive_service, upload_file_to_drive_folder
                        service = get_gdrive_service()
                        if service:
                            upload_file_to_drive_folder(service, gdrive_folder_id, out_path, "clips_cache")
                    except Exception as upload_err:
                        print(f"   ⚠️ Failed to upload clip for scene {t.index} to Google Drive: {upload_err}")
                
                if verbose:
                    print(f"   [Scene {task_idx+1}/{total_scenes}] Finished rendering clip for scene {t.index}.")
                return t.index

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(render_worker, task): task for task in render_tasks}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"   ❌ Error rendering clip in worker thread: {e}")
                        raise e

        # Step 2: Concatenate clips, mix audio, and burn subtitles in a single pass
        if verbose:
            print(f"🔗 Step 2/2: Concatenating clips, mixing audio, and burning subtitles in a single pass...")
        _concat_and_finalize(
            clips=scene_clips,
            voiceover_path=voiceover_path,
            output_path=output_path,
            transition_type=transition_type,
            transition_duration=transition_duration,
            subtitle_path=subtitle_path,
            bgm_path=bgm_path,
            bgm_volume=bgm_volume,
            subtitle_style=preset,
            encoder=encoder,
            verbose=verbose,
        )
        
        # Upload final output files immediately if gdrive is configured
        if gdrive_folder_id:
            try:
                from .gdrive import get_gdrive_service, upload_file_to_drive_folder
                service = get_gdrive_service()
                if service:
                    upload_file_to_drive_folder(service, gdrive_folder_id, output_path, "output")
                    if subtitle_path:
                        upload_file_to_drive_folder(service, gdrive_folder_id, subtitle_path, "output")
            except Exception as upload_err:
                print(f"   ⚠️ Failed to upload output files to Google Drive: {upload_err}")

    except Exception as e:
        if verbose:
            print(f"❌ Error during video assembly: {e}")
        raise e

    if verbose:
        duration = _get_duration(output_path)
        size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0.0
        print(f"✅ Video assembled: {output_path}")
        print(f"   Duration: {duration:.1f}s | Size: {size_mb:.1f} MB")

    return output_path


def _render_scene_clip(
    image_path: Path,
    output_path: Path,
    duration: float,
    ken_burns: bool,
    scene_index: int,
    encoder: str,
    verbose: bool,
):
    """Render a single scene image into a video clip with optional Ken Burns."""
    frames = int(duration * VIDEO_FPS)

    if ken_burns:
        # Alternate between zoom-in and zoom-out for variety
        if scene_index % 3 == 0:
            # Slow zoom in
            zoom_expr = f"min(zoom+0.0008,1.3)"
            x_expr = f"iw/2-(iw/zoom/2)"
            y_expr = f"ih/2-(ih/zoom/2)"
        elif scene_index % 3 == 1:
            # Slow pan left to right
            zoom_expr = "1.15"
            x_expr = f"(iw*0.15)*on/{frames}"
            y_expr = f"ih/2-(ih/zoom/2)"
        else:
            # Slow zoom out
            zoom_expr = f"max(1.3-0.0008*on,1.0)"
            x_expr = f"iw/2-(iw/zoom/2)"
            y_expr = f"ih/2-(ih/zoom/2)"

        filter_str = (
            f"scale=8000:-1,"
            f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':"
            f"d={frames}:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS},"
            f"format=yuv420p"
        )
    else:
        filter_str = (
            f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
            f"format=yuv420p"
        )

    # Build command based on encoder
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(image_path),
        "-vf", filter_str,
        "-t", str(duration),
        "-c:v", encoder,
    ]
    if encoder == "h264_nvenc":
        cmd.extend(["-preset", "fast", "-rc:v", "constqp", "-qp", "23"])
    else:
        cmd.extend(["-preset", "fast", "-crf", "23"])
    cmd.extend(["-an", str(output_path)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Try CPU encoding if GPU encoding fails
    if result.returncode != 0 and encoder == "h264_nvenc":
        if verbose:
            print(f"  ⚠️ GPU encoding failed for scene {scene_index}. Falling back to CPU...")
        cmd_cpu = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-vf", filter_str,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-an",
            str(output_path),
        ]
        result = subprocess.run(cmd_cpu, capture_output=True, text=True)

    if result.returncode != 0 and verbose:
        # Fall back to simple scale if zoompan fails
        print(f"  ⚠️ Ken Burns failed for scene {scene_index}, using simple scale")
        cmd_simple = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-vf", f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-an",
            str(output_path),
        ]
        subprocess.run(cmd_simple, capture_output=True, text=True)


def _build_ffmpeg_cmd(
    clips: list[Path],
    voiceover_path: Path,
    output_path: Path,
    transition_type: str = "fade",
    transition_duration: float = 0.5,
    subtitle_path: Optional[Path] = None,
    bgm_path: Optional[Path] = None,
    bgm_volume: float = 0.15,
    subtitle_style: Optional[dict] = None,
    encoder: str = "libx264",
) -> list[str]:
    """Build the single-pass FFmpeg command combining concatenation, subtitles, and audio mixing."""
    inputs = []
    for clip in clips:
        inputs.extend(["-i", str(clip)])
    inputs.extend(["-i", str(voiceover_path)])

    num_clips = len(clips)
    voice_idx = num_clips
    bgm_idx = None

    if bgm_path and bgm_path.exists():
        inputs.extend(["-i", str(bgm_path)])
        bgm_idx = num_clips + 1

    filter_parts = []

    # 1. Video Concatenation
    if num_clips == 1:
        v_map = "[0:v]"
    elif transition_type == "none" or transition_duration <= 0.0:
        concat_inputs = "".join(f"[{i}:v]" for i in range(num_clips))
        filter_parts.append(f"{concat_inputs}concat=n={num_clips}:v=1:a=0[vout]")
        v_map = "[vout]"
    else:
        # Get duration of each clip to set transition offsets
        durations = [_get_duration(clip) for clip in clips]
        current_label = "[0:v]"
        cumulative = 0.0
        for i in range(num_clips - 1):
            offset = cumulative + durations[i] - transition_duration
            next_label = f"[{i + 1}:v]"
            out_label = f"[v{i}]" if i < num_clips - 2 else "[vout]"
            filter_parts.append(
                f"{current_label}{next_label}xfade=transition={transition_type}:"
                f"duration={transition_duration}:offset={offset:.3f}{out_label}"
            )
            current_label = out_label
            cumulative = offset
        v_map = "[vout]"

    # 2. Subtitle Burning
    if subtitle_path and subtitle_path.exists():
        sub_color = subtitle_style.get("subtitle_color", "#FFFFFF").lstrip("#") if subtitle_style else "FFFFFF"
        font_size = 22
        margin_v = 40
        force_style = (
            f"FontSize={font_size},"
            f"PrimaryColour=&H00{sub_color[4:6]}{sub_color[2:4]}{sub_color[0:2]},"
            f"BackColour=&H00000000,"
            f"BorderStyle=1,"
            f"Outline=3,"
            f"Shadow=0,"
            f"MarginV={margin_v},"
            f"Alignment=2,"
            f"Bold=1"
        )
        sub_filter = f"subtitles={subtitle_path}:force_style='{force_style}'"
        filter_parts.append(f"{v_map}{sub_filter}[vfinal]")
        v_map = "[vfinal]"

    # 3. Audio Mixing
    if bgm_idx is not None:
        filter_parts.append(
            f"[{voice_idx}:a]volume=1.0[vo];"
            f"[{bgm_idx}:a]volume={bgm_volume}[bgm];"
            f"[vo][bgm]amix=inputs=2:duration=first[aout]"
        )
        audio_map = "[aout]"
    else:
        audio_map = f"{voice_idx}:a"

    # Build complete command line
    cmd = ["ffmpeg", "-y"] + inputs

    if filter_parts:
        filter_graph = ";".join(filter_parts)
        cmd.extend([
            "-filter_complex", filter_graph,
            "-map", v_map,
            "-map", audio_map,
        ])
    else:
        cmd.extend([
            "-map", "0:v",
            "-map", f"{voice_idx}:a",
        ])

    cmd.extend(["-c:v", encoder])
    if encoder == "h264_nvenc":
        cmd.extend(["-preset", "fast", "-rc:v", "constqp", "-qp", "20"])
    else:
        cmd.extend(["-preset", "medium", "-crf", "20"])

    cmd.extend([
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path),
    ])

    return cmd


def _concat_and_finalize(
    clips: list[Path],
    voiceover_path: Path,
    output_path: Path,
    transition_type: str = "fade",
    transition_duration: float = 0.5,
    subtitle_path: Optional[Path] = None,
    bgm_path: Optional[Path] = None,
    bgm_volume: float = 0.15,
    subtitle_style: Optional[dict] = None,
    encoder: str = "libx264",
    verbose: bool = False,
):
    """Concatenate clips, mix audio, and burn subtitles in a single high-performance FFmpeg pass."""
    cmd = _build_ffmpeg_cmd(
        clips=clips,
        voiceover_path=voiceover_path,
        output_path=output_path,
        transition_type=transition_type,
        transition_duration=transition_duration,
        subtitle_path=subtitle_path,
        bgm_path=bgm_path,
        bgm_volume=bgm_volume,
        subtitle_style=subtitle_style,
        encoder=encoder,
    )

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Try CPU encoding if GPU encoding fails
    if result.returncode != 0 and encoder == "h264_nvenc":
        if verbose:
            print("  ⚠️ GPU encoding failed during final assembly. Falling back to CPU...")
        cmd_cpu = _build_ffmpeg_cmd(
            clips=clips,
            voiceover_path=voiceover_path,
            output_path=output_path,
            transition_type=transition_type,
            transition_duration=transition_duration,
            subtitle_path=subtitle_path,
            bgm_path=bgm_path,
            bgm_volume=bgm_volume,
            subtitle_style=subtitle_style,
            encoder="libx264",
        )
        result = subprocess.run(cmd_cpu, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg assembly compilation failed: {result.stderr[:400]}")


def _get_duration(path: Path) -> float:
    """Get video/audio duration in seconds."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return 0.0
