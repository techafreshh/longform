"""Video assembly module — FFmpeg-based slideshow with transitions, Ken Burns, and subtitles."""

import json
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from .config import VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS, STYLE_PRESETS


@dataclass
class SceneTiming:
    """Timing info for a single scene in the video."""
    index: int
    image_path: Path
    start_time: float  # seconds
    duration: float    # seconds
    narration: str


def build_scene_timings(
    scenes: list[dict],
    voice_segments: list[dict],
    image_dir: Path,
) -> list[SceneTiming]:
    """
    Build timing map: which image shows at which time, driven by voice durations.

    Args:
        scenes: Parsed scene list with 'index', 'description', 'narration'.
        voice_segments: Voice segment list with 'index', 'duration'.
        image_dir: Directory containing scene_XX.png images.

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
        duration = duration_map.get(idx, len(scene.get("narration", "").split()) * 0.4)

        # Minimum 3 seconds per scene
        duration = max(3.0, duration)

        image_path = image_dir / f"scene_{idx:02d}.png"
        if not image_path.exists():
            print(f"⚠️ Missing image for scene {idx}: {image_path}")
            continue

        timings.append(SceneTiming(
            index=idx,
            image_path=image_path,
            start_time=current_time,
            duration=duration,
            narration=scene.get("narration", ""),
        ))

        current_time += duration

    return timings


def generate_subtitles(
    timings: list[SceneTiming],
    word_timestamps: list[dict],
    output_path: Path,
    style: str = "color_whiteboard",
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

    Returns:
        Path to the generated SRT file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if word_timestamps:
        preset = STYLE_PRESETS.get(style, STYLE_PRESETS["color_whiteboard"])
        highlight_color = preset.get("subtitle_highlight", "#FFCC00")
        srt_content = _word_level_srt(word_timestamps, highlight_color=highlight_color)
    else:
        srt_content = _scene_level_srt(timings)

    output_path.write_text(srt_content, encoding="utf-8")
    return output_path


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _word_level_srt(timestamps: list[dict], words_per_group: int = 6, highlight_color: str = "#FFCC00") -> str:
    """Create SRT from word-level timestamps, grouping words and highlighting the active word."""
    srt_entries = []
    idx = 1

    for i in range(0, len(timestamps), words_per_group):
        group = timestamps[i:i + words_per_group]
        if not group:
            continue

        for word_index, active_word_info in enumerate(group):
            start = active_word_info["start"]
            end = active_word_info["end"]

            # Format words, wrapping the active word in color tags
            words_formatted = []
            for j, w in enumerate(group):
                word_text = w["word"].upper()
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


def _scene_level_srt(timings: list[SceneTiming]) -> str:
    """Create SRT from scene-level timings (fallback when no word timestamps)."""
    import re

    srt_entries = []
    idx = 1

    for timing in timings:
        # Split narration into ~10-word chunks
        words = timing.narration.split()
        chunk_size = 10
        chunks = [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

        if not chunks:
            continue

        chunk_duration = timing.duration / len(chunks)

        for j, chunk in enumerate(chunks):
            start = timing.start_time + j * chunk_duration
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
        scene_clips = []
        total_scenes = len(timings)
        for idx, timing in enumerate(timings):
            clip_filename = f"clip_{timing.index:02d}_dur_{timing.duration:.2f}.mp4"
            clip_path = clips_cache_dir / clip_filename
            
            if resume_from_scene is not None and timing.index < resume_from_scene:
                if verbose:
                    print(f"   [Scene {idx+1}/{total_scenes}] Scene {timing.index} is before resume index {resume_from_scene}. Skipping...")
                scene_clips.append(clip_path)
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
                if verbose:
                    if clip_exists:
                        print(f"   [Scene {idx+1}/{total_scenes}] Clip for scene {timing.index} is stale. Regenerating...")
                    else:
                        print(f"   [Scene {idx+1}/{total_scenes}] Rendering clip for scene {timing.index} ({timing.duration:.1f}s)...")
                _render_scene_clip(
                    image_path=timing.image_path,
                    output_path=clip_path,
                    duration=timing.duration,
                    ken_burns=ken_burns,
                    scene_index=timing.index,
                    encoder=encoder,
                    verbose=verbose,
                )
                
                # Upload scene clip immediately if gdrive is configured
                if gdrive_folder_id:
                    try:
                        from .gdrive import get_gdrive_service, upload_file_to_drive_folder
                        service = get_gdrive_service()
                        if service:
                            upload_file_to_drive_folder(service, gdrive_folder_id, clip_path, "clips_cache")
                    except Exception as upload_err:
                        print(f"   ⚠️ Failed to upload clip for scene {timing.index} to Google Drive: {upload_err}")
                        
            scene_clips.append(clip_path)

        # Step 2: Concatenate clips with transitions
        joined_path = clips_cache_dir / "joined.mp4"
        if transition_type != "none" and len(scene_clips) > 1:
            if verbose:
                print(f"🔗 Step 2/3: Concatenating clips with {transition_type} transitions...")
            _concat_with_transitions(
                clips=scene_clips,
                output=joined_path,
                transition=transition_type,
                transition_duration=transition_duration,
                encoder=encoder,
                verbose=verbose,
            )
        else:
            if verbose:
                print(f"🔗 Step 2/3: Concatenating clips (no transitions)...")
            _concat_simple(scene_clips, joined_path)

        # Step 3: Add audio (voiceover + optional BGM) and subtitles
        if verbose:
            print(f"🔧 Step 3/3: Finalizing video (mixing audio + burning subtitles)...")
        _finalize_video(
            video_path=joined_path,
            voiceover_path=voiceover_path,
            output_path=output_path,
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


def _concat_with_transitions(
    clips: list[Path],
    output: Path,
    transition: str = "fade",
    transition_duration: float = 0.5,
    encoder: str = "libx264",
    verbose: bool = False,
):
    """Concatenate clips with xfade transitions between them."""
    if len(clips) < 2:
        _concat_simple(clips, output)
        return

    # Build the complex filter graph for xfade
    inputs = []
    for clip in clips:
        inputs.extend(["-i", str(clip)])

    durations = [_get_duration(clip) for clip in clips]

    filter_parts = []
    current_label = "[0:v]"

    cumulative = 0.0
    for i in range(len(clips) - 1):
        offset = cumulative + durations[i] - transition_duration
        next_label = f"[{i + 1}:v]"
        out_label = f"[v{i}]" if i < len(clips) - 2 else "[vout]"

        filter_parts.append(
            f"{current_label}{next_label}xfade=transition={transition}:"
            f"duration={transition_duration}:offset={offset:.3f}{out_label}"
        )
        current_label = out_label
        cumulative = offset

    filter_graph = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_graph,
        "-map", "[vout]",
        "-c:v", encoder,
    ]
    if encoder == "h264_nvenc":
        cmd.extend(["-preset", "fast", "-rc:v", "constqp", "-qp", "23"])
    else:
        cmd.extend(["-preset", "fast", "-crf", "23"])
    cmd.append(str(output))

    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Try CPU encoding if GPU encoding fails
    if result.returncode != 0 and encoder == "h264_nvenc":
        if verbose:
            print("  ⚠️ GPU encoding failed during concatenation. Falling back to CPU...")
        cmd_cpu = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_graph,
            "-map", "[vout]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            str(output),
        ]
        result = subprocess.run(cmd_cpu, capture_output=True, text=True)

    if result.returncode != 0:
        if verbose:
            print(f"  ⚠️ xfade failed, falling back to simple concat")
            print(f"     Error: {result.stderr[:200]}")
        _concat_simple(clips, output)


def _concat_simple(clips: list[Path], output: Path):
    """Simple concatenation without transitions."""
    list_file = output.parent / "concat_list.txt"
    with open(list_file, "w") as f:
        for clip in clips:
            f.write(f"file '{clip.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output),
    ]

    subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink(missing_ok=True)


def _finalize_video(
    video_path: Path,
    voiceover_path: Path,
    output_path: Path,
    subtitle_path: Optional[Path],
    bgm_path: Optional[Path],
    bgm_volume: float,
    subtitle_style: dict,
    encoder: str,
    verbose: bool,
):
    """Add audio tracks and subtitles to the assembled video."""
    inputs = ["-i", str(video_path), "-i", str(voiceover_path)]
    filter_parts = []

    # Audio mixing
    if bgm_path and bgm_path.exists():
        inputs.extend(["-i", str(bgm_path)])
        filter_parts.append(
            f"[1:a]volume=1.0[vo];"
            f"[2:a]volume={bgm_volume}[bgm];"
            f"[vo][bgm]amix=inputs=2:duration=first[aout]"
        )
        audio_map = "[aout]"
    else:
        audio_map = "1:a"

    # Build subtitle filter
    sub_filter = ""
    if subtitle_path and subtitle_path.exists():
        sub_color = subtitle_style.get("subtitle_color", "#FFFFFF").lstrip("#")
        sub_bg = "000000"
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

    # Build final command
    cmd = ["ffmpeg", "-y"] + inputs

    if filter_parts and sub_filter:
        full_filter = ";".join(filter_parts)
        cmd.extend([
            "-filter_complex", full_filter,
            "-vf", sub_filter,
            "-map", "0:v",
            "-map", audio_map,
        ])
    elif filter_parts:
        cmd.extend([
            "-filter_complex", ";".join(filter_parts),
            "-map", "0:v",
            "-map", audio_map,
        ])
    elif sub_filter:
        cmd.extend([
            "-vf", sub_filter,
            "-map", "0:v",
            "-map", "1:a",
        ])
    else:
        cmd.extend([
            "-map", "0:v",
            "-map", "1:a",
        ])

    cmd.extend([
        "-c:v", encoder,
    ])
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

    if verbose:
        print("  🔧 Finalizing video (adding audio + subtitles)...")

    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Fallback to CPU if GPU encoding fails
    if result.returncode != 0 and encoder == "h264_nvenc":
        if verbose:
            print("  ⚠️ GPU encoding failed during finalization. Falling back to CPU...")
        cmd_cpu = ["ffmpeg", "-y"] + inputs
        if filter_parts and sub_filter:
            cmd_cpu.extend([
                "-filter_complex", full_filter,
                "-vf", sub_filter,
                "-map", "0:v",
                "-map", audio_map,
            ])
        elif filter_parts:
            cmd_cpu.extend([
                "-filter_complex", ";".join(filter_parts),
                "-map", "0:v",
                "-map", audio_map,
            ])
        elif sub_filter:
            cmd_cpu.extend([
                "-vf", sub_filter,
                "-map", "0:v",
                "-map", "1:a",
            ])
        else:
            cmd_cpu.extend([
                "-map", "0:v",
                "-map", "1:a",
            ])
        cmd_cpu.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(output_path),
        ])
        result = subprocess.run(cmd_cpu, capture_output=True, text=True)

    if result.returncode != 0 and verbose:
        print(f"  ⚠️ Finalize warning/error: {result.stderr[:300]}")


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
