"""Voice generation module — Fish Audio (primary) + Qwen3-TTS (fallback on Colab GPU)."""

import io
import json
import wave
import struct
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from .config import FISH_API_KEY, FISH_VOICE_ID


def _prepare_text_for_tts(text: str) -> str:
    """Clean text and convert punctuation pauses to Fish Audio S2 inline tags.

    Fish Audio S2 models use natural-language inline tags like [pause] and
    [long pause] for breath control — standard punctuation (ellipsis, em-dash)
    is largely ignored for pacing.  This function converts script-level pause
    cues into the tags the model actually respects.
    """
    import re
    # Strip any residual markdown emphasis markers
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.+?)_{1,3}', r'\1', text)
    # Strip markdown headers and list bullets
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    # Convert ellipsis to Fish Audio pause tag
    text = re.sub(r'\.{3,}', ' [pause] ', text)
    # Convert em-dash to Fish Audio pause tag
    text = text.replace('\u2014', ' [pause] ')
    # Convert double newlines to long pause
    text = re.sub(r'\n\s*\n', ' [long pause] ', text)
    # Collapse remaining newlines to spaces
    text = re.sub(r'\n', ' ', text)
    # Clean up multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _strip_markdown_for_tts(text: str) -> str:
    """Strip markdown formatting from text for TTS engines that respect punctuation natively (e.g. Qwen)."""
    import re
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.+?)_{1,3}', r'\1', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    return text.strip()


@dataclass
class VoiceSegment:
    """A single voiced segment with timing info."""
    index: int
    text: str
    audio_path: Path
    duration: float  # seconds


@dataclass
class VoiceResult:
    """Complete voiceover result with all segments and timing."""
    segments: list[VoiceSegment]
    combined_audio: Path
    total_duration: float
    timestamps: list[dict]  # word-level timing from Whisper

    def save_timestamps(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "total_duration": self.total_duration,
                "segments": [
                    {
                        "index": s.index,
                        "duration": s.duration,
                        "audio_path": str(s.audio_path),
                    }
                    for s in self.segments
                ],
                "word_timestamps": self.timestamps,
            }, f, indent=2)


# ---------------------------------------------------------------------------
# Fish Audio (Primary)
# ---------------------------------------------------------------------------

def generate_voice_fish(
    scenes: list[dict],
    output_dir: Path,
    voice_id: Optional[str] = None,
    model: str = "s2.1-pro-free",
    style: str = "color_whiteboard",
    verbose: bool = True,
) -> VoiceResult:
    """
    Generate voiceover using Fish Audio API.

    Args:
        scenes: List of scene dicts with 'index' and 'narration' keys.
        output_dir: Directory to save audio files.
        voice_id: Fish Audio voice ID (from cloned voice).
        model: Fish Audio model name.
        style: Visual style key.
        verbose: Print progress.

    Returns:
        VoiceResult with all segments and combined audio.
    """
    try:
        from fishaudio import FishAudio
    except ImportError:
        raise ImportError("Install fish-audio-sdk: pip install fish-audio-sdk")

    import os
    api_key = FISH_API_KEY or os.getenv("FISH_API_KEY", "")
    if not api_key:
        raise ValueError("FISH_API_KEY is not set. Add it to your .env file.")

    voice_id = voice_id or FISH_VOICE_ID or os.getenv("FISH_VOICE_ID", "")
    if not voice_id:
        raise ValueError(
            "No voice ID set. Clone your voice at https://fish.audio "
            "and set FISH_VOICE_ID in your .env file."
        )
    voice_id = str(voice_id).strip("'\" \t\r\n")

    client = FishAudio(api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = []
    for i, scene in enumerate(scenes):
        idx = scene["index"]
        text = _prepare_text_for_tts(scene["narration"])
        is_last = (i == len(scenes) - 1)

        if verbose:
            print(f"  🎙️ Generating voice for scene {idx}/{len(scenes)}...")

        audio_path = output_dir / f"scene_{idx:02d}.wav"

        try:
            audio = client.tts.convert(
                text=text,
                model=model,
                reference_id=voice_id,
            )
            # Save audio bytes
            with open(audio_path, "wb") as f:
                if hasattr(audio, 'read'):
                    f.write(audio.read())
                elif isinstance(audio, bytes):
                    f.write(audio)
                else:
                    # Handle generator/iterator
                    for chunk in audio:
                        f.write(chunk)

        except Exception as e:
            print(f"  ⚠️ Fish Audio failed for scene {idx}: {e}")
            # Create a silent placeholder
            _create_silence(audio_path, duration=len(text.split()) * 0.4)

        # Get duration and pad if necessary to match min_scene_duration
        duration = _get_audio_duration(audio_path)
        min_dur = 1.0 if style == "stickman" else 3.0
        if duration < min_dur:
            padding_needed = min_dur - duration
            if verbose:
                print(f"    ⏳ Narration too short ({duration:.2f}s). Padding with {padding_needed:.2f}s silence to match min_scene_duration ({min_dur:.1f}s)...")
            _pad_wav_with_silence(audio_path, padding_needed)
            duration = _get_audio_duration(audio_path)

        segments.append(VoiceSegment(
            index=idx,
            text=text,
            audio_path=audio_path,
            duration=duration,
        ))

    # Combine all segments
    combined = output_dir / "voiceover.wav"
    _combine_audio(segments, combined, verbose)

    # Generate word-level timestamps with Whisper (if available)
    script_text = "\n\n".join(s["narration"] for s in scenes)
    timestamps = _transcribe_timestamps(combined, script_text, verbose)

    total_duration = sum(s.duration for s in segments)

    if verbose:
        print(f"✅ Voice generation complete: {total_duration:.1f}s total")

    return VoiceResult(
        segments=segments,
        combined_audio=combined,
        total_duration=total_duration,
        timestamps=timestamps,
    )


# ---------------------------------------------------------------------------
# Qwen3-TTS (Colab GPU fallback)
# ---------------------------------------------------------------------------

def generate_voice_qwen(
    scenes: list[dict],
    output_dir: Path,
    reference_audio: Optional[str] = None,
    model_size: str = "1.7B",
    style: str = "color_whiteboard",
    verbose: bool = True,
) -> VoiceResult:
    """
    Generate voiceover using Qwen3-TTS on local GPU (Colab T4).

    Args:
        scenes: List of scene dicts with 'index' and 'narration' keys.
        output_dir: Directory to save audio files.
        reference_audio: Path to voice reference audio for cloning.
        model_size: "0.6B" or "1.7B".
        style: Visual style key.
        verbose: Print progress.

    Returns:
        VoiceResult with all segments and combined audio.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torchaudio
    except ImportError:
        raise ImportError(
            "Qwen3-TTS requires: pip install torch torchaudio transformers"
        )

    model_name = f"Qwen/Qwen3-TTS-{model_size}"
    if verbose:
        print(f"🔄 Loading Qwen3-TTS {model_size}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and verbose:
        print("⚠️  No GPU detected — Qwen3-TTS will be very slow on CPU")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    segments = []

    for i, scene in enumerate(scenes):
        idx = scene["index"]
        text = _strip_markdown_for_tts(scene["narration"])
        is_last = (i == len(scenes) - 1)

        if verbose:
            print(f"  🎙️ Generating voice for scene {idx}/{len(scenes)}...")

        audio_path = output_dir / f"scene_{idx:02d}.wav"

        try:
            # Build input for voice cloning or voice design
            if reference_audio:
                # Voice cloning mode
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    audio=reference_audio,
                ).to(device)
            else:
                # Voice design mode (describe desired voice)
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    voice_description="A warm, clear, male narrator voice. "
                                      "Confident but friendly tone. Medium pace.",
                ).to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=4096)

            # Decode audio
            audio_tensor = tokenizer.decode_audio(outputs[0])
            torchaudio.save(str(audio_path), audio_tensor.cpu(), 24000)

        except Exception as e:
            print(f"  ⚠️ Qwen3-TTS failed for scene {idx}: {e}")
            _create_silence(audio_path, duration=len(text.split()) * 0.4)

        # Get duration and pad if necessary to match min_scene_duration
        duration = _get_audio_duration(audio_path)
        min_dur = 1.0 if style == "stickman" else 3.0
        if duration < min_dur:
            padding_needed = min_dur - duration
            if verbose:
                print(f"    ⏳ Narration too short ({duration:.2f}s). Padding with {padding_needed:.2f}s silence to match min_scene_duration ({min_dur:.1f}s)...")
            _pad_wav_with_silence(audio_path, padding_needed)
            duration = _get_audio_duration(audio_path)

        segments.append(VoiceSegment(
            index=idx,
            text=text,
            audio_path=audio_path,
            duration=duration,
        ))

    # Cleanup GPU memory
    if device == "cuda":
        import torch
        del model, tokenizer
        torch.cuda.empty_cache()

    combined = output_dir / "voiceover.wav"
    _combine_audio(segments, combined, verbose)
    script_text = "\n\n".join(s["narration"] for s in scenes)
    timestamps = _transcribe_timestamps(combined, script_text, verbose)
    total_duration = sum(s.duration for s in segments)

    if verbose:
        print(f"✅ Qwen3-TTS complete: {total_duration:.1f}s total")

    return VoiceResult(
        segments=segments,
        combined_audio=combined,
        total_duration=total_duration,
        timestamps=timestamps,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _combine_audio(segments: list[VoiceSegment], output: Path, verbose: bool = True):
    """Combine multiple WAV files into one using FFmpeg."""
    if not segments:
        return

    # Create a concat file list — no artificial silence between segments
    list_file = output.parent / "concat_list.txt"
    with open(list_file, "w") as f:
        for seg in segments:
            f.write(f"file '{seg.audio_path.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c:a", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        str(output),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ FFmpeg concat error: {result.stderr[:200]}")

    # Clean up
    list_file.unlink(missing_ok=True)


def _get_audio_duration(path: Path) -> float:
    """Get duration of an audio file in seconds using FFprobe."""
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


def _create_silence(path: Path, duration: float = 5.0, sample_rate: int = 44100):
    """Create a silent WAV file as a placeholder."""
    n_frames = int(sample_rate * duration)
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))


def _transcribe_timestamps(audio_path: Path, script_text: Optional[str] = None, verbose: bool = True) -> list[dict]:
    """
    Generate word-level timestamps using Whisper.
    Returns empty list if Whisper is not available.
    """
    try:
        import whisper
    except ImportError:
        if verbose:
            print("  ℹ️ Whisper not installed — skipping word-level timestamps")
        return []

    if verbose:
        print("  🔤 Generating word-level timestamps with Whisper...")

    try:
        model = whisper.load_model("base")
        # Direct Whisper to transcribe accurately matching original script vocabulary/phrasing
        # Whisper initial_prompt expects ~224 tokens (~1000 characters) limit
        initial_prompt = script_text[:1000] if script_text else None
        result = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language="en",
            initial_prompt=initial_prompt,
        )

        timestamps = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                timestamps.append({
                    "word": word_info["word"].strip(),
                    "start": round(word_info["start"], 3),
                    "end": round(word_info["end"], 3),
                })

        if verbose:
            print(f"  ✅ Got {len(timestamps)} word timestamps")
        return timestamps

    except Exception as e:
        if verbose:
            print(f"  ⚠️ Whisper failed: {e}")
        return []


def clone_voice_fish(
    audio_path: str,
    voice_name: str = "my-voice",
    verbose: bool = True,
) -> str:
    """
    Clone a voice on Fish Audio from a reference audio file.

    Args:
        audio_path: Path to 5-15 second voice sample.
        voice_name: Name for the cloned voice.
        verbose: Print progress.

    Returns:
        The voice_id to use in future TTS calls.
    """
    try:
        from fishaudio import FishAudio
    except ImportError:
        raise ImportError("Install fish-audio-sdk: pip install fish-audio-sdk")

    import os
    api_key = FISH_API_KEY or os.getenv("FISH_API_KEY", "")
    if not api_key:
        raise ValueError("FISH_API_KEY is not set.")

    client = FishAudio(api_key=api_key)

    if verbose:
        print(f"🎤 Cloning voice from {audio_path}...")

    with open(audio_path, "rb") as f:
        result = client.voices.create(
            name=voice_name,
            audio=f,
        )

    voice_id = result.id if hasattr(result, 'id') else str(result)

    if verbose:
        print(f"✅ Voice cloned! ID: {voice_id}")
        print(f"   Add this to your .env: FISH_VOICE_ID={voice_id}")

    return voice_id


def _pad_wav_with_silence(path: Path, padding_duration: float):
    """Append silence to a WAV file."""
    temp_padded = path.parent / f"temp_{path.name}"
    silence_file = path.parent / f"silence_{path.stem}.wav"
    
    # Create silence segment
    _create_silence(silence_file, duration=padding_duration)
    
    # Concatenate original WAV and silence
    list_file = path.parent / f"concat_{path.stem}.txt"
    with open(list_file, "w") as f:
        f.write(f"file '{path.resolve()}'\n")
        f.write(f"file '{silence_file.resolve()}'\n")
        
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",  # Direct stream copy for speed/lossless WAV
        str(temp_padded),
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    
    # Cleanup
    list_file.unlink(missing_ok=True)
    silence_file.unlink(missing_ok=True)
    
    # Overwrite original file
    if temp_padded.exists():
        temp_padded.replace(path)

