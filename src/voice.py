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
    verbose: bool = True,
) -> VoiceResult:
    """
    Generate voiceover using Fish Audio API.

    Args:
        scenes: List of scene dicts with 'index' and 'narration' keys.
        output_dir: Directory to save audio files.
        voice_id: Fish Audio voice ID (from cloned voice).
        model: Fish Audio model name.
        verbose: Print progress.

    Returns:
        VoiceResult with all segments and combined audio.
    """
    try:
        from fishaudio import FishAudio
    except ImportError:
        raise ImportError("Install fish-audio-sdk: pip install fish-audio-sdk")

    if not FISH_API_KEY:
        raise ValueError("FISH_API_KEY is not set. Add it to your .env file.")

    voice_id = voice_id or FISH_VOICE_ID
    if not voice_id:
        raise ValueError(
            "No voice ID set. Clone your voice at https://fish.audio "
            "and set FISH_VOICE_ID in your .env file."
        )
    voice_id = str(voice_id).strip("'\" \t\r\n")

    client = FishAudio(api_key=FISH_API_KEY)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = []
    for scene in scenes:
        idx = scene["index"]
        text = scene["narration"]

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

        # Get duration
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
    timestamps = _transcribe_timestamps(combined, verbose)

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
    verbose: bool = True,
) -> VoiceResult:
    """
    Generate voiceover using Qwen3-TTS on local GPU (Colab T4).

    Args:
        scenes: List of scene dicts with 'index' and 'narration' keys.
        output_dir: Directory to save audio files.
        reference_audio: Path to voice reference audio for cloning.
        model_size: "0.6B" or "1.7B".
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

    for scene in scenes:
        idx = scene["index"]
        text = scene["narration"]

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
    timestamps = _transcribe_timestamps(combined, verbose)
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

    # Create a concat file list
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


def _transcribe_timestamps(audio_path: Path, verbose: bool = True) -> list[dict]:
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
        result = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language="en",
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

    if not FISH_API_KEY:
        raise ValueError("FISH_API_KEY is not set.")

    client = FishAudio(api_key=FISH_API_KEY)

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
