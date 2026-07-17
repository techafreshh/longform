"""Reference video clip downloading and multimodal style analysis using Gemini."""

import json
import time
from pathlib import Path
from typing import Optional

def download_reference_clip_optional(
    video_id: str,
    output_path: Path,
    duration_seconds: int = 60,
    verbose: bool = True
) -> Optional[Path]:
    """
    Download a short clip of a YouTube video using yt-dlp if available.
    Does not crash if yt-dlp is not installed.
    """
    try:
        import yt_dlp
    except ImportError:
        if verbose:
            print("ℹ️  'yt-dlp' is not installed in the environment.")
            print("   To auto-download reference clips, please run: pip install yt-dlp")
            print("   Alternatively, you can manually place reference video clips (.mp4, .webm) in your project's 'reference_clips' folder.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    if verbose:
        print(f"  📥 Downloading short {duration_seconds}s clip for: {video_id}...")

    ydl_opts = {
        'format': 'mp4/best',
        'outtmpl': str(output_path),
        # Download only the first N seconds
        'download_ranges': lambda info_dict, ydl: [{'start_time': 0, 'end_time': duration_seconds}],
        'force_keyframes_at_cuts': True,
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        if output_path.exists() and output_path.stat().st_size > 1024:
            if verbose:
                print(f"  ✅ Reference clip downloaded: {output_path.name}")
            return output_path
    except Exception as e:
        if verbose:
            print(f"  ⚠️ yt-dlp download failed: {e}")
        # Try fallback without ranges in case the download_ranges callback is unsupported
        try:
            if verbose:
                print("  🔄 Retrying full download format fallback...")
            ydl_opts_fallback = {
                'format': 'mp4/best',
                'outtmpl': str(output_path),
                'quiet': True,
                'no_warnings': True,
                'max_filesize': 50 * 1024 * 1024, # Cap at 50MB to save bandwidth
            }
            with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl:
                ydl.download([video_url])
            if output_path.exists() and output_path.stat().st_size > 1024:
                return output_path
        except Exception as e_fallback:
            if verbose:
                print(f"  ⚠️ Fallback download failed: {e_fallback}")
    return None

def analyze_reference_clip_with_transcript(
    client,
    clip_path: Path,
    transcript_text: str,
    verbose: bool = True
) -> Optional[dict]:
    """
    Upload reference clip to Gemini and run multimodal analysis alongside the transcript text.
    """
    if verbose:
        print("📤 Uploading reference clip to Gemini File API...")

    try:
        # Upload the video file
        video_file = client.files.upload(file=clip_path)
        
        # Poll for processing completion
        if verbose:
            print("⏳ Waiting for Gemini to process the video (this can take 10-30s)...")
        
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)
            
        if video_file.state.name == "FAILED":
            raise RuntimeError("Gemini File API processing failed.")
            
        if verbose:
            print(f"✅ Video processed successfully. File ID: {video_file.name}")
            print("🔬 Running multimodal video and transcript analysis...")

        prompt = f"""You are a world-class YouTube producer and visual designer. Analyze this whiteboard or chalkboard animation video clip alongside its text transcript.
Identify visual motifs, backgrounds, styling, colors, and pacing choices so we can reproduce it.

Here is the transcript of this reference video:
---
{transcript_text}
---

Specifically analyze:
1. **Background Style**: The canvas surface (white background, chalkboard green/black, paper grain texture, light glare, grid outlines, hand movement presence).
2. **Drawing Style**: The details of drawings (bold marker strokes, thin chalk lines, coloring techniques, simple cartoons, educational diagrams, connectors/arrows).
3. **Pacing & Editing**: Frequency of drawing reveals, camera pans, camera zooms, or scene transitions (simple cuts vs crossfades vs camera glides).
4. **Visual Metaphors**: Highlight key symbols, icons, text labels, or visual tricks used to represent concepts.

You MUST return a JSON object (and nothing else, no markdown formatting fences or other text) with the following structure:
{{
  "background_style": "detailed background details",
  "drawing_style": "detailed drawing line, outline, and coloring details",
  "pacing_style": "camera movement and visual transition details",
  "visual_metaphors": "types of icons, metaphors, and arrows used",
  "reproduction_prompt_prefix": "A concise text-to-image prompt prefix (approx 30-50 words) that describes this whiteboard/chalkboard style to reproduce it. Do not include subject descriptions, focus on the style assets. e.g., 'Color whiteboard animation style, clean white background, colorful marker outlines...'"
}}"""

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[video_file, prompt]
        )
        
        raw_text = response.text.strip()
        
        # Clean up any potential markdown formatting fences
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()
            
        analysis = json.loads(raw_text)
        
        # Try to clean up the uploaded file to save space on Gemini
        try:
            client.files.delete(name=video_file.name)
        except Exception:
            pass
            
        if verbose:
            print("✅ Multimodal style analysis complete.")
            
        return analysis

    except Exception as e:
        if verbose:
            print(f"⚠️ Multimodal analysis failed: {e}")
        return None
