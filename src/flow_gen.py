"""Google Flow image generation via whisk-api or flow-agent.

Replaces Gemini API image generation with Google Flow's Nano Banana models.
Primary: whisk-api (direct CLI, cookie auth, no Chrome extension needed)
Fallback: flow-agent HTTP API (requires Chrome extension + tunnel)
"""

import glob
import shutil
import subprocess
import time
import json
from pathlib import Path
from typing import Optional

from .config import (
    STYLE_PRESETS,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    FLOW_AGENT_URL,
    FLOW_IMAGE_ASPECT,
    FLOW_REQUEST_DELAY,
    WHISK_COOKIE,
    WHISK_OUTPUT_DIR,
)


import os
from dotenv import load_dotenv

# Aspect ratio mapping for whisk-api
ASPECT_MAP = {
    "landscape": "LANDSCAPE",
    "portrait": "PORTRAIT",
    "square": "SQUARE",
}


def _get_whisk_cookie() -> str:
    """Dynamically fetch WHISK_COOKIE from environment or reload .env."""
    cookie = os.getenv("WHISK_COOKIE", "")
    if not cookie:
        load_dotenv(override=True)
        cookie = os.getenv("WHISK_COOKIE", "")
    return cookie.strip()


def _get_backend() -> str:
    """Determine which backend to use: 'whisk' or 'flow-agent'."""
    if _get_whisk_cookie():
        return "whisk"
    if FLOW_AGENT_URL:
        return "flow-agent"
    return "none"


def check_whisk_available(verbose: bool = True) -> bool:
    """Check if whisk-api CLI is installed and cookie is set."""
    cookie = _get_whisk_cookie()
    if not cookie:
        if verbose:
            print("❌ WHISK_COOKIE not set in .env or os.environ")
        return False

    try:
        result = subprocess.run(
            ["whisk", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            if verbose:
                print(f"✅ whisk-api available: v{result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass

    if verbose:
        print("❌ whisk CLI not found. Install with: npm i -g @rohitaryal/whisk-api")
    return False


def check_flow_agent_available(verbose: bool = True) -> bool:
    """Check if flow-agent API is reachable."""
    if not FLOW_AGENT_URL:
        return False

    import requests
    url = FLOW_AGENT_URL.rstrip("/")
    try:
        resp = requests.get(f"{url}/health", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            ext_connected = data.get("extension_connected", False)
            if verbose:
                print(f"✅ flow-agent reachable at {url}")
                print(f"   Extension connected: {ext_connected}")
            return ext_connected
    except Exception:
        pass

    if verbose:
        print(f"❌ Cannot reach flow-agent at {url}")
    return False


def check_flow_health(verbose: bool = True) -> bool:
    """Check if any Flow backend is available."""
    backend = _get_backend()
    if backend == "whisk":
        return check_whisk_available(verbose=verbose)
    elif backend == "flow-agent":
        return check_flow_agent_available(verbose=verbose)
    else:
        if verbose:
            print("❌ No Flow backend configured.")
            print("   Set WHISK_COOKIE (recommended) or FLOW_AGENT_URL in .env")
        return False


def _build_flow_prompt(description: str, style_prefix: str) -> str:
    """Build a Flow-ready prompt from a scene description and style prefix."""
    return (
        f"{style_prefix}. "
        f"Subject: {description}. "
        f"Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT}, landscape orientation. "
        f"The illustration should clearly communicate the concept. "
        f"No text or words in the image unless specifically relevant."
    )


def _generate_with_whisk(
    prompt: str,
    output_path: Path,
    aspect: str = "landscape",
    verbose: bool = True,
) -> bool:
    """Generate a single image using whisk-api CLI.

    Args:
        prompt: The text prompt for image generation.
        output_path: Where to save the generated image.
        aspect: Image aspect ratio ("landscape", "portrait", "square").
        verbose: Print progress.

    Returns:
        True if successful, False otherwise.
    """
    whisk_aspect = ASPECT_MAP.get(aspect, "LANDSCAPE")
    temp_dir = Path(WHISK_OUTPUT_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Count existing files to detect new ones
    existing_files = set(temp_dir.glob("*.png"))

    cmd = [
        "whisk", "generate",
        "--prompt", prompt,
        "--aspect", whisk_aspect,
        "--cookie", _get_whisk_cookie(),
        "--dir", str(temp_dir),
    ]

    try:
        if verbose:
            print(f"    📤 Generating via whisk-api...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            if verbose:
                print(f"    ❌ whisk failed: {result.stderr[:200]}")
            return False

        # Find the new file (whisk saves with auto-generated name)
        new_files = set(temp_dir.glob("*.png")) - existing_files
        if not new_files:
            # Try .webp too
            new_files = set(temp_dir.glob("*.webp")) - existing_files

        if not new_files:
            if verbose:
                print(f"    ❌ No output file found in {temp_dir}")
            return False

        # Move the newest file to the target path
        src_file = max(new_files, key=lambda f: f.stat().st_mtime)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_file), str(output_path))

        if verbose:
            size_kb = output_path.stat().st_size / 1024
            print(f"    ✅ Saved: {output_path.name} ({size_kb:.0f} KB)")
        return True

    except subprocess.TimeoutExpired:
        if verbose:
            print(f"    ❌ whisk timed out (120s)")
        return False
    except Exception as e:
        if verbose:
            print(f"    ❌ whisk error: {e}")
        return False


def _generate_with_flow_agent(
    prompt: str,
    output_path: Path,
    aspect: str = "landscape",
    verbose: bool = True,
) -> bool:
    """Generate a single image via flow-agent HTTP API.

    Args:
        prompt: The text prompt for image generation.
        output_path: Where to save the downloaded image.
        aspect: Image aspect ratio.
        verbose: Print progress.

    Returns:
        True if successful, False otherwise.
    """
    import requests

    url = FLOW_AGENT_URL.rstrip("/")

    payload = {
        "prompt": prompt,
        "aspect": aspect,
        "n": 1,
    }

    try:
        if verbose:
            print(f"    📤 Generating via flow-agent API...")
        resp = requests.post(
            f"{url}/v1/images/generations",
            json=payload,
            timeout=120,
        )
    except requests.RequestException as e:
        if verbose:
            print(f"    ❌ Request failed: {e}")
        return False

    if resp.status_code != 200:
        if verbose:
            print(f"    ❌ flow-agent returned status {resp.status_code}: {resp.text[:200]}")
        return False

    data = resp.json()

    # Extract image URL from response (OpenAI-compatible format)
    image_url = None
    if "data" in data and data["data"]:
        item = data["data"][0]
        image_url = item.get("url") or item.get("b64_json")
    elif "url" in data:
        image_url = data["url"]
    elif "filename" in data:
        image_url = f"{url}/download/{data['filename']}"

    if not image_url:
        if verbose:
            print(f"    ❌ No image URL in response: {json.dumps(data)[:200]}")
        return False

    # Download the image
    try:
        if verbose:
            print(f"    ⬇️ Downloading image...")
        img_resp = requests.get(image_url, timeout=60)
        img_resp.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(img_resp.content)

        if verbose:
            size_kb = len(img_resp.content) / 1024
            print(f"    ✅ Saved: {output_path.name} ({size_kb:.0f} KB)")
        return True

    except requests.RequestException as e:
        if verbose:
            print(f"    ❌ Download failed: {e}")
        return False


def _generate_single_image(
    prompt: str,
    output_path: Path,
    aspect: str = "landscape",
    verbose: bool = True,
) -> bool:
    """Generate a single image using the best available backend."""
    backend = _get_backend()

    if backend == "whisk":
        return _generate_with_whisk(prompt, output_path, aspect, verbose)
    elif backend == "flow-agent":
        return _generate_with_flow_agent(prompt, output_path, aspect, verbose)
    else:
        if verbose:
            print("    ❌ No Flow backend available")
        return False


def generate_scenes_flow(
    scenes: list[dict],
    style: str,
    output_dir: Path,
    force: bool = False,
    force_scenes: Optional[list[int]] = None,
    resume_from_scene: Optional[int] = None,
    verbose: bool = True,
) -> list[Path]:
    """Generate scene images using Google Flow.

    Drop-in replacement for scene_gen.generate_scenes() that uses
    Google Flow's Nano Banana models instead of the Gemini API.

    Args:
        scenes: List of scene dicts with 'index' and 'description' keys.
        style: Style preset key ("color_whiteboard", "chalkboard", "stickman").
        output_dir: Directory to save generated images.
        force: Force generate all scene images.
        force_scenes: Specific scene indices to force generate.
        resume_from_scene: Optional scene index to resume from.
        verbose: Print progress.

    Returns:
        List of paths to generated scene images.
    """
    # Verify backend is available
    if not check_flow_health(verbose=verbose):
        raise ConnectionError(
            "No Flow backend available. Set one of:\n"
            "  WHISK_COOKIE=<your-google-cookie> (recommended)\n"
            "  FLOW_AGENT_URL=<your-ngrok-url> (requires Chrome extension)"
        )

    # Get style preset
    preset = STYLE_PRESETS.get(style)
    if not preset:
        raise ValueError(f"Unknown style: {style}. Available: {list(STYLE_PRESETS.keys())}")

    # Check for custom style analysis
    reproduction_prefix = None
    analysis_file = output_dir.parent / "reference_style_analysis.json"
    if analysis_file.exists():
        try:
            with open(analysis_file, "r", encoding="utf-8") as f:
                analysis_data = json.load(f)
                reproduction_prefix = analysis_data.get("reproduction_prompt_prefix")
                if verbose and reproduction_prefix:
                    print(f"🎨 Using custom style from reference clips")
        except Exception:
            pass

    style_prefix = reproduction_prefix if reproduction_prefix else preset["image_prompt_prefix"]
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = []
    report_records = []
    aspect = FLOW_IMAGE_ASPECT
    backend = _get_backend()

    if verbose:
        print(f"🌐 Backend: {backend}")

    for scene in scenes:
        idx = scene["index"]
        description = scene["description"]
        image_path = output_dir / f"scene_{idx:02d}.png"

        # Skip if before resume index
        if resume_from_scene is not None and idx < resume_from_scene:
            if verbose:
                print(f"  ℹ️ Scene {idx}: skipped (before resume)")
            generated_paths.append(image_path)
            report_records.append({
                "scene_index": idx, "description": description,
                "status": "Skipped (Before Resume)", "source": "N/A",
            })
            continue

        # Skip if exists and not forcing
        if image_path.exists() and not force:
            if not force_scenes or idx not in force_scenes:
                if verbose:
                    print(f"  ℹ️ Scene {idx}: exists, skipping")
                generated_paths.append(image_path)
                report_records.append({
                    "scene_index": idx, "description": description,
                    "status": "Skipped (Exists)", "source": "N/A",
                })
                continue

        # Build prompt and generate
        prompt = _build_flow_prompt(description, style_prefix)
        if verbose:
            print(f"  🎨 Scene {idx}/{len(scenes)}: {description[:60]}...")

        success = _generate_single_image(
            prompt=prompt,
            output_path=image_path,
            aspect=aspect,
            verbose=verbose,
        )

        if success:
            generated_paths.append(image_path)
            report_records.append({
                "scene_index": idx, "description": description,
                "status": "Success", "source": f"google-flow ({backend})",
            })
        else:
            if verbose:
                print(f"  ⚠️ Scene {idx}: failed, creating placeholder")
            _create_flow_placeholder(image_path, description, preset)
            generated_paths.append(image_path)
            report_records.append({
                "scene_index": idx, "description": description,
                "status": "Failed (Placeholder)", "source": "placeholder",
            })

        # Rate limiting
        if idx < len(scenes):
            time.sleep(FLOW_REQUEST_DELAY)

    # Write report
    report_path = output_dir / "flow_generation_report.csv"
    try:
        import csv
        with open(report_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["scene_index", "description", "status", "source"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in report_records:
                writer.writerow(record)
        if verbose:
            print(f"\n📊 Report saved to: {report_path}")
    except Exception:
        pass

    success_count = sum(1 for r in report_records if r["status"] == "Success")
    if verbose:
        print(f"✅ Generated {success_count}/{len(scenes)} images via Google Flow")

    return generated_paths


def generate_scenes_flow_batch(
    scenes: list[dict],
    style: str,
    output_dir: Path,
    verbose: bool = True,
) -> Path:
    """Export prompts as JSON for batch processing.

    Args:
        scenes: List of scene dicts with 'index' and 'description'.
        style: Style preset key.
        output_dir: Directory to save the batch file.
        verbose: Print progress.

    Returns:
        Path to the generated batch JSON file.
    """
    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["color_whiteboard"])
    style_prefix = preset["image_prompt_prefix"]

    batch_items = []
    for scene in scenes:
        idx = scene["index"]
        description = scene["description"]
        prompt = _build_flow_prompt(description, style_prefix)
        batch_items.append({
            "scene_index": idx,
            "description": description,
            "prompt": prompt,
            "output_filename": f"scene_{idx:02d}.png",
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_path = output_dir / "flow_batch_prompts.json"
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump(batch_items, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"📝 Exported {len(batch_items)} prompts to: {batch_path}")

    return batch_path


def _create_flow_placeholder(path: Path, text: str, preset: dict):
    """Create a placeholder image when Flow generation fails."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        bg_color = preset.get("bg_color", "#FFFFFF")
        text_color = preset.get("text_color", "#1A1A1A")

        img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), bg_color)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        except OSError:
            font = ImageFont.load_default()

        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            test = f"{current_line} {word}".strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] > VIDEO_WIDTH - 200:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test
        if current_line:
            lines.append(current_line)

        total_height = len(lines) * 50
        y_start = (VIDEO_HEIGHT - total_height) // 2
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            x = (VIDEO_WIDTH - (bbox[2] - bbox[0])) // 2
            draw.text((x, y_start + i * 50), line, fill=text_color, font=font)

        draw.rectangle([(20, 20), (VIDEO_WIDTH - 20, VIDEO_HEIGHT - 20)], outline=text_color, width=3)

        try:
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except OSError:
            small_font = font
        draw.text((VIDEO_WIDTH - 250, VIDEO_HEIGHT - 50), "FLOW PLACEHOLDER", fill="#999999", font=small_font)

        img.save(str(path))

    except ImportError:
        import struct
        import zlib

        def create_minimal_png(width, height, color):
            def chunk(chunk_type, data):
                c = chunk_type + data
                return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)
            header = b'\x89PNG\r\n\x1a\n'
            ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            raw = b''
            for _ in range(height):
                raw += b'\x00' + bytes(color) * width
            idat = chunk(b'IDAT', zlib.compress(raw))
            iend = chunk(b'IEND', b'')
            return header + ihdr + idat + iend

        png_data = create_minimal_png(VIDEO_WIDTH, VIDEO_HEIGHT, (255, 255, 255))
        with open(path, 'wb') as f:
            f.write(png_data)
