"""Scene image generation module — Gemini Flash Image for whiteboard/chalkboard illustrations."""

import time
import base64
from pathlib import Path
from typing import Optional

from .config import GOOGLE_API_KEY, STYLE_PRESETS, VIDEO_WIDTH, VIDEO_HEIGHT, slugify, get_genai_client, USE_VERTEX


def _generate_single_image(client, prompt: str, types, verbose: bool = True) -> bytes:
    """Helper to generate an image using Imagen 3 with fallbacks to Pollinations.ai and Gemini 2.5 Flash."""
    # 1. Try Imagen 3 first (official image generation API)
    try:
        result = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9",
                output_mime_type="image/png",
            )
        )
        if result and result.generated_images:
            return result.generated_images[0].image.image_bytes
    except Exception as e:
        if verbose:
            print(f"    ⚠️ Imagen 3 failed: {e}.")

    # 2. Try Pollinations.ai (free, unlimited image generation fallback)
    try:
        if verbose:
            print("    🎨 Trying Pollinations.ai (free keyless fallback)...")
        import urllib.parse
        import requests
        import random
        import time
        
        encoded_prompt = urllib.parse.quote(prompt)
        
        for attempt in range(4):
            seed = random.randint(1, 99999999)
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={VIDEO_WIDTH}&height={VIDEO_HEIGHT}&nologo=true&private=true&seed={seed}"
            response = requests.get(url, timeout=35)
            
            if response.status_code == 200:
                return response.content
            elif response.status_code == 429:
                wait_time = (attempt + 1) * 8
                if verbose:
                    print(f"      ⚠️ Pollinations.ai returned 429 (rate limited). Retrying in {wait_time}s... (attempt {attempt + 1}/4)")
                time.sleep(wait_time)
            else:
                if verbose:
                    print(f"      ⚠️ Pollinations.ai returned status code {response.status_code}")
                break
    except Exception as pe:
        if verbose:
            print(f"    ⚠️ Pollinations.ai fallback failed: {pe}")

    # 3. Fallback: gemini-2.5-flash with IMAGE response modality
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    for part in response.candidates[0].content.parts:
        if hasattr(part, 'inline_data') and part.inline_data:
            image_data = part.inline_data.data
            if isinstance(image_data, str):
                return base64.b64decode(image_data)
            return image_data

    raise ValueError("No image data found in model response.")


def generate_scenes(
    scenes: list[dict],
    style: str,
    output_dir: Path,
    reference_images: Optional[list[str]] = None,
    verbose: bool = True,
) -> list[Path]:
    """
    Generate whiteboard/chalkboard illustration for each scene.

    Args:
        scenes: List of scene dicts with 'index' and 'description' keys.
        style: Style preset key ("color_whiteboard" or "chalkboard").
        output_dir: Directory to save generated images.
        reference_images: Optional list of paths to style reference images.
        verbose: Print progress.

    Returns:
        List of paths to generated scene images.
    """
    from google import genai
    from google.genai import types

    if not USE_VERTEX and not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set. Add it to your .env file.")

    preset = STYLE_PRESETS.get(style)
    if not preset:
        raise ValueError(f"Unknown style: {style}. Available: {list(STYLE_PRESETS.keys())}")

    client = get_genai_client()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = []

    for scene in scenes:
        idx = scene["index"]
        description = scene["description"]

        if verbose:
            print(f"  🎨 Generating scene {idx}/{len(scenes)}: {description[:60]}...")

        image_path = output_dir / f"scene_{idx:02d}.png"

        # Build the prompt with style prefix
        prompt = (
            f"{preset['image_prompt_prefix']}. "
            f"Subject: {description}. "
            f"Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT}, landscape orientation. "
            f"The illustration should clearly communicate the concept. "
            f"No text or words in the image unless specifically relevant."
        )

        try:
            image_bytes = _generate_single_image(client, prompt, types, verbose=verbose)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
        except Exception as e:
            print(f"  ⚠️ Image generation failed for scene {idx}: {e}")
            _create_placeholder(image_path, description, preset)

        generated_paths.append(image_path)

        # Rate limiting — be gentle with the API
        if idx < len(scenes):
            time.sleep(3)

    if verbose:
        print(f"✅ Generated {len(generated_paths)} scene images")

    return generated_paths


def _create_placeholder(path: Path, text: str, preset: dict):
    """Create a placeholder image with text when generation fails."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        bg_color = preset.get("bg_color", "#FFFFFF")
        text_color = preset.get("text_color", "#1A1A1A")

        img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), bg_color)
        draw = ImageDraw.Draw(img)

        # Try to use a nice font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        except OSError:
            font = ImageFont.load_default()

        # Word wrap the text
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

        # Center the text block
        total_height = len(lines) * 50
        y_start = (VIDEO_HEIGHT - total_height) // 2

        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            x = (VIDEO_WIDTH - (bbox[2] - bbox[0])) // 2
            draw.text((x, y_start + i * 50), line, fill=text_color, font=font)

        # Add a border and "PLACEHOLDER" watermark
        draw.rectangle(
            [(20, 20), (VIDEO_WIDTH - 20, VIDEO_HEIGHT - 20)],
            outline=text_color,
            width=3,
        )
        try:
            small_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
            )
        except OSError:
            small_font = font
        draw.text(
            (VIDEO_WIDTH - 200, VIDEO_HEIGHT - 50),
            "PLACEHOLDER",
            fill="#999999",
            font=small_font,
        )

        img.save(str(path))

    except ImportError:
        # If PIL isn't available, create a minimal 1x1 PNG
        # This won't look great but keeps the pipeline from crashing
        import struct
        import zlib

        # Minimal valid PNG
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

        r, g, b = 255, 255, 255  # white background
        png_data = create_minimal_png(VIDEO_WIDTH, VIDEO_HEIGHT, (r, g, b))
        with open(path, 'wb') as f:
            f.write(png_data)


def generate_thumbnail(
    topic: str,
    niche: str,
    style: str,
    output_dir: Path,
    count: int = 3,
    verbose: bool = True,
) -> list[Path]:
    """
    Generate thumbnail variants for the video.

    Args:
        topic: Video topic.
        niche: Content niche.
        style: Visual style key.
        output_dir: Directory to save thumbnails.
        count: Number of variants to generate.
        verbose: Print progress.

    Returns:
        List of paths to generated thumbnails.
    """
    from google import genai
    from google.genai import types

    if not USE_VERTEX and not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set.")

    client = get_genai_client()
    output_dir.mkdir(parents=True, exist_ok=True)

    thumbnails = []

    thumbnail_prompts = [
        # Variant 1: Bold text + illustration
        (
            f"YouTube thumbnail for '{topic}', 1280x720, high contrast, "
            f"bold large text overlay, dramatic lighting, eye-catching, "
            f"educational content about {niche}, vibrant colors, "
            f"clean composition with a clear focal point, "
            f"professional YouTube thumbnail design"
        ),
        # Variant 2: Curiosity-driven
        (
            f"YouTube thumbnail for '{topic}', 1280x720, "
            f"mysterious or surprising visual metaphor related to {niche}, "
            f"dramatic lighting, curiosity-inducing, high contrast, "
            f"clean and bold, professional YouTube thumbnail"
        ),
        # Variant 3: Minimalist + bold
        (
            f"YouTube thumbnail for '{topic}', 1280x720, "
            f"minimalist design with one bold visual element, "
            f"high contrast between subject and background, "
            f"clean typography space, professional YouTube thumbnail "
            f"for {niche} content"
        ),
    ]

    for i in range(min(count, len(thumbnail_prompts))):
        if verbose:
            print(f"  🖼️ Generating thumbnail variant {i + 1}/{count}...")

        topic_slug = slugify(topic)
        thumb_path = output_dir / f"{topic_slug}_thumbnail_{i + 1:02d}.png"

        try:
            image_bytes = _generate_single_image(client, thumbnail_prompts[i], types, verbose=verbose)
            with open(thumb_path, "wb") as f:
                f.write(image_bytes)
        except Exception as e:
            print(f"  ⚠️ Thumbnail generation failed for variant {i + 1}: {e}")
            continue

        thumbnails.append(thumb_path)
        time.sleep(2)

    if verbose:
        print(f"✅ Generated {len(thumbnails)} thumbnail variants")

    return thumbnails
