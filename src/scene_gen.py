"""Scene image generation module — Gemini Flash Image for whiteboard/chalkboard illustrations."""

import time
import base64
from pathlib import Path
from typing import Optional

from .config import (
    GOOGLE_API_KEY,
    STYLE_PRESETS,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    slugify,
    get_genai_client,
    USE_VERTEX,
    IMAGEN_MODEL,
    GCP_PROJECT,
    GCP_LOCATION,
)


# Module-level cache for discovered models to avoid listing on every single image generation call
_discovered_vertex_models = None
_discovered_aistudio_models = None


def _get_available_imagen_models(client, is_vertex: bool, verbose: bool = True) -> list[str]:
    """Helper to query and cache available Imagen models for a specific client type."""
    global _discovered_vertex_models, _discovered_aistudio_models
    
    if is_vertex:
        if _discovered_vertex_models is not None:
            return _discovered_vertex_models
        _discovered_vertex_models = []
        cache = _discovered_vertex_models
        label = "Vertex AI"
    else:
        if _discovered_aistudio_models is not None:
            return _discovered_aistudio_models
        _discovered_aistudio_models = []
        cache = _discovered_aistudio_models
        label = "AI Studio"

    try:
        if verbose:
            print(f"    🔍 Querying available Imagen models from {label}...")
        models = client.models.list()
        for model in models:
            name = model.name
            if "imagen" in name.lower() or name.lower().endswith("-image") or name.lower().endswith("-image-preview"):
                if name not in cache:
                    cache.append(name)
                short_name = name.split("/")[-1]
                if short_name not in cache:
                    cache.append(short_name)
    except Exception as e:
        if verbose:
            print(f"    ⚠️ Could not query models dynamically from {label}: {e}")
            
    return cache


def _generate_single_image(client, prompt: str, types, verbose: bool = True) -> bytes:
    """Helper to generate an image using Imagen 3 with fallbacks (AI Studio / Vertex)."""
    # Check if the primary client is Vertex AI
    primary_is_vertex = getattr(client, '_vertexai', False) or (hasattr(client, 'config') and getattr(client.config, 'vertexai', False))

    # Initialize candidate models list
    model_candidates = []

    # 1. Start with the configured model
    if IMAGEN_MODEL:
        model_candidates.append(IMAGEN_MODEL)

    # 2. Add dynamically queried models for the primary client
    discovered_primary = _get_available_imagen_models(client, primary_is_vertex, verbose=verbose)
    for model_name in discovered_primary:
        # For Vertex, try to keep full path if discovered
        if primary_is_vertex and not model_name.startswith("publishers/"):
            fp = f"publishers/google/models/{model_name}"
            if fp not in model_candidates:
                model_candidates.append(fp)
        if model_name not in model_candidates:
            model_candidates.append(model_name)

    # 3. Add standard fallback models to ensure we try all potential options
    standard_fallbacks = [
        "gemini-3.1-flash-image",
        "gemini-3-pro-image",
        "gemini-3.1-flash-lite-image",
        "gemini-2.5-flash-image",
        "imagen-4.0-generate-001",
        "imagen-4.0-ultra-generate-001",
        "imagen-4.0-fast-generate-001",
        "imagen-3.0-generate-002",
        "imagen-3.0-generate-001",
        "imagen-3.0-fast-generate-001",
        "imagen-3.0-capability-001",
        "imagen-2.0-generate-002",
    ]
    for fb in standard_fallbacks:
        if fb not in model_candidates:
            model_candidates.append(fb)
        if primary_is_vertex:
            full_path = f"publishers/google/models/{fb}"
            if full_path not in model_candidates:
                model_candidates.append(full_path)

    # Remove duplicates while preserving order
    seen = set()
    model_candidates = [x for x in model_candidates if not (x in seen or seen.add(x))]

    # 4. Try Imagen models using the primary client
    for model_name in model_candidates:
        current_model = model_name
        if not primary_is_vertex and "publishers/" in current_model:
            current_model = current_model.split("/")[-1]
        try:
            if verbose:
                print(f"    🎨 Trying Imagen model {current_model} on primary client...")
            result = client.models.generate_images(
                model=current_model,
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
                err_msg = str(e)
                if "404" in err_msg or "not found" in err_msg.lower() or "access" in err_msg.lower():
                    print(f"    ⚠️ Imagen 3 ({current_model}) not available on primary client: {err_msg}.")
                else:
                    print(f"    ⚠️ Imagen 3 ({current_model}) failed on primary client: {e}.")

    # 5. Try alternative client fallback (if Vertex failed and AI Studio credentials exist, or vice versa)
    alt_client = None
    alt_is_vertex = not primary_is_vertex
    try:
        from google import genai
        
        if primary_is_vertex and GOOGLE_API_KEY:
            if verbose:
                print("    🎨 Vertex AI failed. Trying Google AI Studio (API key fallback)...")
            alt_client = genai.Client(api_key=GOOGLE_API_KEY)
        elif not primary_is_vertex and GCP_PROJECT:
            if verbose:
                print("    🎨 AI Studio failed. Trying Vertex AI fallback...")
            alt_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    except Exception as ce:
        if verbose:
            print(f"    ⚠️ Alternative client initialization failed: {ce}")

    if alt_client:
        # Build candidates for the alternative client (querying dynamically as well)
        alt_model_candidates = []
        if IMAGEN_MODEL:
            alt_model_candidates.append(IMAGEN_MODEL)
            
        discovered_alt = _get_available_imagen_models(alt_client, alt_is_vertex, verbose=verbose)
        for model_name in discovered_alt:
            if alt_is_vertex and not model_name.startswith("publishers/"):
                fp = f"publishers/google/models/{model_name}"
                if fp not in alt_model_candidates:
                    alt_model_candidates.append(fp)
            if model_name not in alt_model_candidates:
                alt_model_candidates.append(model_name)
                
        for fb in standard_fallbacks:
            if fb not in alt_model_candidates:
                alt_model_candidates.append(fb)
            if alt_is_vertex:
                full_path = f"publishers/google/models/{fb}"
                if full_path not in alt_model_candidates:
                    alt_model_candidates.append(full_path)
                    
        # Remove duplicates while preserving order
        seen_alt = set()
        alt_model_candidates = [x for x in alt_model_candidates if not (x in seen_alt or seen_alt.add(x))]

        for model_name in alt_model_candidates:
            current_model = model_name
            if not alt_is_vertex and "publishers/" in current_model:
                current_model = current_model.split("/")[-1]
            try:
                if verbose:
                    print(f"      🎨 Trying Imagen model {current_model} on alternative client...")
                result = alt_client.models.generate_images(
                    model=current_model,
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
                    print(f"      ⚠️ Alternative client Imagen ({current_model}) failed: {e}")

    raise ValueError("No image generation models succeeded.")


def generate_scenes(
    scenes: list[dict],
    style: str,
    output_dir: Path,
    reference_images: Optional[list[str]] = None,
    force: bool = False,
    force_scenes: Optional[list[int]] = None,
    verbose: bool = True,
) -> list[Path]:
    """
    Generate whiteboard/chalkboard illustration for each scene.

    Args:
        scenes: List of scene dicts with 'index' and 'description' keys.
        style: Style preset key ("color_whiteboard" or "chalkboard").
        output_dir: Directory to save generated images.
        reference_images: Optional list of paths to style reference images.
        force: Force generate all scene images.
        force_scenes: Specific scene indices to force generate.
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
        image_path = output_dir / f"scene_{idx:02d}.png"

        # Check if we should skip this specific image
        if image_path.exists() and not force:
            if not force_scenes or idx not in force_scenes:
                if verbose:
                    print(f"  ℹ️ Scene {idx} image already exists. Skipping.")
                generated_paths.append(image_path)
                continue

        if verbose:
            print(f"  🎨 Generating scene {idx}/{len(scenes)}: {description[:60]}...")

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
