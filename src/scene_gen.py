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


def _get_model_cost(model_name: str) -> float:
    """Return estimated cost per image generation for a given model."""
    name_lower = model_name.lower()
    if "ultra" in name_lower:
        return 0.06
    elif "fast" in name_lower or "lite" in name_lower:
        return 0.02
    elif "pro" in name_lower:
        return 0.06
    elif "gemini-3.1-flash-image" in name_lower or "gemini-2.5-flash-image" in name_lower:
        return 0.03
    elif "imagen-3.0-generate" in name_lower:
        return 0.04
    elif "imagen-2.0" in name_lower:
        return 0.02
    # Default fallbacks
    if "gemini" in name_lower:
        return 0.03
    if "imagen" in name_lower:
        return 0.04
    return 0.00


def _is_gemini_model(model_name: str) -> bool:
    """Check if a model name refers to a Gemini model (uses generate_content) vs Imagen (uses generate_images)."""
    base = model_name.split("/")[-1].lower()
    return base.startswith("gemini")


def _get_available_imagen_models(client, is_vertex: bool, verbose: bool = True) -> dict[str, list[str]]:
    """Helper to query and cache available image generation models.
    
    Returns a dict with two keys:
        'gemini': list of Gemini model names (use generate_content)
        'imagen': list of Imagen model names (use generate_images)
    """
    global _discovered_vertex_models, _discovered_aistudio_models
    
    cache_key = "_discovered_vertex_models" if is_vertex else "_discovered_aistudio_models"
    existing = _discovered_vertex_models if is_vertex else _discovered_aistudio_models
    
    if existing is not None:
        return existing
    
    result = {"gemini": [], "imagen": []}
    label = "Vertex AI" if is_vertex else "AI Studio"

    try:
        if verbose:
            print(f"    🔍 Querying available image generation models from {label}...")
        models = client.models.list()
        for model in models:
            name = model.name
            name_lower = name.lower()
            # Match image-capable models
            is_image_model = (
                "imagen" in name_lower
                or name_lower.endswith("-image")
                or name_lower.endswith("-image-preview")
            )
            if not is_image_model:
                continue
                
            short_name = name.split("/")[-1]
            bucket = "gemini" if _is_gemini_model(name) else "imagen"
            
            if short_name not in result[bucket]:
                result[bucket].append(short_name)
    except Exception as e:
        if verbose:
            print(f"    ⚠️ Could not query models dynamically from {label}: {e}")
    
    # Cache it
    if is_vertex:
        _discovered_vertex_models = result
    else:
        _discovered_aistudio_models = result
            
    return result


def _try_gemini_image_generation(client, model_name: str, prompt: str, types, verbose: bool = True) -> Optional[bytes]:
    """Try generating an image using Gemini's generate_content with IMAGE response modality."""
    max_retries = 10
    base_delay = 5  # seconds
    for attempt in range(max_retries):
        try:
            if verbose:
                if attempt > 0:
                    print(f"    🎨 Trying Gemini model {model_name} (generate_content) - Attempt {attempt+1}/{max_retries}...")
                else:
                    print(f"    🎨 Trying Gemini model {model_name} (generate_content)...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio="16:9",
                    ),
                )
            )
            # Extract image bytes from the response
            if response and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        return part.inline_data.data
            break  # If we got response but no image data found, do not retry
        except Exception as e:
            err_msg = str(e)
            is_429 = "429" in err_msg or "resource_exhausted" in err_msg.lower() or "resource exhausted" in err_msg.lower()
            if is_429 and attempt < max_retries - 1:
                sleep_time = base_delay * (2 ** attempt)
                if verbose:
                    print(f"    ⚠️ Gemini ({model_name}) resource exhausted: {err_msg[:120]}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            
            if verbose:
                if "404" in err_msg or "not found" in err_msg.lower():
                    print(f"    ⚠️ Gemini ({model_name}) not available: {err_msg[:200]}.")
                else:
                    print(f"    ⚠️ Gemini ({model_name}) failed: {err_msg[:200]}.")
            break
    return None


def _try_imagen_generation(client, model_name: str, prompt: str, types, verbose: bool = True) -> Optional[bytes]:
    """Try generating an image using Imagen's generate_images endpoint."""
    max_retries = 10
    base_delay = 5  # seconds
    for attempt in range(max_retries):
        try:
            if verbose:
                if attempt > 0:
                    print(f"    🎨 Trying Imagen model {model_name} (generate_images) - Attempt {attempt+1}/{max_retries}...")
                else:
                    print(f"    🎨 Trying Imagen model {model_name} (generate_images)...")
            result = client.models.generate_images(
                model=model_name,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="16:9",
                    output_mime_type="image/png",
                )
            )
            if result and result.generated_images:
                return result.generated_images[0].image.image_bytes
            break
        except Exception as e:
            err_msg = str(e)
            is_429 = "429" in err_msg or "resource_exhausted" in err_msg.lower() or "resource exhausted" in err_msg.lower()
            if is_429 and attempt < max_retries - 1:
                sleep_time = base_delay * (2 ** attempt)
                if verbose:
                    print(f"    ⚠️ Imagen ({model_name}) resource exhausted: {err_msg[:120]}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                continue

            if verbose:
                if "404" in err_msg or "not found" in err_msg.lower():
                    print(f"    ⚠️ Imagen ({model_name}) not available: {err_msg[:200]}.")
                else:
                    print(f"    ⚠️ Imagen ({model_name}) failed: {err_msg[:200]}.")
            break
    return None


def _generate_single_image(client, prompt: str, types, verbose: bool = True) -> tuple[bytes, str, str]:
    """Generate an image with smart fallbacks across Gemini and Imagen models.
    
    Strategy:
        1. Try Gemini models via generate_content (most widely available)
        2. Try Imagen models via generate_images (requires specific API access)
        3. Fall back to alternative client (Vertex ↔ AI Studio) and repeat
    
    Returns (image_bytes, model_name, client_type).
    """
    # Check if the primary client is Vertex AI
    primary_is_vertex = getattr(client, '_vertexai', False) or (hasattr(client, 'config') and getattr(client.config, 'vertexai', False))

    # --- Build ordered model lists ---
    # Gemini models: use generate_content with response_modalities=["IMAGE"]
    # These are the confirmed-available image generation models on the project.
    gemini_fallbacks = [
        "gemini-2.5-flash-image",          # Older but reliable / confirmed working
        "gemini-3.1-flash-image",          # Latest stable flash image gen
        "gemini-3-pro-image",              # Pro quality image gen
        "gemini-3.1-flash-lite-image",     # Fast/cheap option
        "gemini-3.1-flash-image-preview",  # Preview variant
        "gemini-3-pro-image-preview",      # Preview variant
    ]
    
    # Imagen models: use generate_images endpoint
    # NOTE: No imagen-* models are currently available on this project.
    # Kept as fallback in case they are enabled in the future.
    imagen_fallbacks = []

    # If the user configured a specific model, put it first in the right list
    if IMAGEN_MODEL:
        if _is_gemini_model(IMAGEN_MODEL):
            if IMAGEN_MODEL not in gemini_fallbacks:
                gemini_fallbacks.insert(0, IMAGEN_MODEL)
        else:
            if IMAGEN_MODEL not in imagen_fallbacks:
                imagen_fallbacks.insert(0, IMAGEN_MODEL)

    # Add dynamically discovered models 
    discovered = _get_available_imagen_models(client, primary_is_vertex, verbose=verbose)
    for m in discovered.get("gemini", []):
        if m not in gemini_fallbacks:
            gemini_fallbacks.append(m)
    for m in discovered.get("imagen", []):
        if m not in imagen_fallbacks:
            imagen_fallbacks.append(m)

    client_label = "Vertex AI" if primary_is_vertex else "AI Studio"

    # --- Phase 1: Try Gemini models on primary client ---
    for model_name in gemini_fallbacks:
        image_bytes = _try_gemini_image_generation(client, model_name, prompt, types, verbose=verbose)
        if image_bytes:
            return image_bytes, model_name, client_label

    # --- Phase 2: Try Imagen models on primary client ---
    for model_name in imagen_fallbacks:
        image_bytes = _try_imagen_generation(client, model_name, prompt, types, verbose=verbose)
        if image_bytes:
            return image_bytes, model_name, client_label

    # --- Phase 3: Try alternative client fallback ---
    alt_client = None
    alt_is_vertex = not primary_is_vertex
    try:
        from google import genai
        
        if primary_is_vertex and GOOGLE_API_KEY:
            if verbose:
                print("    🔄 Primary client exhausted. Trying Google AI Studio fallback...")
            alt_client = genai.Client(api_key=GOOGLE_API_KEY)
        elif not primary_is_vertex and GCP_PROJECT:
            if verbose:
                print("    🔄 Primary client exhausted. Trying Vertex AI fallback...")
            alt_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    except Exception as ce:
        if verbose:
            print(f"    ⚠️ Alternative client initialization failed: {ce}")

    if alt_client:
        alt_label = "Vertex AI" if alt_is_vertex else "AI Studio"
        
        # Discover models on alt client too
        alt_discovered = _get_available_imagen_models(alt_client, alt_is_vertex, verbose=verbose)
        
        alt_gemini = list(gemini_fallbacks)
        for m in alt_discovered.get("gemini", []):
            if m not in alt_gemini:
                alt_gemini.append(m)
                
        alt_imagen = list(imagen_fallbacks)
        for m in alt_discovered.get("imagen", []):
            if m not in alt_imagen:
                alt_imagen.append(m)

        # Try Gemini on alt client
        for model_name in alt_gemini:
            image_bytes = _try_gemini_image_generation(alt_client, model_name, prompt, types, verbose=verbose)
            if image_bytes:
                return image_bytes, model_name, alt_label

        # Try Imagen on alt client
        for model_name in alt_imagen:
            image_bytes = _try_imagen_generation(alt_client, model_name, prompt, types, verbose=verbose)
            if image_bytes:
                return image_bytes, model_name, alt_label

    raise ValueError("No image generation models succeeded. Check your API access and GCP project permissions.")


def generate_scenes(
    scenes: list[dict],
    style: str,
    output_dir: Path,
    reference_images: Optional[list[str]] = None,
    force: bool = False,
    force_scenes: Optional[list[int]] = None,
    gdrive_folder_id: Optional[str] = None,
    resume_from_scene: Optional[int] = None,
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
        gdrive_folder_id: Optional Google Drive folder ID to upload images immediately.
        resume_from_scene: Optional scene index to resume generation from (skips smaller indices).
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

    # Check if a custom style analysis exists from reference clips
    import json
    reproduction_prefix = None
    analysis_file = output_dir.parent / "reference_style_analysis.json"
    if analysis_file.exists():
        try:
            with open(analysis_file, "r", encoding="utf-8") as f:
                analysis_data = json.load(f)
                reproduction_prefix = analysis_data.get("reproduction_prompt_prefix")
                if verbose and reproduction_prefix:
                    print(f"🎨 Found custom style reproduction prompt prefix from reference clips:")
                    print(f"   '{reproduction_prefix}'")
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to read reference style analysis for scene gen: {e}")

    generated_paths = []
    report_records = []

    for scene in scenes:
        idx = scene["index"]
        description = scene["description"]
        image_path = output_dir / f"scene_{idx:02d}.png"

        # Check if we should skip due to resume index
        if resume_from_scene is not None and idx < resume_from_scene:
            if verbose:
                print(f"  ℹ️ Scene {idx} is before resume index {resume_from_scene}. Skipping.")
            generated_paths.append(image_path)
            report_records.append({
                "scene_index": idx,
                "description": description,
                "status": "Skipped (Before Resume)",
                "model": "N/A",
                "client": "N/A",
                "cost": 0.00
            })
            continue

        # Check if we should skip this specific image
        if image_path.exists() and not force:
            if not force_scenes or idx not in force_scenes:
                if verbose:
                    print(f"  ℹ️ Scene {idx} image already exists. Skipping.")
                generated_paths.append(image_path)
                report_records.append({
                    "scene_index": idx,
                    "description": description,
                    "status": "Skipped (Exists)",
                    "model": "N/A",
                    "client": "N/A",
                    "cost": 0.00
                })
                continue

        if verbose:
            print(f"  🎨 Generating scene {idx}/{len(scenes)}: {description[:60]}...")

        # Build the prompt with style prefix
        style_prefix = reproduction_prefix if reproduction_prefix else preset['image_prompt_prefix']
        prompt = (
            f"{style_prefix}. "
            f"Subject: {description}. "
            f"Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT}, landscape orientation. "
            f"The illustration should clearly communicate the concept. "
            f"No text or words in the image unless specifically relevant."
        )

        try:
            image_bytes, model_name, client_type = _generate_single_image(client, prompt, types, verbose=verbose)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            # Upload immediately if gdrive is configured
            if gdrive_folder_id:
                try:
                    from .gdrive import get_gdrive_service, upload_file_to_drive_folder
                    service = get_gdrive_service()
                    if service:
                        upload_file_to_drive_folder(service, gdrive_folder_id, image_path, "scenes")
                except Exception as upload_err:
                    print(f"  ⚠️ Failed to upload scene {idx} to Google Drive: {upload_err}")

            cost = _get_model_cost(model_name)
            report_records.append({
                "scene_index": idx,
                "description": description,
                "status": "Success",
                "model": model_name,
                "client": client_type,
                "cost": cost
            })
        except Exception as e:
            print(f"  ⚠️ Image generation failed for scene {idx}: {e}")
            _create_placeholder(image_path, description, preset)
            
            # Upload placeholder immediately if gdrive is configured
            if gdrive_folder_id:
                try:
                    from .gdrive import get_gdrive_service, upload_file_to_drive_folder
                    service = get_gdrive_service()
                    if service:
                        upload_file_to_drive_folder(service, gdrive_folder_id, image_path, "scenes")
                except Exception as upload_err:
                    print(f"  ⚠️ Failed to upload placeholder scene {idx} to Google Drive: {upload_err}")

            report_records.append({
                "scene_index": idx,
                "description": description,
                "status": "Failed (Placeholder)",
                "model": "Placeholder",
                "client": "Local",
                "cost": 0.00
            })

        generated_paths.append(image_path)

        # Rate limiting — be gentle with the API
        if idx < len(scenes):
            time.sleep(3)

    # Write report to CSV
    import csv
    report_path = output_dir / "image_generation_report.csv"
    try:
        with open(report_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["scene_index", "description", "status", "model", "client", "cost"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in report_records:
                writer.writerow(record)
        if verbose:
            print(f"\n📊 Image generation report saved to: {report_path}")
            total_cost = sum(r["cost"] for r in report_records)
            print(f"💰 Total Estimated Cost: ${total_cost:.4f}")
    except Exception as cre:
        print(f"⚠️ Could not write report CSV: {cre}")

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
        # Variant 1: Whiteboard Cartoon / AsapSCIENCE style
        (
            f"YouTube thumbnail in 16:9 aspect ratio, 1280x720. AsapSCIENCE style whiteboard cartoon illustration, clean solid white background. "
            f"In the center, a simple humorous 2D hand-drawn cartoon character or animal with bold black outlines and minimal vibrant flat coloring, looking curious or thinking. "
            f"A thought bubble with a question mark or simple icon. At the top, bold large yellow bubble text with a thick black outline showing a short curiosity-provoking question related to '{topic}'. "
            f"Simple, high-contrast, extremely readable composition."
        ),
        # Variant 2: Chalkboard style
        (
            f"YouTube thumbnail in 16:9 aspect ratio, 1280x720. Chalkboard style, dark forest green chalkboard background with subtle chalk dust texture. "
            f"Hand-drawn chalk sketches and diagrams in white and colored chalk illustrating a surprising concept from '{topic}'. "
            f"Bold yellow and white hand-drawn chalk text overlay at the top. "
            f"High contrast, educational and highly engaging visual composition."
        ),
        # Variant 3: Dramatic / Explainer style
        (
            f"YouTube thumbnail in 16:9 aspect ratio, 1280x720. Dramatic educational explainer style, high contrast, clean minimalist design with a single powerful visual metaphor for '{topic}' as the central focal point. "
            f"Bold, vibrant color palette, dark professional background. "
            f"Large, easy-to-read sans-serif yellow text overlay. "
            f"Clean, eye-catching, and curiosity-inducing."
        ),
    ]

    for i in range(min(count, len(thumbnail_prompts))):
        if verbose:
            print(f"  🖼️ Generating thumbnail variant {i + 1}/{count}...")

        topic_slug = slugify(topic)
        thumb_path = output_dir / f"{topic_slug}_thumbnail_{i + 1:02d}.png"

        try:
            image_bytes, _, _ = _generate_single_image(client, thumbnail_prompts[i], types, verbose=verbose)
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
