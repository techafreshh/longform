# 🎬 Longform Video Factory

Automated faceless whiteboard animation YouTube video pipeline. Goes from topic → published video with minimal manual intervention.

## Features

- **Google Sheets integration** — Manage your content calendar, pick topics, track status
- **Deep research** — Gemini Deep Research API for comprehensive topic analysis
- **AI scriptwriting** — Claude/Gemini via OpenRouter with scene markers
- **Voice cloning** — Fish Audio (primary) or Qwen3-TTS (Colab GPU fallback)
- **AI scene generation** — Gemini Flash Image for consistent whiteboard/chalkboard illustrations
- **Video assembly** — FFmpeg with Ken Burns effects, transitions, and subtitles
- **Thumbnail generation** — 3 AI-generated thumbnail variants
- **SEO optimization** — Auto-generated titles, descriptions, and tags
- **Variable visual styles** — Color whiteboard or chalkboard (per-video setting)
- **Two review gates** — Review script + review final video before publishing

## Resilient & Cloud-Native Execution (Google Drive & GPU-Accelerated Assembly)

The pipeline is built with maximum resilience to handle Colab VM session losses, rate limits, and cross-account handoffs:

- **GPU-Accelerated Video Rendering**: Auto-detects and utilizes the `h264_nvenc` encoder (e.g. on T4 GPU instances in Google Colab) to accelerate rendering. Gracefully falls back to `libx264` (CPU) if a GPU is unavailable.
- **Incremental Scene Caching**: Every scene is rendered to a persistent cache directory (`clips_cache/`) as an individual `.mp4` clip. If a crash or timeout occurs, the system checks size and modification times to skip already-rendered clips, ensuring assembly can resume instantly.
- **Targeted Scene/Clip Resumption**: If you need to force-skip generation and rendering up to a specific index, you can configure `RESUME_FROM_SCENE = <scene_index>` in the notebook configuration cell. The pipeline will treat all prior scenes as completed and resume generation and assembly starting from that index.
- **Granular Fetch-on-Demand (Google Drive)**: Replaces slow, full-folder downloads with bandwidth-efficient, granular syncing. Root files (`script.md`, `scenes.json`) are pulled on startup, while larger stage assets (`audio`, `scenes`, `clips_cache`) are synced incrementally *only* when the corresponding stage starts.
- **Immediate Cloud Backup & Synchronization**: To prevent losing work from volatile Colab disks, generated assets from every stage (research, script, voice segments, scene images, thumbnail variants, rendering clips, and final video/SRT files) are immediately uploaded to a shared Google Drive folder upon creation.
- **Multi-Account Resume Support**: By sharing a single project Google Drive folder link, you can swap between Google accounts (to bypass rate limits or VM daily quotas) and run the notebook on a different VM. The pipeline automatically recovers all completed progress from Google Drive and resumes exactly where the previous account left off.

## Quick Start (Google Colab)

1. Open `Longform_Video_Factory.ipynb` in Google Colab
2. Run Cell 0 to install dependencies and mount Drive
3. Set your API keys in the config cell
4. Run cells sequentially — the notebook pauses for your review at two points

## Setup

### API Keys & Configuration

| Key / Variable | Required | Description | Get from |
|---|---|---|---|
| `GOOGLE_API_KEY` | ✅ (unless using Vertex) | Google AI Studio Key | [Google AI Studio](https://aistudio.google.com/apikey) |
| `OPENROUTER_API_KEY` | ✅ | OpenRouter Key | [OpenRouter](https://openrouter.ai/keys) |
| `FISH_API_KEY` | ✅ | Fish Audio API Key | [Fish Audio](https://fish.audio/app/api-keys/) |
| `FISH_VOICE_ID` | ✅ | Created after voice cloning | Fish Audio Console |
| `PEXELS_API_KEY` | Optional | Pexels API Key | [Pexels](https://www.pexels.com/api/) |
| `GOOGLE_SHEET_ID` | Optional | Your Google Sheet ID | Google Sheet URL |
| `USE_VERTEX` | Optional | Set to `true` to use GCP Vertex AI | Toggle |
| `GCP_PROJECT` | Optional | Your GCP Project ID (for Vertex AI) | Google Cloud Console |
| `GCP_LOCATION` | Optional | GCP Region (defaults to `us-central1`) | Google Cloud Console |

### Vertex AI Integration (Using Google Cloud $300 Credits)

If you have a Google Cloud account with $300 welcome credits, you can use **Vertex AI** for image and research generation to avoid spending real money on AI Studio prepay accounts:

1. Enable the **Vertex AI API** in your Google Cloud Console.
2. In Google Colab Secrets (or manual configs), set:
   * `USE_VERTEX` to `true`
   * `GCP_PROJECT` to your GCP Project ID
   * `GCP_LOCATION` to `us-central1` (or your preferred region)
3. Run the Colab setup cell, approving the `auth.authenticate_user()` OAuth prompt. **Make sure to log in with the Google Account that holds the GCP project and credits.**

### Image Generation Fallbacks
If you are running the pipeline using Vertex AI (`USE_VERTEX=true`) and run into model availability or permissions errors, the pipeline will automatically attempt to fall back to Google AI Studio's Imagen models using the `GOOGLE_API_KEY`. If all generation backends fail, it automatically generates styled local placeholders using Python's Pillow library so the video rendering process can proceed uninterrupted.

### Google Sheet Setup

Create a Google Sheet with these columns:

| Topic | Niche | Style | Additional Prompt | Target Length | Status | Video URL | Drive Link |
|-------|-------|-------|-------------------|---------------|--------|-----------|------------|

Set `Status` to `ready` for topics you want to produce.

### Voice Cloning

1. Record a 5-15 second voice sample (clear, no background noise)
2. Run the voice cloning utility cell in the notebook
3. Save the returned `voice_id` to your config

## Project Structure

```
longform/
├── src/
│   ├── config.py         # API keys, style presets, prompts
│   ├── researcher.py     # Gemini Deep Research
│   ├── scriptwriter.py   # Script generation + scene parsing
│   ├── voice.py          # Fish Audio + Qwen3-TTS
│   ├── scene_gen.py      # Image generation + thumbnails
│   ├── assembler.py      # FFmpeg video assembly
│   ├── seo.py            # SEO metadata generation
│   ├── sheets.py         # Google Sheets integration
│   ├── stock.py          # Pexels B-roll (optional)
│   └── pipeline.py       # End-to-end orchestrator
├── Longform_Video_Factory.ipynb  # Main Colab notebook
├── pyproject.toml
├── .env.example
└── README.md
```

## Visual Styles

Set per-video in the Google Sheet `Style` column:

- **`color_whiteboard`** — Colorful marker on white background
- **`chalkboard`** — Chalk on dark green chalkboard

## Cost Per Video

| Component | Cost |
|-----------|------|
| Research (Gemini) | $0 (AI Pro sub) |
| Script (Claude) | ~$0.10 |
| Voice (Fish Audio free) | $0 |
| Images (Gemini) | $0 (AI Pro sub) |
| Assembly (FFmpeg) | $0 |
| **Total** | **~$0.10** |
