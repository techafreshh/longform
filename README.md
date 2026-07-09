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

## Quick Start (Google Colab)

1. Open `Longform_Video_Factory.ipynb` in Google Colab
2. Run Cell 0 to install dependencies and mount Drive
3. Set your API keys in the config cell
4. Run cells sequentially — the notebook pauses for your review at two points

## Setup

### API Keys Needed

| Key | Required | Get from |
|-----|----------|----------|
| `GOOGLE_API_KEY` | ✅ | [Google AI Studio](https://aistudio.google.com/apikey) |
| `OPENROUTER_API_KEY` | ✅ | [OpenRouter](https://openrouter.ai/keys) |
| `FISH_API_KEY` | ✅ | [Fish Audio](https://fish.audio/app/api-keys/) |
| `FISH_VOICE_ID` | ✅ | Created after voice cloning |
| `PEXELS_API_KEY` | Optional | [Pexels](https://www.pexels.com/api/) |
| `GOOGLE_SHEET_ID` | Optional | Your Google Sheet URL |

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
