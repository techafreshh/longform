"""YouTube search and transcript downloader module — fetches reference scripts for LLM writing."""

import json
import re
from pathlib import Path
from typing import Optional
import requests

from .config import slugify


def search_youtube_videos(query: str, count: int = 3) -> list[dict]:
    """
    Search YouTube for a query and return the video IDs and titles of the top results.
    Does not require a YouTube API key.
    
    Args:
        query: The search query.
        count: Number of results to return.
        
    Returns:
        List of dicts with 'video_id' and 'title'.
    """
    url = f"https://www.youtube.com/results?search_query={requests.utils.quote(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()

        # Look for the ytInitialData JSON
        json_match = re.search(r'var ytInitialData\s*=\s*({.*?});', r.text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'window\["ytInitialData"\]\s*=\s*({.*?});', r.text, re.DOTALL)

        if json_match:
            data = json.loads(json_match.group(1))
            videos = []

            # Helper to recursively traverse JSON to find videoRenderer objects
            def find_video_renderers(obj):
                results = []
                if isinstance(obj, dict):
                    if "videoRenderer" in obj:
                        results.append(obj["videoRenderer"])
                    else:
                        for k, v in obj.items():
                            results.extend(find_video_renderers(v))
                elif isinstance(obj, list):
                    for item in obj:
                        results.extend(find_video_renderers(item))
                return results

            renderers = find_video_renderers(data)
            for rdr in renderers:
                video_id = rdr.get("videoId")
                title = ""
                title_obj = rdr.get("title", {})
                if "runs" in title_obj and title_obj["runs"]:
                    title = title_obj["runs"][0].get("text", "")
                elif "simpleText" in title_obj:
                    title = title_obj["simpleText"]

                if video_id and title:
                    videos.append({
                        "video_id": video_id,
                        "title": title
                    })
                    if len(videos) >= count:
                        break
            return videos

    except Exception as e:
        print(f"⚠️ Failed to parse YouTube search JSON: {e}")

    # Fallback to simple regex URL scraping if JSON parsing fails
    try:
        r = requests.get(url, headers=headers)
        video_ids = re.findall(r'/watch\?v=([a-zA-Z0-9_-]{11})', r.text)
        unique_ids = []
        for vid in video_ids:
            if vid not in unique_ids:
                unique_ids.append(vid)
        return [{"video_id": vid, "title": f"YouTube Video {vid}"} for vid in unique_ids[:count]]
    except Exception as fallback_err:
        print(f"⚠️ Fallback YouTube search failed: {fallback_err}")

    return []


def _fetch_raw_transcript(video_id: str) -> Optional[list[dict]]:
    """Fetch raw transcript list using any available API method across all youtube_transcript_api versions."""
    try:
        import youtube_transcript_api
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return None

    # 1. Try class-level list_transcripts
    if hasattr(YouTubeTranscriptApi, "list_transcripts") and callable(getattr(YouTubeTranscriptApi, "list_transcripts")):
        try:
            tl = YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                t = tl.find_transcript(['en', 'en-US', 'en-GB'])
            except Exception:
                try:
                    for item in tl:
                        if getattr(item, "is_translatable", False):
                            t = item.translate('en')
                            break
                    else:
                        t = next(iter(tl))
                except Exception:
                    t = None
            if t is not None:
                res = t.fetch()
                if isinstance(res, list) and len(res) > 0:
                    return res
        except Exception:
            pass

    # 2. Try instance-level list_transcripts
    try:
        api = YouTubeTranscriptApi()
        if hasattr(api, "list_transcripts") and callable(getattr(api, "list_transcripts")):
            tl = api.list_transcripts(video_id)
            try:
                t = tl.find_transcript(['en', 'en-US', 'en-GB'])
            except Exception:
                try:
                    for item in tl:
                        if getattr(item, "is_translatable", False):
                            t = item.translate('en')
                            break
                    else:
                        t = next(iter(tl))
                except Exception:
                    t = None
            if t is not None:
                res = t.fetch()
                if isinstance(res, list) and len(res) > 0:
                    return res
    except Exception:
        pass

    # 3. Try class-level get_transcript
    if hasattr(YouTubeTranscriptApi, "get_transcript") and callable(getattr(YouTubeTranscriptApi, "get_transcript")):
        try:
            res = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
            if isinstance(res, list) and len(res) > 0:
                return res
        except Exception:
            pass
        try:
            res = YouTubeTranscriptApi.get_transcript(video_id)
            if isinstance(res, list) and len(res) > 0:
                return res
        except Exception:
            pass

    # 4. Try instance-level get_transcript
    try:
        api = YouTubeTranscriptApi()
        if hasattr(api, "get_transcript") and callable(getattr(api, "get_transcript")):
            try:
                res = api.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
                if isinstance(res, list) and len(res) > 0:
                    return res
            except Exception:
                pass
            try:
                res = api.get_transcript(video_id)
                if isinstance(res, list) and len(res) > 0:
                    return res
            except Exception:
                pass
    except Exception:
        pass

    # 5. Try module-level functions on youtube_transcript_api
    if hasattr(youtube_transcript_api, "get_transcript") and callable(getattr(youtube_transcript_api, "get_transcript")):
        try:
            res = youtube_transcript_api.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
            if isinstance(res, list) and len(res) > 0:
                return res
        except Exception:
            pass
        try:
            res = youtube_transcript_api.get_transcript(video_id)
            if isinstance(res, list) and len(res) > 0:
                return res
        except Exception:
            pass

    # 6. Fallback: dynamically inspect all callables containing 'transcript'
    for target in [YouTubeTranscriptApi, getattr(youtube_transcript_api, 'YouTubeTranscriptApi', None), youtube_transcript_api]:
        if target is None:
            continue
        for attr_name in dir(target):
            if "transcript" in attr_name.lower() and not attr_name.startswith("_"):
                try:
                    method = getattr(target, attr_name)
                    if callable(method):
                        res = method(video_id)
                        if isinstance(res, list) and len(res) > 0:
                            return res
                except Exception:
                    pass

    return None


def download_youtube_transcript(
    video_id: str,
    video_title: str,
    output_dir: Path,
    verbose: bool = True
) -> Optional[Path]:
    """
    Download the English transcript of a YouTube video and save it as text.
    
    Args:
        video_id: YouTube video ID.
        video_title: Video title (used for filename).
        output_dir: Path to save the text file.
        verbose: Print progress.
        
    Returns:
        Path to the saved file or None if it failed.
    """
    try:
        import youtube_transcript_api
    except ImportError:
        if verbose:
            print("⚠️ youtube-transcript-api is not installed. Run 'pip install youtube-transcript-api'.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Safe slug for file name
    clean_title = re.sub(r'[^\w\s-]', '', video_title.lower()).strip()
    clean_title = re.sub(r'[-\s_]+', '-', clean_title)
    filename = f"{clean_title[:50]}_{video_id}.txt"
    dest_path = output_dir / filename

    if verbose:
        print(f"  📥 Fetching transcript for: '{video_title}' ({video_id})...")

    try:
        data = _fetch_raw_transcript(video_id)
        if not data:
            if verbose:
                print(f"  ⚠️ No transcript available or readable for video ID {video_id}.")
            return None

        # Merge short timeline segments into natural readable prose paragraphs
        lines = []
        current_paragraph = []
        for entry in data:
            text = entry.get("text", "").strip()
            # Clean up raw HTML subtitle encoding
            text = (
                text.replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&#39;", "'")
                .replace("&quot;", '"')
            )
            if not text:
                continue

            current_paragraph.append(text)
            if text.endswith(('.', '!', '?')) or len(current_paragraph) > 12:
                lines.append(" ".join(current_paragraph))
                current_paragraph = []

        if current_paragraph:
            lines.append(" ".join(current_paragraph))

        merged_text = "\n\n".join(lines)
        dest_path.write_text(merged_text, encoding="utf-8")

        if verbose:
            print(f"  ✅ Saved reference script to: {dest_path.name}")
        return dest_path

    except Exception as e:
        if verbose:
            print(f"  ⚠️ Could not download transcript for {video_id}: {e}")
        return None
