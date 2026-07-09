"""Google Sheets integration — read topics, update status, write results."""

import json
from pathlib import Path
from typing import Optional

from .config import GOOGLE_SHEET_ID, slugify


# Column mapping (1-indexed for gspread)
COLUMNS = {
    "topic": 1,
    "niche": 2,
    "style": 3,
    "additional_prompt": 4,
    "target_length": 5,
    "status": 6,
    "video_url": 7,
    "drive_link": 8,
    "thumbnail_link": 9,
    "seo_title": 10,
    "seo_description": 11,
    "seo_tags": 12,
}


def _get_client():
    """Get authenticated gspread client."""
    try:
        import gspread
        from google.auth import default
    except ImportError:
        raise ImportError("Install gspread: pip install gspread google-auth")

    try:
        # In Colab, use the default credentials from google.colab.auth
        creds, _ = default()
        return gspread.authorize(creds)
    except Exception:
        # Fallback: try service account
        try:
            return gspread.service_account()
        except Exception as e:
            raise RuntimeError(
                f"Could not authenticate with Google Sheets: {e}\n"
                "In Colab, run: from google.colab import auth; auth.authenticate_user()\n"
                "Locally, set up a service account key."
            )


def get_ready_topics(sheet_id: Optional[str] = None, verbose: bool = True) -> list[dict]:
    """
    Get all topics with status 'ready' from the Google Sheet.

    Returns:
        List of dicts with topic info.
    """
    sheet_id = sheet_id or GOOGLE_SHEET_ID
    if not sheet_id:
        raise ValueError("GOOGLE_SHEET_ID is not set.")

    client = _get_client()
    sheet = client.open_by_key(sheet_id).sheet1

    all_rows = sheet.get_all_records()

    ready = []
    for i, row in enumerate(all_rows):
        status = str(row.get("Status", "")).strip().lower()
        if status == "ready":
            ready.append({
                "row_number": i + 2,  # +2 for header row + 0-indexed
                "topic": str(row.get("Topic", "")).strip(),
                "niche": str(row.get("Niche", "")).strip(),
                "style": str(row.get("Style", "color_whiteboard")).strip(),
                "additional_prompt": str(row.get("Additional Prompt", "")).strip(),
                "target_length": str(row.get("Target Length", "10-12 min")).strip(),
            })

    if verbose:
        print(f"📋 Found {len(ready)} topics ready for production")
        for t in ready:
            print(f"   • {t['topic']} ({t['niche']}, {t['style']})")

    return ready


def update_status(
    row_number: int,
    status: str,
    sheet_id: Optional[str] = None,
    extra_data: Optional[dict] = None,
):
    """
    Update the status of a topic row in the Google Sheet.

    Args:
        row_number: The row number (1-indexed) to update.
        status: New status value.
        sheet_id: Google Sheet ID.
        extra_data: Optional dict of additional columns to update
                    (e.g., {"video_url": "...", "drive_link": "..."}).
    """
    sheet_id = sheet_id or GOOGLE_SHEET_ID
    if not sheet_id:
        return

    try:
        client = _get_client()
        sheet = client.open_by_key(sheet_id).sheet1

        # Update status
        sheet.update_cell(row_number, COLUMNS["status"], status)

        # Update extra columns if provided
        if extra_data:
            for key, value in extra_data.items():
                col = COLUMNS.get(key)
                if col:
                    sheet.update_cell(row_number, col, value)

    except Exception as e:
        print(f"⚠️ Failed to update sheet: {e}")


def create_sheet_template(sheet_id: Optional[str] = None):
    """
    Set up the header row in a new Google Sheet.
    Run this once to initialize your content calendar.
    """
    sheet_id = sheet_id or GOOGLE_SHEET_ID
    if not sheet_id:
        raise ValueError("GOOGLE_SHEET_ID is not set.")

    client = _get_client()
    sheet = client.open_by_key(sheet_id).sheet1

    headers = [
        "Topic", "Niche", "Style", "Additional Prompt",
        "Target Length", "Status", "Video URL", "Drive Link",
        "Thumbnail Link", "SEO Title", "SEO Description", "SEO Tags"
    ]

    sheet.update('A1:L1', [headers])
    print(f"✅ Sheet headers set: {headers}")
