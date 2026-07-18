import pytest
from unittest.mock import MagicMock, patch
from src.sheets import _should_update_cell, update_status

def test_should_update_cell():
    # Empty new value -> False (preserve what is in sheet)
    assert not _should_update_cell("seo_title", "Existing Title", "")
    assert not _should_update_cell("seo_title", "", "")
    
    # Empty existing value + non-empty new value -> True
    assert _should_update_cell("seo_title", "", "New Title")
    
    # Web link in existing value, local path in new value -> False
    assert not _should_update_cell("drive_link", "https://drive.google.com/123", "/content/local/path")
    
    # Overwriting a web link with another web link if it's already set -> False (keep what is there)
    assert not _should_update_cell("drive_link", "https://drive.google.com/123", "https://drive.google.com/456")
    
    # Status column is updated if status changes
    assert _should_update_cell("status", "review_script", "complete")
    assert not _should_update_cell("status", "complete", "complete")
    
    # Content fields already populated -> False (keep what is there)
    assert not _should_update_cell("seo_title", "Manual Title", "Auto Title")
    assert not _should_update_cell("seo_tags", "Tag1, Tag2", "NewTag1, NewTag2")


@patch("src.sheets._get_client")
def test_update_status_safety_guards(mock_get_client):
    mock_sheet = MagicMock()
    mock_client = MagicMock()
    mock_client.open_by_key.return_value.sheet1 = mock_sheet
    mock_get_client.return_value = mock_client
    
    # Mock row_values return to simulate:
    # Col 6 (status) = "generating"
    # Col 8 (drive_link) = "https://drive.google.com/existing"
    # Col 10 (seo_title) = ""
    # Other columns empty
    mock_sheet.row_values.return_value = [
        "Topic", "Niche", "Style", "Prompt", "10 min", "generating", "", "https://drive.google.com/existing", "", "", "", ""
    ]
    
    update_status(
        row_number=2,
        status="complete",
        sheet_id="mock_sheet_id",
        extra_data={
            "drive_link": "/content/local/video.mp4", # Web link exists, should not overwrite
            "seo_title": "Newly Generated SEO Title",  # Existing is empty, should write
            "seo_description": "",                     # New value is empty, should not write
        }
    )
    
    # Assert status was updated to "complete" (Col 6)
    mock_sheet.update_cell.assert_any_call(2, 6, "complete")
    
    # Assert seo_title was updated (Col 10)
    mock_sheet.update_cell.assert_any_call(2, 10, "Newly Generated SEO Title")
    
    # Assert drive_link was NOT updated since we had a web link and new was local path
    for call in mock_sheet.update_cell.call_args_list:
        args = call[0]
        # Col 8 is drive_link
        assert args[1] != 8
