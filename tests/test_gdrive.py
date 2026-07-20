import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock googleapiclient and google.auth modules
sys.modules['googleapiclient'] = MagicMock()
sys.modules['googleapiclient.discovery'] = MagicMock()
sys.modules['googleapiclient.http'] = MagicMock()
sys.modules['google.auth'] = MagicMock()

from src.gdrive import extract_gdrive_folder_id

def test_extract_gdrive_folder_id_urls():
    url1 = "https://drive.google.com/drive/folders/1abc123-xyz_789?usp=sharing"
    url2 = "https://drive.google.com/open?id=1abc123-xyz_789"
    url3 = "1abc123-xyz_789"
    
    assert extract_gdrive_folder_id(url1) == "1abc123-xyz_789"
    assert extract_gdrive_folder_id(url2) == "1abc123-xyz_789"
    assert extract_gdrive_folder_id(url3) == "1abc123-xyz_789"
    assert extract_gdrive_folder_id("") is None


def test_get_file_id_found():
    from src.gdrive import get_file_id
    service = MagicMock()
    
    # Mocking list().execute()
    mock_list = MagicMock()
    mock_list.execute.return_value = {
        "files": [{"id": "file_id_123"}]
    }
    service.files().list.return_value = mock_list
    
    file_id = get_file_id(service, "folder_id_xyz", "my_video.mp4")
    assert file_id == "file_id_123"
    
    # Verify subfolder traversal
    mock_list.execute.side_effect = [
        {"files": [{"id": "subfolder_id_abc"}]}, # subfolder search
        {"files": [{"id": "file_id_456"}]}       # file search
    ]
    file_id_sub = get_file_id(service, "folder_id_xyz", "my_video.mp4", "output")
    assert file_id_sub == "file_id_456"

