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
