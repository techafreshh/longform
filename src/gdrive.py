"""Google Drive API helpers for downloading/uploading shared project folders in Colab."""

import io
import os
import re
from pathlib import Path
from typing import Optional

def extract_gdrive_folder_id(url: str) -> Optional[str]:
    """Extract folder ID from a Google Drive URL."""
    if not url:
        return None
    # If it's already an ID and doesn't look like a URL
    if "/" not in url and "?" not in url and len(url) >= 10:
        return url
        
    # Match drive/folders/ID
    match = re.search(r'folders/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    # Match open?id=ID
    match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return None

def get_gdrive_service():
    """Build and return an authenticated Google Drive API service client."""
    try:
        from googleapiclient.discovery import build
        from google.auth import default
        creds, _ = default()
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"⚠️ Could not initialize Google Drive API service: {e}")
        return None

def download_drive_folder_api(service, folder_id: str, local_path: str):
    """Wrapper around incremental_sync_drive_to_local to maintain backward compatibility."""
    incremental_sync_drive_to_local(service, folder_id, Path(local_path))

def incremental_sync_drive_to_local(service, folder_id: str, local_path: Path):
    """
    Incrementally sync files and subfolders from a Google Drive folder to local_path.
    Only downloads new or size-mismatched files, and uses temp file renaming for data integrity.
    """
    from googleapiclient.http import MediaIoBaseDownload
    
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    
    page_token = None
    while True:
        query = f"'{folder_id}' in parents and trashed = false"
        results = service.files().list(
            q=query,
            pageToken=page_token,
            fields="nextPageToken, files(id, name, mimeType, size)"
        ).execute()
        
        items = results.get('files', [])
        for item in items:
            item_id = item['id']
            item_name = item['name']
            mime_type = item['mimeType']
            dest_item_path = local_path / item_name
            
            if mime_type == 'application/vnd.google-apps.folder':
                incremental_sync_drive_to_local(service, item_id, dest_item_path)
            else:
                remote_size = int(item.get('size', 0))
                
                # Check if local file exists and matches size
                if dest_item_path.exists():
                    local_size = dest_item_path.stat().st_size
                    if local_size == remote_size:
                        continue
                
                print(f"  📥 Downloading: {item_name} ({remote_size / (1024*1024):.2f} MB)...")
                try:
                    request = service.files().get_media(fileId=item_id)
                    fh = io.BytesIO()
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                    
                    # Write to temporary file first to guarantee download completeness/integrity
                    temp_path = dest_item_path.with_suffix(dest_item_path.suffix + ".tmp")
                    with open(temp_path, 'wb') as f:
                        f.write(fh.getvalue())
                    temp_path.replace(dest_item_path)
                except Exception as e:
                    print(f"  ⚠️ Failed to download {item_name}: {e}")
                    
        page_token = results.get('nextPageToken', None)
        if not page_token:
            break

def sync_gdrive_root_files(service, folder_id: str, local_path: Path):
    """Syncs only the top-level files of a Google Drive folder (no folders)."""
    from googleapiclient.http import MediaIoBaseDownload
    
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    
    page_token = None
    while True:
        query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder' and trashed = false"
        results = service.files().list(
            q=query,
            pageToken=page_token,
            fields="nextPageToken, files(id, name, size)"
        ).execute()
        
        items = results.get('files', [])
        for item in items:
            item_id = item['id']
            item_name = item['name']
            dest_item_path = local_path / item_name
            remote_size = int(item.get('size', 0))
            
            if dest_item_path.exists():
                if dest_item_path.stat().st_size == remote_size:
                    continue
                    
            print(f"  📥 Downloading root file: {item_name}...")
            try:
                request = service.files().get_media(fileId=item_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                
                temp_path = dest_item_path.with_suffix(dest_item_path.suffix + ".tmp")
                with open(temp_path, 'wb') as f:
                    f.write(fh.getvalue())
                temp_path.replace(dest_item_path)
            except Exception as e:
                print(f"  ⚠️ Failed to download {item_name}: {e}")
                
        page_token = results.get('nextPageToken', None)
        if not page_token:
            break

def download_gdrive_subfolder_incremental(service, parent_folder_id: str, subfolder_name: str, local_parent_path: Path) -> bool:
    """Finds a subfolder in the parent Google Drive folder and incrementally syncs it locally."""
    local_parent_path = Path(local_parent_path)
    query = f"'{parent_folder_id}' in parents and name = '{subfolder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    try:
        results = service.files().list(q=query, fields="files(id)").execute()
        files = results.get('files', [])
        if files:
            subfolder_id = files[0]['id']
            dest_path = local_parent_path / subfolder_name
            incremental_sync_drive_to_local(service, subfolder_id, dest_path)
            return True
    except Exception as e:
        print(f"⚠️ Failed to sync subfolder '{subfolder_name}': {e}")
    return False

def upload_file_to_drive_folder(service, folder_id: str, local_file_path: Path, gdrive_subfolder_name: Optional[str] = None) -> str:
    """Uploads a local file to a Google Drive folder. If a subfolder name is specified,
    it finds or creates that subfolder inside the parent folder first. Overwrites if file exists.
    """
    from googleapiclient.http import MediaFileUpload
    
    local_file_path = Path(local_file_path)
    if not local_file_path.exists():
        print(f"  ⚠️ File not found locally: {local_file_path}")
        return ""
        
    parent_id = folder_id
    if gdrive_subfolder_name:
        # Find or create the subfolder
        query = f"'{folder_id}' in parents and name = '{gdrive_subfolder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        results = service.files().list(q=query, fields="files(id)").execute()
        files = results.get('files', [])
        if files:
            parent_id = files[0]['id']
        else:
            # Create subfolder
            file_metadata = {
                'name': gdrive_subfolder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [folder_id]
            }
            subfolder = service.files().create(body=file_metadata, fields='id').execute()
            parent_id = subfolder.get('id')
            print(f"  Created subfolder '{gdrive_subfolder_name}' on Google Drive.")
            
    # Check if file already exists in the parent_id folder to update/overwrite it
    filename = local_file_path.name
    query = f"'{parent_id}' in parents and name = '{filename}' and trashed = false"
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get('files', [])
    
    media = MediaFileUpload(str(local_file_path), resumable=True)
    
    if files:
        file_id = files[0]['id']
        print(f"  Updating existing file '{filename}' on Google Drive...")
        updated_file = service.files().update(
            fileId=file_id,
            media_body=media
        ).execute()
        return updated_file.get('id')
    else:
        print(f"  Uploading new file '{filename}' to Google Drive...")
        file_metadata = {
            'name': filename,
            'parents': [parent_id]
        }
        new_file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        return new_file.get('id')

def sync_local_outputs_to_drive(service, folder_id: str, paths) -> bool:
    """Sync newly generated files in output, thumbnails, and seo.json back to the Google Drive folder."""
    print(f"📤 Syncing local outputs back to Google Drive (Folder ID: {folder_id})...")
    try:
        # 1. Subtitles and Final Video in output_dir
        if paths.output_dir.exists():
            for f in paths.output_dir.iterdir():
                if f.is_file():
                    upload_file_to_drive_folder(service, folder_id, f, "output")
                    
        # 2. Thumbnails
        if paths.thumbnail_dir.exists():
            for f in paths.thumbnail_dir.iterdir():
                if f.is_file():
                    upload_file_to_drive_folder(service, folder_id, f, "thumbnails")
                    
        # 3. SEO json
        if paths.seo_file.exists():
            upload_file_to_drive_folder(service, folder_id, paths.seo_file)
            
        print("✅ Sync to Google Drive complete!")
        return True
    except Exception as e:
        print(f"⚠️ Error syncing outputs to Google Drive: {e}")
        return False


def get_file_id(service, parent_folder_id: str, filename: str, gdrive_subfolder_name: Optional[str] = None) -> str:
    """Finds the ID of a file in Google Drive without uploading/modifying it."""
    try:
        parent_id = parent_folder_id
        if gdrive_subfolder_name:
            query = f"'{parent_folder_id}' in parents and name = '{gdrive_subfolder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            results = service.files().list(q=query, fields="files(id)").execute()
            files = results.get('files', [])
            if files:
                parent_id = files[0]['id']
            else:
                return ""
        
        query = f"'{parent_id}' in parents and name = '{filename}' and trashed = false"
        results = service.files().list(q=query, fields="files(id)").execute()
        files = results.get('files', [])
        if files:
            return files[0]['id']
    except Exception as e:
        print(f"⚠️ Error finding file ID for '{filename}' on Google Drive: {e}")
    return ""


def get_file_id_by_path(service, local_path: Path) -> str:
    """Finds the Google Drive file ID for a local path situated within mounted Google Drive."""
    try:
        parts = list(local_path.resolve().parts)
        
        # Check if the path contains 'drive' and 'MyDrive'
        if 'drive' not in parts or 'MyDrive' not in parts:
            return ""
            
        # Find where 'MyDrive' is in the path
        idx = parts.index('MyDrive')
        
        # Get the path components after 'MyDrive'
        relative_parts = parts[idx+1:]
        if not relative_parts:
            return ""
            
        # Traverse from 'root'
        parent_id = "root"
        for part in relative_parts[:-1]:
            # Find the subfolder
            query = f"'{parent_id}' in parents and name = '{part}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            results = service.files().list(
                q=query, 
                fields="files(id)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            files = results.get('files', [])
            if files:
                parent_id = files[0]['id']
            else:
                return ""
                
        # Find the leaf file
        filename = relative_parts[-1]
        query = f"'{parent_id}' in parents and name = '{filename}' and trashed = false"
        results = service.files().list(
            q=query, 
            fields="files(id)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        files = results.get('files', [])
        if files:
            return files[0]['id']
    except Exception as e:
        print(f"⚠️ Error resolving file ID by path: {e}")
    return ""


