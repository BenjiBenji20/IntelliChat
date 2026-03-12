import os
import sys
import time
import requests
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment before anything else
load_dotenv(os.path.join(project_root, '.env'))

from modules.documents.gcs_service import gcs_service

def run_integration_tests():
    # Setup paths
    results_dir = os.path.join(project_root, 'tests/results/module')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'gcs_service.txt')

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=== GCS Service Integration Test Results ===\n\n")

    def log_result(test_name, result):
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(f"--- {test_name} ---\n")
            f.write(result + "\n\n")
        print(f"[{test_name}] executed.")

    # Shared Test Config
    test_files_dir = os.path.join(os.path.dirname(__file__), 'test_files')
    happy_path_file = os.path.join(test_files_dir, 'happy_path.txt')
    
    # ----------------------------------------------------------------------
    # Test Case 1: Happy Path
    # ----------------------------------------------------------------------
    try:
        happy_object_key = "test-uploads/happy-path.txt"
        file_size = os.path.getsize(happy_path_file)
        content_type = "text/plain"

        with open(happy_path_file, 'rb') as f:
            file_data = f.read()

        # Generate upload URL
        upload_url = gcs_service.generate_signed_upload_url(happy_object_key, content_type, file_size)
        
        # Upload
        upload_response = requests.put(
            upload_url,
            data=file_data,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(file_size),
                "x-goog-content-length-range": f"1,{file_size}"
            }
        )
        
        if upload_response.status_code != 200:
            raise Exception(f"Upload failed. Status code: {upload_response.status_code}, Response: {upload_response.text}")

        # Check exists
        exists_after_upload = gcs_service.object_exists(happy_object_key)
        assert exists_after_upload is True, "object_exists() should return True after upload"
        
        time.sleep(30)  # 30sec interuption to check file upload in GCS before file deletion 

        # Get metadata
        metadata = gcs_service.get_object_metadata(happy_object_key)

        # Generate download URL
        download_url = gcs_service.generate_signed_download_url(happy_object_key)
        
        # Download and verify
        download_response = requests.get(download_url)
        downloaded_text = download_response.text

        # Delete
        gcs_service.delete_object(happy_object_key)
        
        # Check exists again
        exists_after_delete = gcs_service.object_exists(happy_object_key)
        assert exists_after_delete is False, "object_exists() should return False after deletion"

        log_result(
            "Test Case 1: Happy Path",
            f"Upload URL generated successfully.\n"
            f"Upload Status Code: {upload_response.status_code}\n"
            f"Exists after upload: {exists_after_upload}\n"
            f"Metadata Size: {metadata.get('size') if metadata else 'None'}\n"
            f"Metadata Content-Type: {metadata.get('content_type') if metadata else 'None'}\n"
            f"Download URL generated successfully.\n"
            f"Downloaded Data matches initial file: {downloaded_text == file_data.decode('utf-8')}\n"
            f"Exists after delete: {exists_after_delete}"
        )

    except Exception as e:
        log_result("Test Case 1: Happy Path", f"FAILED: {e}")
        raise e

    # ----------------------------------------------------------------------
    # Test Case 2: Reject Unsupported Content Type (Google Side Size Restriction / Invalid content type header mismatches)
    # ----------------------------------------------------------------------
    try:
        unsupported_key = "test-uploads/unsupported.jpg"
        content_type = "image/jpeg"
        # We will attempt to generate a signed URL for an image but pass a text payload
        upload_url = gcs_service.generate_signed_upload_url(unsupported_key, content_type, file_size)
        
        upload_response = requests.put(
            upload_url,
            data=file_data, # Using text data but declaring it as an image
            headers={
                "Content-Type": "text/plain", # We try to trick google
                "Content-Length": str(file_size),
                "x-goog-content-length-range": f"1,{file_size}"
            }
        )
        
        # Google should reject it because the signature mandates image/jpeg as content_type
        log_result(
            "Test Case 2: Reject Unsupported Content Type",
            f"Attempted to upload text/plain to a URL signed for image/jpeg.\n"
            f"Upload Status Code: {upload_response.status_code}\n"
            f"Expected rejection (403 Forbidden): {upload_response.status_code == 403}"
        )
        gcs_service.delete_object(unsupported_key)
        
    except Exception as e:
        log_result("Test Case 2: Reject Unsupported Content Type", f"FAILED: {e}")

    # ----------------------------------------------------------------------
    # Test Case 3: Check Existence of a Non-Existent Object
    # ----------------------------------------------------------------------
    try:
        missing_key = "test-uploads/does-not-exist.txt"
        exists = gcs_service.object_exists(missing_key)
        metadata = gcs_service.get_object_metadata(missing_key)

        log_result(
            "Test Case 3: Check Existence of Non-Existent Object",
            f"Exists: {exists}\n"
            f"Metadata is None: {metadata is None}"
        )
    except Exception as e:
        log_result("Test Case 3: Check Existence of Non-Existent Object", f"FAILED: {e}")
        raise e

    # ----------------------------------------------------------------------
    # Test Case 4: Delete a Non-Existent Object
    # ----------------------------------------------------------------------
    try:
        missing_key = "test-uploads/already-deleted.txt"
        gcs_service.delete_object(missing_key)

        log_result(
            "Test Case 4: Delete Non-Existent Object",
            f"Deletion executed silently without throwing handled exceptions."
        )
    except Exception as e:
        log_result("Test Case 4: Delete Non-Existent Object", f"FAILED: {e}")
        raise e

if __name__ == "__main__":
    run_integration_tests()
