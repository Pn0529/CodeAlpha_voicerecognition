import os
import zipfile
import requests

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1 MB
    downloaded = 0
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end='\r')
                else:
                    print(f"Downloaded: {downloaded / (1024*1024):.1f} MB", end='\r')
    print("\nDownload completed.")


def is_valid_zip(path):
    return os.path.exists(path) and zipfile.is_zipfile(path)


def main():
    dataset_dir = "dataset"
    zip_path = "Audio_Speech_Actors_01-24.zip"
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
    mirror_url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    if os.path.exists(os.path.join(dataset_dir, "Actor_01")):
        print("Dataset already downloaded and extracted.")
        return
        
    if os.path.exists(zip_path) and not is_valid_zip(zip_path):
        print("Existing dataset archive appears corrupted or incomplete. Removing and re-downloading.")
        os.remove(zip_path)

    if not os.path.exists(zip_path):
        try:
            download_file(url, zip_path)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print(f"Retrying with mirror: {mirror_url}")
            download_file(mirror_url, zip_path)

    if not is_valid_zip(zip_path):
        raise RuntimeError(f"Downloaded zip file '{zip_path}' is not a valid archive.")
        
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    print("Extraction completed.")
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("Cleaned up zip file.")

if __name__ == "__main__":
    main()
