import os
import requests
from tqdm import tqdm
import sys
import time
import re
import config

API_KEY = config.API_KEY
SEARCH_ENGINE_ID = config.SEARCH_ENGINE_ID
IMAGE_FOLDER = os.path.join(config.DATA_FOLDER, "raw_objects")
PER_PAGE = 10
MAX_RESULTS = 50


def sanitize_folder_name(name):
    """
    Sanitize the folder name by removing or replacing invalid characters.
    This ensures compatibility across different operating systems.
    """
    sanitized = re.sub(r"[^\w\s-]", "", name)
    sanitized = sanitized.strip().replace(" ", "_")
    return sanitized


def create_folder(folder_path):
    """Create a folder if it doesn't exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")


def get_search_results(query, start_index=1):
    """Fetch image search results from Google Custom Search API."""
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "cx": SEARCH_ENGINE_ID,
        "key": API_KEY,
        "searchType": "image",
        # "imgLicense": "cc_publicdomain,cc_attribute,cc_sharealike,cc_noncommercial,cc_nonderived",
        "num": PER_PAGE,
        "start": start_index,
        "safe": "medium",
    }
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching search results: {e}")
        return None


def download_image(url, folder, idx):
    """Download a single image from a URL."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        ext = url.split(".")[-1].split("?")[0].lower()
        if ext not in ["jpg", "jpeg", "png", "gif", "bmp", "webp"]:
            ext = "jpg"
        image_name = os.path.join(folder, f"image_{idx}.{ext}")
        with open(image_name, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return False


def main():
    query = input("Enter search query: ").strip()
    if not query:
        print("Search query cannot be empty.")
        return

    try:
        max_results = int(
            input(f"Number of images to download (1-{MAX_RESULTS}) [50]: ") or 50
        )
        if max_results < 1 or max_results > MAX_RESULTS:
            print(
                f"Number of images must be between 1 and {MAX_RESULTS}. Using default value of 50."
            )
            max_results = 50
    except ValueError:
        print("Invalid input. Using default number of images: 50.")
        max_results = 50

    sanitized_query = sanitize_folder_name(query)
    target_folder = os.path.join(IMAGE_FOLDER, sanitized_query)
    create_folder(target_folder)

    images_downloaded = 0
    idx = 1
    start_index = 1

    while images_downloaded < max_results:
        print(f"\nFetching results starting at index {start_index}...")
        results = get_search_results(query, start_index)
        if not results or "items" not in results:
            print("No more results found or an error occurred.")
            break

        for item in tqdm(
            results["items"],
            desc=f"Downloading images {images_downloaded + 1} to {min(images_downloaded + PER_PAGE, max_results)}",
            unit="image",
        ):
            if images_downloaded >= max_results:
                break
            image_url = item.get("link")
            if image_url:
                success = download_image(image_url, target_folder, idx)
                if success:
                    images_downloaded += 1
                    idx += 1

        start_index += PER_PAGE
        if start_index > 100:
            print("Reached the maximum number of retrievable results (100).")
            break

        time.sleep(1)

    print(f"\nDownloaded {images_downloaded} images to '{target_folder}'.")


if __name__ == "__main__":
    main()
