from pathlib import Path
import zipfile
import requests
from tqdm import tqdm


ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
ZIP_PATH = DATA_EXTERNAL / "ESC-50-master.zip"
EXTRACTED_DIR = DATA_EXTERNAL / "ESC-50-master"


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with output_path.open("wb") as file, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {output_path.name}",
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        print(f"Dataset already extracted at: {output_dir}")
        return

    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(zip_path.parent)

    print(f"Dataset extracted at: {output_dir}")


def main() -> None:
    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)

    if EXTRACTED_DIR.exists():
        print(f"ESC-50 already exists at: {EXTRACTED_DIR}")
        return

    if not ZIP_PATH.exists():
        print("Downloading ESC-50...")
        download_file(ESC50_URL, ZIP_PATH)
    else:
        print(f"Zip file already exists at: {ZIP_PATH}")

    extract_zip(ZIP_PATH, EXTRACTED_DIR)

    metadata_path = EXTRACTED_DIR / "meta" / "esc50.csv"
    audio_dir = EXTRACTED_DIR / "audio"

    print("")
    print("Done.")
    print(f"Metadata: {metadata_path}")
    print(f"Audio directory: {audio_dir}")


if __name__ == "__main__":
    main()