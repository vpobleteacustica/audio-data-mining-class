from pathlib import Path
import shutil

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ESC50_DIR = PROJECT_ROOT / "data" / "external" / "ESC-50-master"
ESC50_METADATA = ESC50_DIR / "meta" / "esc50.csv"
ESC50_AUDIO_DIR = ESC50_DIR / "audio"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "esc50_subset"
OUTPUT_AUDIO_DIR = OUTPUT_DIR / "audio"
OUTPUT_METADATA = OUTPUT_DIR / "esc50_subset_metadata.csv"

SELECTED_CLASSES = [
    "rain",
    "sea_waves",
    "dog",
    "chirping_birds",
    "clock_tick",
    "keyboard_typing",
    "sneezing",
    "helicopter",
]

MAX_FILES_PER_CLASS = 10


def main() -> None:
    if not ESC50_METADATA.exists():
        raise FileNotFoundError(
            f"ESC-50 metadata not found: {ESC50_METADATA}\n"
            "Run first: python scripts/01_download_esc50.py"
        )

    metadata = pd.read_csv(ESC50_METADATA)

    subset = metadata[metadata["category"].isin(SELECTED_CLASSES)].copy()

    subset = (
        subset
        .sort_values(["category", "fold", "filename"])
        .groupby("category", group_keys=False)
        .head(MAX_FILES_PER_CLASS)
        .reset_index(drop=True)
    )

    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    copied_files = []

    for _, row in subset.iterrows():
        src = ESC50_AUDIO_DIR / row["filename"]
        dst = OUTPUT_AUDIO_DIR / row["filename"]

        if not src.exists():
            raise FileNotFoundError(f"Audio file not found: {src}")

        shutil.copy2(src, dst)
        copied_files.append(dst.name)

    subset["subset_audio_path"] = subset["filename"].apply(
        lambda name: str(Path("data") / "processed" / "esc50_subset" / "audio" / name)
    )

    subset.to_csv(OUTPUT_METADATA, index=False)

    print("ESC-50 subset prepared successfully.")
    print(f"Selected classes: {SELECTED_CLASSES}")
    print(f"Files per class: {MAX_FILES_PER_CLASS}")
    print(f"Total files copied: {len(copied_files)}")
    print(f"Output audio directory: {OUTPUT_AUDIO_DIR}")
    print(f"Output metadata: {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()