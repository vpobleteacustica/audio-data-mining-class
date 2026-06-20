from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUBSET_DIR = PROJECT_ROOT / "data" / "processed" / "esc50_subset"
SUBSET_METADATA = SUBSET_DIR / "esc50_subset_metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "features"
OUTPUT_FEATURES = OUTPUT_DIR / "esc50_audio_features.csv"

SAMPLE_RATE = 22050
N_MFCC = 13


def extract_features(audio_path: Path) -> dict:
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    features = {
        "duration": duration,
        "rms_mean": np.mean(rms),
        "rms_std": np.std(rms),
        "zcr_mean": np.mean(zcr),
        "zcr_std": np.std(zcr),
        "spectral_centroid_mean": np.mean(spectral_centroid),
        "spectral_centroid_std": np.std(spectral_centroid),
        "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
        "spectral_bandwidth_std": np.std(spectral_bandwidth),
        "spectral_rolloff_mean": np.mean(spectral_rolloff),
        "spectral_rolloff_std": np.std(spectral_rolloff),
        "spectral_flatness_mean": np.mean(spectral_flatness),
        "spectral_flatness_std": np.std(spectral_flatness),
    }

    for i in range(N_MFCC):
        features[f"mfcc_{i + 1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i + 1}_std"] = np.std(mfcc[i])

    return features


def main() -> None:
    if not SUBSET_METADATA.exists():
        raise FileNotFoundError(
            f"Subset metadata not found: {SUBSET_METADATA}\n"
            "Run first: python scripts/02_prepare_subset.py"
        )

    metadata = pd.read_csv(SUBSET_METADATA)

    rows = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting features"):
        audio_path = PROJECT_ROOT / row["subset_audio_path"]

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        features = extract_features(audio_path)

        output_row = {
            "filename": row["filename"],
            "fold": row["fold"],
            "target": row["target"],
            "category": row["category"],
            "esc10": row["esc10"],
            "audio_path": row["subset_audio_path"],
        }

        output_row.update(features)
        rows.append(output_row)

    features_df = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_FEATURES, index=False)

    print("")
    print("Feature extraction completed successfully.")
    print(f"Number of audio files: {len(features_df)}")
    print(f"Number of columns: {features_df.shape[1]}")
    print(f"Output file: {OUTPUT_FEATURES}")

    print("")
    print("Class distribution:")
    print(features_df["category"].value_counts().sort_index())


if __name__ == "__main__":
    main()