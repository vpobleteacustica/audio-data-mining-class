# Audio Data Mining Class

Teaching material for an introductory class on audio data mining, audio signal processing, spectrograms, mel-spectrograms, MFCCs, feature extraction, and simple audio classification.

The repository uses the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset as a public dataset for environmental sound analysis.

## Main topics

- Audio as a discrete-time signal
- Sampling rate, duration, and waveform visualization
- Dataset inspection and basic audio auditing
- Fourier Transform, FFT, and magnitude spectra
- Short-Time Fourier Transform (STFT)
- Sliding windows, hop length, overlap, and smooth windowing
- Spectrograms and time-frequency representations
- Mel scale, mel filter banks, mel-spectrograms, and log-mel representations
- MFCCs, cepstrum intuition, DCT, and compact audio features
- Tabular feature extraction from audio signals
- Simple audio classification workflow

## Repository structure

```text
audio-data-mining-class/
│
├── data/
│   ├── raw/
│   ├── external/
│   ├── processed/
│   └── metadata/
│
├── notebooks/
│   ├── 00_dataset_audit.ipynb
│   ├── 01_audio_waveform.ipynb
│   ├── 02_spectrograms_stft.ipynb
│   ├── 03_mel_spectrograms.ipynb
│   ├── 03b_listening_experiment_mel_scale.ipynb
│   ├── 04_mfcc_feature_representation.ipynb
│   ├── 05_feature_extraction.ipynb
│   └── 06_audio_features_classification.ipynb
│
├── src/
│   ├── __init__.py
│   ├── audio_utils.py
│   ├── feature_extraction.py
│   ├── plot_config.py
│   └── visualization.py
│
├── scripts/
│   ├── 01_download_esc50.py
│   ├── 02_prepare_subset.py
│   └── 03_extract_features.py
│
├── outputs/
│   ├── figures/
│   └── features/
│
├── docs/
│   └── class_notes.md
│
├── environment.yml
├── requirements.txt
└── README.md
```

## Notebooks

### `00_dataset_audit.ipynb`

Dataset inspection and consistency checks.  
This notebook introduces the idea of auditing an audio dataset before applying signal processing or machine learning.

### `01_audio_waveform.ipynb`

Digital audio signals, sampling rate, pure tones, waveform visualization, and basic audio inspection.

### `02_spectrograms_stft.ipynb`

Fourier Transform, FFT, magnitude spectra, Short-Time Fourier Transform, spectrograms, sliding windows, hop length, overlap, and smooth windowing.

### `03_mel_spectrograms.ipynb`

Mel scale, pitch perception, mel filter banks, power spectrum, mel-spectrograms, log-mel representations, and visualization choices.

### `03b_listening_experiment_mel_scale.ipynb`

Optional listening experiment to illustrate that equal frequency differences in Hz are not perceived equally across the frequency range.

### `04_mfcc_feature_representation.ipynb`

MFCC feature representation, spectral envelope, cepstrum intuition, Discrete Cosine Transform, slow and fast spectral variations, and MFCC summary features.

### `05_feature_extraction.ipynb`

Extraction of time-domain, frequency-domain, and cepstral features for tabular machine learning.

### `06_audio_features_classification.ipynb`

Simple audio classification workflow using extracted features.

## Setup

Clone the repository:

```bash
git clone https://github.com/vpobleteacustica/audio-data-mining-class.git
cd audio-data-mining-class
```

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate audio_data_mining
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name audio_data_mining --display-name "Python (audio_data_mining)"
```

In VS Code or Jupyter, select the kernel:

```text
Python (audio_data_mining)
```

## Download and prepare ESC-50

The ESC-50 dataset is not stored in this repository.  
Students should download and prepare it locally using the provided scripts.

From the repository root, run:

```bash
python scripts/01_download_esc50.py
python scripts/02_prepare_subset.py
python scripts/03_extract_features.py
```

The scripts perform the following steps:

```text
01_download_esc50.py      downloads and extracts ESC-50
02_prepare_subset.py      creates a small subset for the class
03_extract_features.py    extracts a tabular feature table
```

The full dataset is stored locally in:

```text
data/external/ESC-50-master/
```

The subset used in the notebooks is stored locally in:

```text
data/processed/esc50_subset/
```

The extracted feature table is stored in:

```text
outputs/features/esc50_audio_features.csv
```

## Notes about data files

Audio files and downloaded datasets are intentionally excluded from Git.

The repository should contain code, notebooks, documentation, and lightweight metadata, but not large audio datasets.

If the dataset or processed audio files are missing, run:

```bash
python scripts/01_download_esc50.py
python scripts/02_prepare_subset.py
```

## Recommended workflow for students

1. Clone the repository.
2. Create and activate the conda environment.
3. Download ESC-50 using the script.
4. Prepare the class subset.
5. Open the notebooks in order.
6. Run each notebook using the `Python (audio_data_mining)` kernel.

Suggested order:

```text
00 → 01 → 02 → 03 → 03b → 04 → 05 → 06
```

## Teaching goal

The goal of this repository is not only to run audio processing code, but to understand the conceptual path from raw audio to machine-learning-ready features:

```text
audio waveform
    ↓
frames and windows
    ↓
Fourier Transform / STFT
    ↓
spectrogram
    ↓
mel filter bank
    ↓
mel-spectrogram
    ↓
log-mel-spectrogram
    ↓
MFCCs and other audio features
    ↓
tabular dataset
    ↓
simple classification model
```

## Requirements

The project uses Python and common scientific/audio libraries, including:

- `numpy`
- `pandas`
- `matplotlib`
- `librosa`
- `scikit-learn`
- `soundfile`
- `tqdm`
- `jupyter`
- `ipykernel`

The recommended installation method is through:

```bash
conda env create -f environment.yml
```

## License and dataset attribution

This repository is intended for teaching purposes.

The ESC-50 dataset is an external public dataset and is not redistributed in this repository.  
Please refer to the original ESC-50 repository and documentation for dataset licensing and citation information.
