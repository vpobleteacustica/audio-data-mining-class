# Audio Data Mining Class

This repository contains teaching material for an introductory class on audio data mining, audio signal processing, spectrograms, mel-spectrograms, and feature extraction.

The class uses the ESC-50 dataset as a public audio dataset for environmental sound analysis.

## Main topics

- Audio as a discrete-time signal
- Waveform visualization
- Short-Time Fourier Transform
- Spectrograms
- Mel-spectrograms
- Audio feature extraction
- Tabular datasets from audio features
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
│   ├── 01_audio_waveform.ipynb
│   ├── 02_spectrograms_stft.ipynb
│   ├── 03_mel_spectrograms.ipynb
│   ├── 04_feature_extraction.ipynb
│   └── 05_audio_features_classification.ipynb
│
├── src/
│   ├── audio_utils.py
│   ├── feature_extraction.py
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
└── docs/
    └── class_notes.md
