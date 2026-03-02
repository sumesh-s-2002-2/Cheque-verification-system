# Cheque Signature ML — Verification Pipeline

Signature verification system for bank cheques using a Siamese network pretrained on EfficientNet.

---

## Project Structure

```
cheque-signature-ml/
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── src/
│   ├── config/
│   │   ├── preprocessing.yaml   ← all preprocessing globals
│   │   ├── augmentation.yaml    ← all augmentation globals
│   │   └── model.yaml           ← model, training & MLflow globals
│   ├── data/
│   │   ├── raw/                 ← place raw cheque scans here
│   │   ├── synthetic/
│   │   ├── processed/
│   │   ├── signatures/          ← extracted ROI outputs
│   │   └── labels/
│   ├── preprocessing/           ← ✅ fully implemented
│   │   ├── pipeline.py          ← orchestrator
│   │   ├── load_image.py
│   │   ├── dpi_normalization.py
│   │   ├── resolution_enforcement.py
│   │   ├── skew_correction.py
│   │   ├── perspective_correction.py
│   │   ├── background_removal.py
│   │   ├── noise_removal.py
│   │   └── roi_extraction.py
│   ├── augmentation/            ← TODO
│   ├── features/                ← TODO
│   ├── models/                  ← TODO
│   ├── verification/            ← TODO
│   ├── evaluation/              ← TODO
│   ├── utils/
│   │   ├── config_loader.py     ← YAML singleton loader
│   │   └── logging.py
│   └── main.py                  ← pipeline entry point
├── notebooks/
└── requirements.txt
```

---

## Quick Start

### 1. Clone & configure
```bash
cp .env.example .env
# Edit .env if needed
```

### 2. Run with Docker Compose
```bash
docker compose up --build
```

This starts three services:
| Service | URL | Purpose |
|---------|-----|---------|
| `mlflow` | http://localhost:5000 | Experiment tracker UI |
| `pipeline` | — | Runs `src/main.py` |
| `jupyter` | http://localhost:8888 | Notebook exploration |

### 3. Run locally (dev)
```bash
pip install -r requirements.txt
python src/main.py
```

---

## Configuration

All tunable values live in `src/config/*.yaml`. There are **no hardcoded parameters** in pipeline modules — every module imports from `src/utils/config_loader.py`.

| File | Controls |
|------|----------|
| `preprocessing.yaml` | DPI, resolution, skew, perspective, binarisation, noise, ROI |
| `augmentation.yaml` | Geometric, ink, noise, fault injection settings |
| `model.yaml` | Backbone, Siamese config, training hyperparams, MLflow settings |

---

## Preprocessing Pipeline

```
Raw cheque image
      ↓
[1] load_image          — read file, extract DPI metadata
      ↓
[2] dpi_normalization   — resample to target_dpi (default 300)
      ↓
[3] resolution_enforce  — clamp to [min, max] resolution
      ↓
[4] skew_correction     — Hough-line deskew
      ↓
[5] perspective_correct — four-point warp to top-down view
      ↓
[6] background_removal  — Otsu / adaptive binarisation
      ↓
[7] noise_removal       — median + morphological cleanup
      ↓
[8] roi_extraction      — crop & resize signature region
      ↓
Signature ROI (128×256 PNG)
```

Each stage is independently togglable via `enabled: true/false` in `preprocessing.yaml`.

---

## MLflow

All pipeline runs log params and metrics to MLflow automatically. View the UI at http://localhost:5000 after starting Docker Compose.
