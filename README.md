# Un-supervised: Crisis Tweet Clustering

[![Project Status](https://img.shields.io/badge/status-active-success?style=flat-square)](#)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Domain](https://img.shields.io/badge/domain-humanitarian%20NLP-8A2BE2?style=flat-square)](#)
[![Workflow](https://img.shields.io/badge/pipeline-unsupervised%20learning-4B8BBE?style=flat-square)](#)
[![License](https://img.shields.io/badge/license-MIT-informational?style=flat-square)](#license)

An unsupervised learning pipeline for crisis tweet analysis using the HumAID and CrisisBench datasets.
The project focuses on discovering latent structure in humanitarian crisis tweets via preprocessing, sentence embeddings, dimensionality reduction, and clustering.

---

## Overview

This repository implements an end-to-end unsupervised NLP workflow for humanitarian crisis tweets.
The pipeline is modular and notebook-driven, starting from raw text and ending with interpretable tweet clusters suitable for downstream analysis or prototyping supervised models.

Key components:

- Text preprocessing and normalization
- Exploratory data analysis (EDA)
- Sentence-level feature extraction
- Dimensionality reduction (UMAP, PCA, t-SNE)
- KMeans clustering with elbow and silhouette-based evaluation

---

## Pipeline

| Notebook | Description |
|---|---|
| `01_data_download.ipynb` | Download HumAID and CrisisBench datasets |
| `02_preprocessing.ipynb` | Clean and normalize tweet text (URLs, mentions, emojis, casing, etc.) |
| `03_eda.ipynb` | Exploratory analysis: label distributions, tweet length, hashtags |
| `04_feature_extraction.ipynb` | Generate sentence embeddings for tweets |
| `05_dimensionality_reduction.ipynb` | Compare UMAP, PCA, t-SNE and run small grid searches |
| `06_clustering.ipynb` | KMeans clustering with elbow method and silhouette analysis |

Run the notebooks in order (01 -> 06) to reproduce the full pipeline.

---

## Project Structure

```bash
Un-supervised/
├── notebooks/          # Jupyter notebooks (pipeline steps)
├── config/             # YAML configuration files
├── data/
│   └── processed/      # Cleaned data, embeddings, intermediate artifacts
├── figures/            # Generated plots and visualizations
├── scripts/            # Shell scripts to run notebooks end-to-end
├── requirements.txt    # Python dependencies
└── plan.md             # Project plan and design notes
```

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/Cyril-36/Un-supervised.git
cd Un-supervised
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## How to Run

### Option 1: End-to-end via script

```bash
bash scripts/run_notebooks.sh
```

This executes all pipeline steps sequentially and populates `data/processed` and `figures`.

### Option 2: Step-by-step in Jupyter

**1.** Launch Jupyter:

```bash
jupyter notebook
```

**2.** Open the notebooks in `notebooks/` from `01_data_download.ipynb` through `06_clustering.ipynb` and run them in order.

---

## Datasets

- **HumAID** - Humanitarian AI dataset of crisis tweets with event-level annotations.
- **CrisisBench** - Multi-event crisis tweet benchmark including events such as:
  - Alberta floods
  - Nepal earthquake
  - Hurricane Harvey

Refer to the original dataset documentation for licensing, terms of use, and citation requirements.

---

## Technical Stack

[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![HuggingFace](https://img.shields.io/badge/sentence--transformers-embeddings-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://www.sbert.net/)
[![UMAP](https://img.shields.io/badge/umap--learn-dimensionality%20reduction-5C6BC0?style=flat-square)](#)
[![Pandas](https://img.shields.io/badge/pandas-data%20processing-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-visualization-11557C?style=flat-square)](https://matplotlib.org/)

---

## Possible Extensions

- Experiment with alternative embedding models (e.g., different sentence-transformers variants).
- Evaluate additional clustering algorithms (HDBSCAN, spectral clustering, Gaussian mixtures).
- Add automatic cluster labeling and qualitative analysis reports or dashboards.
- Extend the pipeline to multilingual crisis tweet streams.

---

## License

This project is intended to be released under the MIT License.
Add a `LICENSE` file in the repository root to finalize the licensing information.
