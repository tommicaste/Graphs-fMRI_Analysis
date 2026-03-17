# Graph based fMRI Classification

This repository provides a PyTorch Lightning pipeline for classifying sleep stages from brain-network data formatted as fMRI BOLD correlation matrices where nodes correspond to ROIs.

## Overview

The framework lets you train, evaluate, and compare several models:

* **GNN** — standard graph neural network (GCN, GraphSAGE, and similar) with architectural tweaks (residual connections, dropouts and similar)
* **WeightedGNN** — same network with edge weights
* **Neurograph** — custom architecture inspired by the NeuroGraph paper
* **Brain Network Transformer** — transformer-style model for graph inputs
* **MLP** and **Logistic Regression** — baselines on flattened connectivity

The code handles patient-aware splitting, class-imbalance augmentation, and fully automated training plus evaluation. All options live in YAML, so experiments remain reproducible and easy to tweak.

## Project Structure

```text
.
├── config/                  # YAML configuration files for experiments
│   ├── config.yaml
│   ├── config_baselines.yaml
│   ├── config_bnt.yaml
│   ├── config_neurograph.yaml
│   └── config_test.yaml    
├── data/                    # data loading, splitting, augmentation
│   ├── dataset.py          
│   ├── loader.py
│   ├── split.py
│   ├── augment.py
│   └── graph.py
├── static_models/           # model implementations
│   ├── gnn.py
│   ├── weighted_gnn.py
│   ├── neurograph.py
│   ├── bnt.py
│   ├── mlp.py
│   ├── logistic.py
│   ├── pipeline.py
│   ├── transforms.py
│   └── utils.py
├── tests/
│   └── create_fake_dataset.py   # generates a synthetic dataset for testing
├── main.py                  # training entry point
├── requirements.txt         # Python dependencies
├── pyproject.toml           # package metadata
└── README.md
```

## Installation

We recommend using `uv` for fast, reliable dependency management.

```bash
conda install uv
```

```bash
git clone https://github.com/tommicaste/Graphs-fMRI_Analysis.git
cd Graphs-fMRI_Analysis

uv venv --python 3.10
source .venv/bin/activate

# 1. Install PyTorch (pick your CUDA version, or use "cpu")
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match

# 2. Install PyG C++ extensions (prebuilt wheels matched to your torch + CUDA)
uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

# 3. Install remaining dependencies
uv pip install numpy matplotlib scikit-learn tqdm PyYAML torchmetrics torch-geometric==2.6.1

# 4. Install the project
uv pip install -e .
```

For **CPU-only** training, replace `cu126` with `cpu` in steps 1 and 2.

## Data format

The pipeline expects a single `.pt` file containing a list of `torch_geometric.data.Data` objects. Each object must have:

| Attribute | Description |
|-----------|-------------|
| `x` | Symmetric correlation matrix, shape `[num_nodes, num_nodes]` |
| `y` | Integer class label (e.g. sleep stage 0–3) |
| `metadata` | Dict with keys `"sample"` (patient ID) and `"segment"` (timepoint index) |

On load, the pipeline wraps each split into an `FMRIGraphDataset` (`torch_geometric.data.InMemoryDataset` subclass) before constructing DataLoaders. This is the standard PyG pattern for in-memory graph datasets.

## Smoke test (no real data needed)

Before running on real data, you can verify the full pipeline end-to-end with a synthetic dataset:

```bash
# 1. Generate fake data (80 samples, 26 nodes, 4 classes, 8 patients)
python tests/create_fake_dataset.py

# 2. Run all 6 models for 2 epochs on the fake data
python main.py config/config_test.yaml
```

Results are written to `runs_test/<model_name>/results/`. Check that all six models complete without errors before switching to real data.

## Usage

1. **Prepare data** — produce a `.pt` file in the format described above and place it somewhere accessible.

2. **Edit a YAML** — pick one in `config/` or create your own. Key blocks:
   * `data` — file path, batch size, split ratios, augmentation strategy
   * `models` — list of models with `name`, `architecture`, and `params`
   * `trainer` — `max_epochs`, `accelerator`, `devices`

3. **Run training**

```bash
python main.py config/config.yaml
```

The script iterates through every model defined in the YAML, trains each one, and stores results under `runs/<model_name>/`.

## Augmentation strategies

Set `data.augment.strategy` in your YAML to one of:

| Strategy | Description |
|----------|-------------|
| `null` | No augmentation |
| `upsample` | Random duplication of minority-class samples |
| `interpolate` | Linear interpolation between consecutive temporal samples, projected to the PSD correlation manifold |
| `geodesic` | Geodesic interpolation on the PSD matrix manifold |

All strategies require `data.augment.proportion` (target proportion of the largest class, in `(0, 1]`).

## Evaluation

After training completes, the pipeline reloads the checkpoint with the best validation balanced accuracy and:

* computes test metrics
* saves `confusion_matrix.png`
* writes `classification_report.json`

All artefacts are stored in `runs/<model_name>/results/`.

## Citations

NeuroGraph
Anwar Said, Roza G. Bayrak, Tyler Derr, Mudassir Shabbir, Daniel Moyer, Catie Chang, and Xenofon Koutsoukos. "NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics." arXiv preprint arXiv:2306.06202, 2024.

Brain Network Transformer
Xuan Kan, Wei Dai, Hejie Cui, Zilong Zhang, Ying Guo, and Carl Yang. "Brain Network Transformer." Advances in Neural Information Processing Systems 35 (NeurIPS 2022).
