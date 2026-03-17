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
│   └── config_neurograph.yaml
├── data/                    # data loading, splitting, augmentation
│   ├── loader.py
│   ├── split.py
│   ├── augment.py
│   ├── graph.py
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
├── main.py                  # training entry point
├── requirements.txt         # Python dependencies
├── setup.py                 # package metadata
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

uv venv
source .venv/bin/activate

# For GPU training
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match

# For CPU training only:
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

uv pip install -e .
```

## Usage

1. **Prepare data** — provide a single `.pt` file containing a list of `torch_geometric.data.Data` objects. Each object must include  
   * `x` – correlation matrix with shape `[num_nodes, num_nodes]`  
   * `y` – integer label for the sleep stage  
   * `metadata` – dictionary with keys `sample`  and `segment` ; leave empty if unavailable  

2. **Edit a YAML** — pick one in `config/` or create your own. Key blocks:  
   * `data` – file path, batch size, split ratios, augmentation strategy  
   * `models` – list of models with `name`, `architecture`, and `params`  
   * `trainer` – `max_epochs`, `accelerator`, `devices`  

3. **Run training**

```bash
python main.py config/config.yaml
```

The script iterates through every model defined in the YAML, trains each one, and stores results under `runs/<model_name>/`.

## Evaluation

After training completes, the pipeline reloads the checkpoint with the best validation balanced accuracy and

* computes test metrics  
* saves `confusion_matrix.png`  
* writes `classification_report.json`

All artefacts are stored in `runs/<model_name>/results/`.

## Citations

NeuroGraph  
Anwar Said, Roza G. Bayrak, Tyler Derr, Mudassir Shabbir, Daniel Moyer, Catie Chang, and Xenofon Koutsoukos. “NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics.” arXiv preprint arXiv:2306.06202, 2024.

Brain Network Transformer  
Xuan Kan, Wei Dai, Hejie Cui, Zilong Zhang, Ying Guo, and Carl Yang. “Brain Network Transformer.” Advances in Neural Information Processing Systems 35 (NeurIPS 2022).
