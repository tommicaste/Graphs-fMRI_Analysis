# Graph based Sleep Stage Classification

This repository provides a PyTorch Lightning pipeline for classifying sleep stages from brain-network data such as fMRI or EEG correlation matrices. Nodes correspond to sensors and edges capture functional connectivity.

## Overview

The framework lets you train, evaluate, and compare several models:

* **GNN** — standard graph neural network (GCN, GraphSAGE, and similar)  
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
│   └── __init__.py
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

```bash
git clone https://github.com/tommicaste/Graphs-sleepstages.git
cd Graphs-sleepstages

# install every required library in one shot
pip install -r requirements.txt

# optional, for development
pip install -e .
```

`requirements.txt` already lists PyTorch, torch_geometric, PyTorch Lightning, and the usual data-science stack, so you no longer need to install them individually.

### GPU note

If you need a specific CUDA wheel, follow the official instructions on the PyTorch and torch_geometric websites.

## Usage

1. **Prepare data** — provide a single `.pt` file containing a list of `torch_geometric.data.Data` objects. Each object must include  
   * `x` – correlation matrix with shape `[num_nodes, num_nodes]`  
   * `y` – integer label for the sleep stage  
   * `metadata` – dictionary with keys `sample` (patient identifier) and `segment` (epoch index); leave empty if unavailable  

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

## CitationS

NeuroGraph  
Anwar Said, Roza G. Bayrak, Tyler Derr, Mudassir Shabbir, Daniel Moyer, Catie Chang, and Xenofon Koutsoukos. “NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics.” arXiv preprint arXiv:2306.06202, 2024.

Brain Network Transformer  
Xuan Kan, Wei Dai, Hejie Cui, Zilong Zhang, Ying Guo, and Carl Yang. “Brain Network Transformer.” Advances in Neural Information Processing Systems 35 (NeurIPS 2022).
