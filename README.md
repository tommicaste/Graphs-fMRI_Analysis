# Graph-based Sleep Stage Classification

This repository contains a PyTorch Lightning-based pipeline for classifying sleep stages using various machine learning models, with a focus on Graph Neural Networks (GNNs). The project is designed to work with sleep data (likely EEG) pre-processed into graph-structured format, where nodes represent sensors and edges represent functional connectivity.

## Overview

The primary goal of this project is to explore different modeling approaches for sleep stage classification. It provides a flexible framework for training, evaluating, and comparing several architectures, including standard GNNs, weighted GNNs, a custom Neurograph model, a Brain Network Transformer (BNT), and baseline models like MLP and Logistic Regression.

The pipeline handles data loading, patient-aware splitting, data augmentation, model training, and evaluation. It is highly configurable through YAML files, allowing for easy experimentation with different models, hyperparameters, and data settings.

## Project Structure

The repository is organized as follows:


.
├── config/                  # YAML configuration files for experiments
│   ├── config.yaml
│   ├── config_baselines.yaml
│   ├── config_bnt.yaml
│   └── config_neurograph.yaml
├── data/                    # Scripts for data handling
│   ├── loader.py            # Main data loading and splitting logic
│   ├── split.py             # Patient-aware data splitting
│   ├── augment.py           # Data augmentation strategies
│   ├── graph.py             # Graph construction utilities
│   └── init.py
├── static_models/           # Model implementations
│   ├── gnn.py               # Standard GNN model
│   ├── weighted_gnn.py      # GNN that uses edge weights
│   ├── neurograph.py        # Custom Neurograph architecture
│   ├── bnt.py               # Brain Network Transformer model
│   ├── mlp.py               # MLP baseline
│   ├── logistic.py          # Logistic Regression baseline
│   ├── pipeline.py          # Core training and evaluation pipeline
│   ├── transforms.py        # Custom PyG transforms
│   └── utils.py             # Evaluation and plotting utilities
├── main.py                  # Main script to run the training pipeline
├── setup.py                 # Setup script for installing the package
└── README.md                # This file


## Main Features

* **Multiple Model Architectures:** Implements and compares several models:
    * `GNN`: A standard GNN model with configurable layers (e.g., GCN, SAGE).
    * `WeightedGNN`: A GNN that can utilize edge weights (e.g., correlation values).
    * `Neurograph`: A custom GNN architecture inspired by the NeuroGraph paper.
    * `BNT`: A Transformer-based model for brain network data.
    * `MLP` & `Logistic Regression`: Baseline models operating on flattened connectivity matrices.
* **Configurable Pipeline:** All aspects of the training process (model selection, hyperparameters, data splitting, augmentation) are controlled via YAML configuration files.
* **Data Augmentation:** Includes several strategies to address class imbalance:
    * `upsample`: Duplicates samples from minority classes.
    * `interpolate`: Creates synthetic samples by linearly interpolating between existing ones.
    * `geodesic`: Generates synthetic samples using geodesic interpolation on the manifold of SPD matrices.
* **Patient-Aware Splitting:** Ensures that data from the same patient does not appear in both the training and testing sets, preventing data leakage.
* **Automated Evaluation:** Automatically evaluates the best-performing model on the test set and generates a confusion matrix and a detailed classification report.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (e.g., conda or venv).
    ```bash
    pip install torch torchvision torchaudio
    pip install torch_geometric
    pip install pytorch-lightning numpy scikit-learn matplotlib pyyaml tqdm
    ```

3.  **Install the project package:**
    ```bash
    pip install -e .
    ```

## Usage

The main entry point for running experiments is `main.py`, which takes the path to a configuration file as an argument.

1.  **Prepare your data:**
    The `data/loader.py` expects a single `.pt` file containing a list of `torch_geometric.data.Data` objects. Each `Data` object should represent a single sample (e.g., a 30-second sleep epoch) and have at least the following attributes:
    * `x`: The feature matrix. For graph-based models, this is typically the node features (if any) or the adjacency/correlation matrix of shape `[num_nodes, num_nodes]`.
    * `y`: The target label (sleep stage).
    * `metadata`: A dictionary containing at least the `sample` (patient ID) and `segment` (epoch index) for data splitting and augmentation.

2.  **Configure your experiment:**
    Modify one of the YAML files in the `config/` directory or create a new one. Key sections include:
    * `data`: Specify the path to your data file, batch size, train/val/test split ratios, and augmentation strategy.
    * `models`: A list of models to train. For each model, specify its `name`, `architecture`, and `params` (hyperparameters).
    * `trainer`: Configure PyTorch Lightning trainer settings like `max_epochs`, `accelerator`, and `devices`.

3.  **Run the training pipeline:**
    ```bash
    python main.py config/config.yaml
    ```
    Replace `config/config.yaml` with the path to your desired configuration file.

## Configuration Details

The `config.yaml` files have three main sections:

* **`data`**:
    * `path`: Path to the input `.pt` data file.
    * `split`: A list of three floats for `[train_ratio, val_ratio, test_ratio]`.
    * `augment`:
        * `strategy`: `null`, `upsample`, `interpolate`, or `geodesic`.
        * `proportion`: The target proportion for upsampling/interpolation.
    * `edge`:
        * `tsh`: Threshold for creating edges from the adjacency matrix.
        * `top`: Percentage of top edges to keep. (Use either `tsh` or `top`, not both).

* **`models`**: A list of model configurations.
    * `name`: A custom name for the run, used for the output directory.
    * `architecture`: The model to use (e.g., `gnn`, `mlp`, `neurograph`). Must match a key in `MODEL_REGISTRY` in `static_models/pipeline.py`.
    * `params`: Model-specific hyperparameters (e.g., `hidden_channels`, `num_layers`, `lr`).

* **`trainer`**:
    * `max_epochs`: Maximum number of training epochs.
    * `accelerator`: `cpu`, `gpu`, `auto`.
    * `devices`: Number of devices to use.

## Evaluation

After training, the pipeline automatically runs the `test` step using the checkpoint with the best validation balanced accuracy. The results, including a `confusion_matrix.png` and `classification_report.json`, are saved in the run directory (e.g., `runs/<model_name>/results/`).
