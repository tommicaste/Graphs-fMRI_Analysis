import os
import torch
import numpy as np
from collections import Counter
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

def import_dynamic_data(
    data_list, 
    window_size=25, 
    out_dir="/project2/cdonnat/sleepstages/data/pt/dynamic"
):
    """
    For each unique patient in data_list, creates sliding windows of size window_size 
    (allowing a shorter final window), packs each into a StaticGraphTemporalSignal loader,
    and saves the list of dicts to '{sample}_dynamic.pt' under out_dir.
    
    Each Data in data_list must have:
      - .x: Tensor [num_nodes, num_feats]
      - .y: Tensor label
      - .metadata['sample']: patient ID string
      - .metadata['segment']: integer segment index
    """
    os.makedirs(out_dir, exist_ok=True)
    # Identify all unique samples
    samples = sorted({d.metadata["sample"] for d in data_list})
    
    for sample in samples:
        # Filter and sort
        patient_data = [
            d for d in data_list if d.metadata["sample"] == sample
        ]
        patient_data_sorted = sorted(
            patient_data, key=lambda d: d.metadata["segment"]
        )
        
        window_records = []
        N = len(patient_data_sorted)
        
        for start in range(0, N, window_size):
            window = patient_data_sorted[start : start + window_size]
            
            # Build dynamic_id
            start_idx = window[0].metadata["segment"]
            end_idx   = window[-1].metadata["segment"]
            dynamic_id = f"{sample}_{start_idx}-{end_idx}"
        
            
            # Convert to NumPy for loader
            features_np = [d.x.cpu().numpy() for d in window]
            targets_np  = [d.y.cpu().numpy() for d in window]
            
            # Empty edge topology
            edge_index  = np.empty((2, 0), dtype=int)
            edge_weight = None
            
            # Create loader
            loader = StaticGraphTemporalSignal(
                edge_index=edge_index,
                edge_weight=edge_weight,
                features=features_np,
                targets=targets_np,
            )
            
            window_records.append({
                "id":           dynamic_id,
                "loader":       loader,
            })
        
        # Save per-sample dynamic data
        out_path = os.path.join(out_dir, f"{sample}_dynamic.pt")
        torch.save(window_records, out_path)
        print(f"Saved {len(window_records)} windows for {sample} to {out_path}")
