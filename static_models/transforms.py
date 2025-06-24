import torch
from torch_geometric.transforms import BaseTransform
import torch.nn as nn


class BatchEdgeListTransform(BaseTransform):
    """
    Build a single `edge_index` for a PyG Batch from stacked correlation
    matrices in `batch.x`.
    """

    def __init__(self, *, top: float | None = None, tsh: float | None = None):
        super().__init__()
        
        # --- Parameter Validation ---
        if top is not None and tsh is not None:
            raise ValueError("Please specify either 'top' or 'tsh', but not both.")
        if top is None and tsh is None:
            pass

        if top is not None and not (0.0 < top <= 1.0):
            raise ValueError("Parameter 'top' must be in the interval (0, 1].")
        
        self.top = top
        self.tsh = tsh

    def __call__(self, batch):
        # Passthrough behaviour when no edge creation strategy is specified
        if self.top is None and self.tsh is None:
            if hasattr(batch, "edge_index"):
                return batch
            else:
                raise AttributeError(
                    "No edge creation strategy specified and `batch` lacks `edge_index`."
                )

        if not hasattr(batch, "x"):
            raise AttributeError(
                "Batch must contain attribute `batch.x` of shape (B*N, N)."
            )

        # Reshape to (B, N, N)
        total_rows, N = batch.x.shape
        if total_rows % N != 0:
            raise ValueError("batch.x rows not divisible by N; graphs have unequal sizes?")
        B = total_rows // N
        X = batch.x.view(B, N, N)
        dev = X.device

        # Get strict lower triangle indices and their values
        i, j = torch.tril_indices(N, N, offset=-1, device=dev)
        vals = X[:, i, j] 

        
        if self.tsh is not None:
            
            mask = vals >= self.tsh
            edge_coords = mask.nonzero(as_tuple=False)
            batch_idx, edge_in_tri_idx = edge_coords[:, 0], edge_coords[:, 1]
            selected_i, selected_j = i[edge_in_tri_idx], j[edge_in_tri_idx]
            offset = batch_idx * N
            src, dst = selected_j + offset, selected_i + offset
        else:
            
            k = int(self.top * vals.size(1))
            if k == 0:
                batch.edge_index = torch.empty((2, 0), dtype=torch.long, device=dev)
                return batch
            _, idx = vals.topk(k, dim=1)
            i_sel = torch.gather(i.expand(B, -1), 1, idx)
            j_sel = torch.gather(j.expand(B, -1), 1, idx)
            offset = (torch.arange(B, device=dev) * N).unsqueeze(1)
            src = (j_sel + offset).flatten()
            dst = (i_sel + offset).flatten()

        
        batch.edge_index = torch.stack((src, dst), dim=0)
        return batch
    

class LowerTriFlattenBatch(nn.Module):
    """
    PyG batch of symmetric N × N matrices:
        data.x      (B·N, N)
        data.batch  (B·N,)
    returns
        (B, N(N−1)//2)   strictly lower triangular, diagonal excluded
    """
    def __init__(self, n: int):
        super().__init__()
        r, c = torch.tril_indices(n, n, offset=-1)
        self.register_buffer("rows", r, persistent=False)
        self.register_buffer("cols", c, persistent=False)
        self.n = n

    def forward(self, data):
        x, batch_vec = data.x, data.batch
        B = int(batch_vec.max()) + 1
        x = x.view(B, self.n, self.n)
        return x[:, self.rows, self.cols].contiguous()