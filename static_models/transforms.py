import torch
from torch_geometric.transforms import BaseTransform
import torch.nn as nn
from torch_sparse import SparseTensor, matmul as sparse_matmul

class BatchEdgeListTransform(BaseTransform):

    def __init__(self, *, top: float | None = None, tsh: float | None = None, weighted: bool = False):
        super().__init__()
        
        if top is not None and tsh is not None:
            raise ValueError("Please specify either 'top' or 'tsh', but not both.")
        if top is None and tsh is None:
            pass

        if top is not None and not (0.0 < top <= 1.0):
            raise ValueError("Parameter 'top' must be in the interval (0, 1].")
        
        self.top = top
        self.tsh = tsh
        self.weighted = weighted

    def __call__(self, batch):
        if self.top is None and self.tsh is None:
            if hasattr(batch, "edge_index") and batch.edge_index is not None:
                return batch

            dev = batch.x.device if hasattr(batch, "x") and torch.is_tensor(batch.x) else torch.device("cpu")
            batch.edge_index = torch.empty((2, 0), dtype=torch.long, device=dev)
            if self.weighted:
                dtype = batch.x.dtype if hasattr(batch, "x") and torch.is_tensor(batch.x) else torch.float32
                batch.edge_weight = torch.empty((0,), dtype=dtype, device=dev)
            return batch

        if not hasattr(batch, "x"):
            raise AttributeError(
                "Batch must contain attribute `batch.x` of shape (B*N, N)."
            )

        total_rows, N = batch.x.shape
        if total_rows % N != 0:
            raise ValueError("batch.x rows not divisible by N; graphs have unequal sizes?")
        B = total_rows // N
        X = batch.x.view(B, N, N)
        dev = X.device

        i, j = torch.tril_indices(N, N, offset=-1, device=dev)
        vals = X[:, i, j] 

        
        if self.tsh is not None:
            
            mask = vals >= self.tsh
            edge_coords = mask.nonzero(as_tuple=False)
            batch_idx, edge_in_tri_idx = edge_coords[:, 0], edge_coords[:, 1]
            selected_i, selected_j = i[edge_in_tri_idx], j[edge_in_tri_idx]
            offset = batch_idx * N
            src, dst = selected_j + offset, selected_i + offset
            if self.weighted:
                weight_vals = vals[mask]
        else:
            
            k = int(self.top * vals.size(1))
            if k == 0:
                batch.edge_index = torch.empty((2, 0), dtype=torch.long, device=dev)
                if self.weighted:
                    dtype = batch.x.dtype if hasattr(batch, "x") and torch.is_tensor(batch.x) else torch.float32
                    batch.edge_weight = torch.empty((0,), dtype=dtype, device=dev)
                return batch
            _, idx = vals.topk(k, dim=1)
            i_sel = torch.gather(i.expand(B, -1), 1, idx)
            j_sel = torch.gather(j.expand(B, -1), 1, idx)
            offset = (torch.arange(B, device=dev) * N).unsqueeze(1)
            src = (j_sel + offset).flatten()
            dst = (i_sel + offset).flatten()
            if self.weighted:
                weight_vals = torch.gather(vals, 1, idx).flatten()

        
        edge_index = torch.stack((src, dst), dim=0)

        if edge_index.dtype != torch.long:
            edge_index = edge_index.to(torch.long)

        if edge_index.dim() == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.t().contiguous()

        rev_edge_index = edge_index[[1, 0], :]
        edge_index = torch.cat((edge_index, rev_edge_index), dim=1)

        if self.weighted:
            edge_weight = torch.cat((weight_vals, weight_vals), dim=0)

        if not self.weighted:
            edge_index = torch.unique(edge_index, dim=1)

        batch.edge_index = edge_index
        if self.weighted:
            batch.edge_weight = edge_weight
        return batch
    
class LowerTriFlattenBatch(nn.Module):

    def __init__(self, n: int, self_conv: bool = False, self_conv_adj: bool = False):
        super().__init__()
        r, c = torch.tril_indices(n, n, offset=-1)
        self.register_buffer("rows", r, persistent=False)
        self.register_buffer("cols", c, persistent=False)
        self.n = n
        self.self_conv = self_conv
        self.self_conv_adj = self_conv_adj

    def forward(self, data):
        x, batch_vec = data.x, data.batch
        B = int(batch_vec.max()) + 1
        x = x.view(B, self.n, self.n)

        
        if self.self_conv_adj:
            if not hasattr(data, "edge_index") or data.edge_index is None:
                raise AttributeError("data must contain 'edge_index' when self_conv_adj=True")

            edge_index = data.edge_index.to(x.device)
            if edge_index.numel() == 0:
                raise ValueError("edge_index is empty; cannot build adjacency matrix for self_conv_adj")

            sizes = (B * self.n, B * self.n)
            A = SparseTensor.from_edge_index(edge_index, sparse_sizes=sizes)

            x_flat = x.view(B * self.n, self.n)
            x_flat = sparse_matmul(A, x_flat)
            x = x_flat.to_dense().view(B, self.n, self.n)
        elif self.self_conv:
            x = torch.matmul(x, x)

        rows = torch.as_tensor(self.rows)
        cols = torch.as_tensor(self.cols)
        return x[:, rows, cols].contiguous()

class BatchFeatureTransform(BaseTransform):

    def __init__(self, feature_type: str = "corr"):
        super().__init__()
        if feature_type not in {"corr", "identity"}:
            raise ValueError("feature_type must be either 'corr' or 'identity'.")
        self.feature_type = feature_type

    def __call__(self, batch):
        if self.feature_type == "corr":
            return batch

        if not hasattr(batch, "x"):
            raise AttributeError("Batch must contain attribute 'x'.")

        total_rows, N = batch.x.shape
        if total_rows % N != 0:
            raise ValueError("batch.x rows not divisible by N; graphs have unequal sizes?")

        B = total_rows // N

        eye = torch.eye(N, dtype=batch.x.dtype, device=batch.x.device)

        batch.x = eye.repeat(B, 1)

        return batch