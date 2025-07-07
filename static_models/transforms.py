import torch
from torch_geometric.transforms import BaseTransform
import torch.nn as nn
from torch_sparse import SparseTensor, matmul as sparse_matmul

class BatchEdgeListTransform(BaseTransform):
    """
    Build a single `edge_index` for a PyG Batch from stacked correlation
    matrices in `batch.x`.
    """

    def __init__(self, *, top: float | None = None, tsh: float | None = None, weighted: bool = False):
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
        self.weighted = weighted

    def __call__(self, batch):
        # Passthrough behaviour when no edge creation strategy is specified
        if self.top is None and self.tsh is None:
            # When an edge list already exists and is not None → leave it.
            if hasattr(batch, "edge_index") and batch.edge_index is not None:
                return batch

            # Otherwise create an empty edge list with correct dtype/shape.
            dev = batch.x.device if hasattr(batch, "x") and torch.is_tensor(batch.x) else torch.device("cpu")
            batch.edge_index = torch.empty((2, 0), dtype=torch.long, device=dev)
            if self.weighted:
                # Create an empty edge_weight tensor with the same dtype as the input features (if available)
                dtype = batch.x.dtype if hasattr(batch, "x") and torch.is_tensor(batch.x) else torch.float32
                batch.edge_weight = torch.empty((0,), dtype=dtype, device=dev)
            return batch

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
            if self.weighted:
                weight_vals = vals[mask]
        else:
            
            k = int(self.top * vals.size(1))
            if k == 0:
                batch.edge_index = torch.empty((2, 0), dtype=torch.long, device=dev)
                if self.weighted:
                    # Create an empty edge_weight tensor with the same dtype as the input features (if available)
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

        # Ensure undirected edges by adding reverse directions --------------
        rev_edge_index = edge_index[[1, 0], :]
        edge_index = torch.cat((edge_index, rev_edge_index), dim=1)

        # Handle edge weights if requested --------------------------------
        if self.weighted:
            edge_weight = torch.cat((weight_vals, weight_vals), dim=0)

        # Remove potential duplicate edges (optional but safer) -------------
        if not self.weighted:  # keep alignment between edge_index and edge_weight if present
            edge_index = torch.unique(edge_index, dim=1)

        batch.edge_index = edge_index
        if self.weighted:
            batch.edge_weight = edge_weight
        return batch
    
class LowerTriFlattenBatch(nn.Module):
    """
    PyG batch of symmetric N × N matrices:
        data.x      (B·N, N)
        data.batch  (B·N,)
    returns
        (B, N(N−1)//2)   strictly lower triangular, diagonal excluded
    """
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

            # Build a sparse adjacency for the whole batch (B*N, B*N)
            sizes = (B * self.n, B * self.n)
            A = SparseTensor.from_edge_index(edge_index, sparse_sizes=sizes)

            # Multiply: (B*N, B*N) @ (B*N, N) → (B*N, N)
            x_flat = x.view(B * self.n, self.n)
            x_flat = sparse_matmul(A, x_flat)
            x = x_flat.view(B, self.n, self.n)
        elif self.self_conv:
            x = torch.matmul(x, x)

        return x[:, self.rows, self.cols].contiguous()

class BatchFeatureTransform(BaseTransform):
    """Modify ``batch.x`` for an entire PyG ``Batch``.

    Parameters
    ----------
    feature_type : str, optional (default="corr")
        • "corr"     – keep input correlation matrices unchanged.
        • "identity" – replace every graph's feature matrix with an *identity* matrix
          of matching size.

    The transform expects ``batch.x`` to contain a *stack* of square matrices:

    (B·N, N) where B is the batch size and N is the node count per graph.
    """

    def __init__(self, feature_type: str = "corr"):
        super().__init__()
        if feature_type not in {"corr", "identity"}:
            raise ValueError("feature_type must be either 'corr' or 'identity'.")
        self.feature_type = feature_type

    def __call__(self, batch):
        # Early exit if no change required
        if self.feature_type == "corr":
            return batch

        # From here on we know we must build identity matrices
        if not hasattr(batch, "x"):
            raise AttributeError("Batch must contain attribute 'x'.")

        total_rows, N = batch.x.shape
        if total_rows % N != 0:
            raise ValueError("batch.x rows not divisible by N; graphs have unequal sizes?")

        B = total_rows // N

        # Build identity matrix (N, N) once on the correct device / dtype
        eye = torch.eye(N, dtype=batch.x.dtype, device=batch.x.device)

        # Repeat B times and reshape to (B·N, N)
        batch.x = eye.repeat(B, 1)

        return batch