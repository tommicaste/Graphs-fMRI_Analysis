import torch
from torch_geometric.transforms import BaseTransform


class BatchEdgeListTransform(BaseTransform):
    """
    Build a single `edge_index` for a PyG Batch whose stacked correlation
    matrices live in `batch.x` (shape = (B*N, N)).

    Parameters
    ----------
    top : float, default 0.10
        Keep the strongest `top` fraction of strict lower triangle entries
        in each graph. Ignored when `tsh` is not None.
    tsh : float | None, default None
        Hard threshold; keep every entry ≥ `tsh`.
        If None, the `top` policy is used.
    """

    def __init__(self, *, top: float = 0.10, tsh: float | None = None):
        super().__init__()
        if tsh is None and not (0.0 < top <= 1.0):
            raise ValueError("`top` must be in (0, 1] when `tsh` is None.")
        self.top = top
        self.tsh = tsh

    # ------------------------------------------------------------------
    def __call__(self, batch):
        if not hasattr(batch, "x"):
            raise AttributeError(
                "Batch must contain attribute `batch.x` of shape (B*N, N)."
            )

        # reshape to (B, N, N)
        total_rows, N = batch.x.shape
        if total_rows % N != 0:
            raise ValueError("batch.x rows not divisible by N; graphs have unequal sizes?")
        B = total_rows // N
        X = batch.x.view(B, N, N)  # (B, N, N)
        dev = X.device

        # strict lower triangle indices (shared)
        i, j = torch.tril_indices(N, N, offset=-1, device=dev)  # (E,), E = N*(N-1)/2
        vals = X[:, i, j]  # (B, E)

        # choose edges per graph
        if self.tsh is None:
            k = int(self.top * vals.size(1))
            if k == 0:
                batch.edge_index = torch.empty((2, 0), dtype=torch.long, device=dev)
                return batch
            _, idx = vals.topk(k, dim=1)  # (B, k)
            i_sel = torch.gather(i.expand(B, -1), 1, idx)  # (B, k)
            j_sel = torch.gather(j.expand(B, -1), 1, idx)  # (B, k)
        else:
            mask = vals >= self.tsh  # (B, E)
            i_sel = [i[m] for m in mask]  # ragged
            j_sel = [j[m] for m in mask]

        # node offsets so indices refer to the concatenated node tensor
        offset = (torch.arange(B, device=dev) * N).unsqueeze(1)  # (B, 1)

        if self.tsh is None:
            src = (j_sel + offset).flatten()  # [E_total]
            dst = (i_sel + offset).flatten()
        else:
            src = torch.cat([j_sel[b] + offset[b, 0] for b in range(B)])
            dst = torch.cat([i_sel[b] + offset[b, 0] for b in range(B)])

        batch.edge_index = torch.stack((src, dst), dim=0)  # (2, E_total)

        # ensure `batch.batch` is valid
        if not hasattr(batch, "batch") or batch.batch.max() == 0:
            batch.batch = torch.arange(B, device=dev).repeat_interleave(N)
        return batch
