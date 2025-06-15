import torch
from functools import lru_cache
from tqdm.auto import tqdm

def add_edge_list(data_list, tsh: float = None, top: float = 0.10, verbose: bool = True):
    """
    Add an edge_index to each Data in data_list by thresholding its c matrix (hard or top-percentile).
    """
    @lru_cache(maxsize=None)
    def _get_tril_indices(n, device):
        return torch.tril_indices(n, n, offset=-1, device=device)

    def _process(corr_mat):
        n, device = corr_mat.size(0), corr_mat.device
        i, j = _get_tril_indices(n, device)
        vals = corr_mat[i, j]

        if tsh is None:
            k = int(top * vals.numel())
            if k == 0:
                return torch.empty((2, 0), dtype=torch.long, device=device)
            _, idx = torch.topk(vals, k)
            i_sel, j_sel = i[idx], j[idx]
        else:
            mask = vals >= tsh
            i_sel, j_sel = i[mask], j[mask]

        return torch.stack([j_sel, i_sel], dim=0)

    iterator = tqdm(data_list, desc="add_edge_list", disable=not verbose)
    for d in iterator:
        assert hasattr(d, 'x'), "Data object must have a 'x' attribute"
        edge_index = _process(d.x)
        d.edge_index = edge_index

    return data_list
