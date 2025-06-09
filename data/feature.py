from torch_geometric.data import Data

def feature_corr(data_list):
    """
    Move connectivity matrix to metadata and use it as node features.
    """
    for d in data_list:
        d.x = d.c
        d.metadata['c'] = d.c
        delattr(d, 'c')
        for attr in list(d.keys()):
            if attr not in ('x', 'edge_index', 'metadata', 'y'):
                delattr(d, attr)
        assert all(k in d for k in ('x', 'metadata', 'y')), \
            f"Missing required attributes in Data object: {set(d.keys())}"
        assert set(d.keys()).issubset({'x', 'edge_index', 'metadata', 'y'}), \
            f"Unexpected attributes {set(d.keys())} in Data object"
    return data_list

