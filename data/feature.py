from torch_geometric.data import Data

def feature_corr(data_list):
    """
    Move connectivity matrix to metadata and use it as node features.
    """
    for d in data_list:
        # Set node features
        d.x = d.c
        # Preserve original connectivity in metadata
        d.metadata['c'] = d.c
        # Remove the original c attribute
        delattr(d, 'c')
        # Drop any other graph attributes (e.g., labels)
        for attr in list(d.keys):
            if attr not in ('x', 'edge_index', 'metadata'):
                delattr(d, attr)
        # Verify only the desired attributes remain
        assert set(d.keys) == {'x', 'edge_index', 'metadata'}, \
            f"Unexpected attributes {set(d.keys)} in Data object"
    return data_list
