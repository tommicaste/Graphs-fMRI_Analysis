import torch
import torch.nn as nn

def sincos_pe(x: torch.Tensor) -> torch.Tensor:
    """
    Add vanilla sine–cosine positional encoding
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)

    B, T, d = x.shape
    device = x.device
    dtype  = x.dtype

    
    pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)          
    
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=dtype) *
                    (-torch.log(torch.tensor(10000.0, dtype=dtype)) / d))  

    pe = torch.zeros(T, d, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div)   
    pe[:, 1::2] = torch.cos(pos * div)   

    x = x + pe                          
    return x if x.dim() == 3 else x.squeeze(0)

class TimeSeriesTransformer(nn.Module):
    """
    * One real-valued vector per timestep → class logits.
    * Uses a prepended learnable [CLS] token for pooling.
    """
    def __init__(
        self,
        input_dim: int = 347,          
        num_classes: int = 4,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj  = nn.Linear(input_dim, d_model)
        self.cls_tok  = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer     = nn.TransformerEncoderLayer(
                            d_model=d_model,
                            nhead=num_heads,
                            dim_feedforward=4 * d_model,
                            dropout=dropout,
                            batch_first=True,
                        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers)
        self.head     = nn.Sequential(
                            nn.LayerNorm(d_model),
                            nn.Linear(d_model, num_classes)
                        )

    def forward(self, x):                
        B, T, _ = x.shape
        x = self.in_proj(x)              

        cls = self.cls_tok.expand(B, -1, -1)   
        x   = torch.cat([cls, x], dim=1)       
        x   = sincos_pe(x)                 
        h   = self.encoder(x)                 

        return self.head(h[:, 0])