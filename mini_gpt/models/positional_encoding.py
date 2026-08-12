import torch.nn as nn
import torch
import math
class PositionalEncoding(nn.Module):
    def __init__(self,embedding_dim:int,max_seq_length:int):
        super().__init__()
        pe = torch.zeros(max_seq_length,embedding_dim)
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0,
            embedding_dim,
            2,
            dtype=torch.float
        )*(-math.log(10000)/embedding_dim))
        pe[:, 0::2] = torch. sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape:
        (Batch, Sequence Length, Embedding Dimension)
        """

        seq_len = x.size(1)

        return x + self.pe[:, :seq_len]