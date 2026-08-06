import torch.nn as nn
import torch
class Embedding(nn.Module):
    def __init__(self,vocabulary_size:int,embedding_dim:int):
        super().__init__()
        assert vocabulary_size > 0
        assert embedding_dim > 0
        self.weight = nn.Parameter(
            torch.randn(vocabulary_size,embedding_dim)
        )
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        return self.weights[x]
        
