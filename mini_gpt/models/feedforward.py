import torch.nn as nn,torch
class FeedForward(nn.Module):
    def __init__(self,embedding_dim:int,ffn_hidden_dim:int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim,ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim,embedding_dim)
        )
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
