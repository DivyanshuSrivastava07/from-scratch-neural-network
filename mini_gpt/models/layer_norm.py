import torch.nn as nn,torch
class LayerNorm(nn.Module):
    def __init__(self,embedding_dim:int,eps=1e-5):
        super().__init__()
        assert eps > 0
        assert embedding_dim > 0
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        mean = x.mean(
            dim=-1,
            keepdim=True
        )
        variance = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False
        )
        x_hat = (x - mean)/torch.sqrt(variance + self.eps)
        return self.gamma * x_hat + self.beta
        