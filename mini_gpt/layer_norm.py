import torch.nn as nn,torch
class LayerNorm(nn.Module):
    def __init__(self,embedding_dim=256,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))
    def forward(self,x):
        mean = x.mean(
            dim=-1,
            keepdim=True
        )
        var = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False
        )
        x_hat = (x - mean)/torch.sqrt(var + self.eps)
        output = self.gamma * x_hat + self.beta
        return output