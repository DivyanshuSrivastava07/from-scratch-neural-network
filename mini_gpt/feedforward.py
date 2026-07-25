import torch.nn as nn
class FeedForward(nn.Module):
    def __init__(self,embedding_dim=356):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim,4*embedding_dim),
            nn.GELU(),
            nn.Linear(4*embedding_dim,embedding_dim)
        )
    def forward(self,x):
        return self.ffn(x)
