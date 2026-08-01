import torch.nn as nn
import torch
class Embedding(nn.Module):
    def __init__(self,vocabulary=10000,embedding_dim=256):
        super().__init__()
        self.vocabulary = vocabulary
        self.embedding_dim = embedding_dim
        self.weights = nn.Parameter(
            torch.randn(self.vocabulary,self.embedding_dim)
        )
    def forward(self,x):
        output = self.weights[x]
        return output
