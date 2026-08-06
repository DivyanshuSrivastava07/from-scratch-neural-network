import torch.nn as nn,torch
import math
class SelfAttention(nn.Module):
    def __init__(self,embedding_dim,d_k):
        super().__init__()
        self.d_k = d_k
        self.wq = nn.Linear(embedding_dim,d_k)
        self.wk = nn.Linear(embedding_dim,d_k)
        self.wv = nn.Linear(embedding_dim,d_k)
        # self.qkv = nn.Linear(embedding_dim,3*embedding_dim)

    def forward(self,x : torch.Tensor) -> torch.Tensor:
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        # qkv = self.qkv(x)
        # Q, K, V = qkv.chunk(3,dim=-1)
        scores = (Q @ K.transpose(-2,-1))/math.sqrt(self.d_k)
        weights = torch.softmax(scores,dim=-1)
        self.attention = weights
        return weights @ V
        