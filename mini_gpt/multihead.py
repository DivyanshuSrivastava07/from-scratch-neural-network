import torch.nn as nn,torch
import math
class SelfAttention(nn.Module):
    def __init__(self,embedding_dim,heads):
        super().__init__()
        self.heads = heads
        self.embedding_dim = embedding_dim
        self.head_dim = self.embedding_dim // self.heads
        assert self.embedding_dim % self.heads == 0
        self.qkv = nn.Linear(embedding_dim,3*embedding_dim)
        self.out = nn.Linear(
            embedding_dim,
            embedding_dim
        )
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        B, S, E = x.shape
        Q, K, V = qkv.chunk(3,dim=-1)

        Q = Q.view(B,S,self.heads,self.head_dim).transpose(2,1)
        K = K.view(B,S,self.heads,self.head_dim).transpose(2,1)
        V = V.view(B,S,self.heads,self.head_dim).transpose(2,1)

        scores = (Q @ K.transpose(-2,-1))/math.sqrt(self.head_dim)
        weights = torch.softmax(scores,dim=-1)
        self.attention = weights
        out = weights @ V
        out = out.transpose(2,1).contiguous().view(B,S,E)
        out = self.out(out)
        return out

