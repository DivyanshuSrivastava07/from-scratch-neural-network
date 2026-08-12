import torch.nn as nn,torch
import math
class MultiHeadAttention(nn.Module):
    def __init__(self,embedding_dim:int,num_heads:int,max_seq_len:int,dropout:float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = self.embedding_dim // self.num_heads
        assert embedding_dim % num_heads == 0, (
            "embedding_dim must be divisible by num_heads"
        )
        self.qkv = nn.Linear(embedding_dim,3*embedding_dim)
        self.out = nn.Linear(
            embedding_dim,
            embedding_dim
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    max_seq_len,max_seq_len
                )
            )
        )
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        B, S, E = x.shape
        mask = self.mask[:S, :S]
        Q, K, V = qkv.chunk(3,dim=-1)

        Q = Q.view(B,S,self.num_heads,self.head_dim).transpose(2,1)
        K = K.view(B,S,self.num_heads,self.head_dim).transpose(2,1)
        V = V.view(B,S,self.num_heads,self.head_dim).transpose(2,1)

        scores = (Q @ K.transpose(-2,-1))/math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            mask==0,
            float("-inf")
        )
        weights = torch.softmax(scores,dim=-1)

        weights = self.attn_dropout(weights)

        """During training, some attention probabilities are randomly zeroed out.
        This prevents the model from becoming overly dependent on a few specific attention paths,
        improving generalization."""
        out = weights @ V
        out = out.transpose(2,1).contiguous().view(B,S,E)
        return self.out(out)
        

