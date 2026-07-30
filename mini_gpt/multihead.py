import torch.nn as nn,torch
import math
class Multihead(nn.Module):
    def __init__(self,embedding_dim,heads,max_seq_len):
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
        self.mask = self.register_buffer(
            "mask",
            torch.tril(
                torch.ones_like(
                    max_seq_len,max_seq_len
                )
            )
        )
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        B, S, E = x.shape
        mask = self.mask[:S, :S]
        Q, K, V = qkv.chunk(3,dim=-1)

        Q = Q.view(B,S,self.heads,self.head_dim).transpose(2,1)
        K = K.view(B,S,self.heads,self.head_dim).transpose(2,1)
        V = V.view(B,S,self.heads,self.head_dim).transpose(2,1)

        scores = (Q @ K.transpose(-2,-1))/math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            mask==0,
            float("-inf")
        )
        weights = torch.softmax(scores,dim=-1)

        # if(self.training):
        #     self.attention = weights        #### Usually we only store attention weights when debugging or visualizing.
        weights = self.attn_dropout(weights)

        """During training, some attention probabilities are randomly zeroed out.
        This prevents the model from becoming overly dependent on a few specific attention paths,
        improving generalization."""
        out = weights @ V
        out = out.transpose(2,1).contiguous().view(B,S,E)
        out = self.out(out)
        return out

