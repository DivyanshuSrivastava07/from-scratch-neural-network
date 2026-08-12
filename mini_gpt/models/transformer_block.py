import torch.nn as nn,torch
from mini_gpt.models.layer_norm import LayerNorm
from mini_gpt.models.multihead import MultiHeadAttention
from mini_gpt.models.feedforward import FeedForward
class TransformerBlock(nn.Module):
    def __init__(
            self,
            embedding_dim:int,
            max_seq_len:int,
            num_heads:int,
            ffn_hidden_dim:int
        ):
        super().__init__()
        self.norm1 = LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim,num_heads,max_seq_len)
        self.norm2 = LayerNorm(embedding_dim)
        self.ffn = FeedForward(embedding_dim,ffn_hidden_dim)
       
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))
        
    