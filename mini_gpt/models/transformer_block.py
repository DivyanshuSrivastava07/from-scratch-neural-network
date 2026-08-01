import torch.nn as nn
from mini_gpt.models.layer_norm import LayerNorm
from mini_gpt.models.multihead import MultiHead
from mini_gpt.models.feedforward import FeedForward
class TransformerBlock(nn.Module):
    def __init__(
            self,
            embedding_dim,
            max_seq_len,
            heads,
            hidden_dim
        ):
        super().__init__()
        self.norm1 = LayerNorm(embedding_dim)
        self.attention = MultiHead(embedding_dim,heads,max_seq_len)
        self.norm2 = LayerNorm(embedding_dim)
        self.ffn = FeedForward(embedding_dim,hidden_dim)
       
    def forward(self,x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    