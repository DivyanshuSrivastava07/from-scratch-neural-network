import torch.nn as nn
from layer_norm import LayerNorm
from multihead import MultiHead
from feedforward import FeedForward
class TransformerBlock(nn):
    def __init__(self):
        super().__init__()
        self.norm1 = LayerNorm(...)
        self.attn = MultiHead(...)
        self.norm2 = LayerNorm(...)
        self.ffn = FeedForward(...)