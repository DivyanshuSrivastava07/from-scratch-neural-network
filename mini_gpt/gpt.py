import torch.nn as nn
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=50000,
            embedding_dim=256
        )
    def forward(self,x):
        return self.embedding(x)
    