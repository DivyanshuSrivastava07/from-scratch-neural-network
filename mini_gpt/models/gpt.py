import torch.nn as nn,torch
import mini_gpt.models.embedding as embedding
import mini_gpt.models.positional_encoding as positional_encoding
import mini_gpt.models.layer_norm as layer_norm
import mini_gpt.models.transformer_block as transformer_block
class GPT(nn.Module):
    def __init__(
            self,
            num_layers:int,
            embedding_dim,
            vocabulary_size,
            max_seq_len,
            num_heads,
            ffn_hidden_dim
            ):
        super().__init__()
        assert embedding_dim > 0

        assert num_heads > 0

        assert max_seq_len > 0
        self.embedding = embedding.Embedding(vocabulary_size,embedding_dim)
        self.pos_encoding = positional_encoding.PositionalEncoding(embedding_dim,max_seq_len)
        self.blocks = nn.ModuleList(
            [
                transformer_block.TransformerBlock(
                    embedding_dim,max_seq_len,num_heads,ffn_hidden_dim
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = layer_norm.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(
            embedding_dim,
            vocabulary_size
        )
    def forward(self,input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)
       