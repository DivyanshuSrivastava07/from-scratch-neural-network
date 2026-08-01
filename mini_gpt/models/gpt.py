import torch.nn as nn
import mini_gpt.models.embedding as embedding,mini_gpt.models.positional_encoding as positional_encoding,mini_gpt.models.layer_norm as layer_norm,multihead,mini_gpt.models.transformer_block as transformer_block
class GPT(nn.Module):
    def __init__(
            self,
            num_layers,
            embedding_dim,
            vocab_size,
            max_seq_len,
            heads,
            ffn_hidden_dim
            ):
        super().__init__()
        self.embedding = embedding.Embedding(vocab_size,embedding_dim)
        self.pos_encoding = positional_encoding.PositionalEncoding(embedding,max_seq_len)
        self.blocks = nn.ModuleList(
            [
                transformer_block.TransformerBlock(
                    embedding_dim,max_seq_len,heads,ffn_hidden_dim
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = layer_norm.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )
    def forward(self,input_ids):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits