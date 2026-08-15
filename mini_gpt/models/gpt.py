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
        self.max_seq_len = max_seq_len
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
    @torch.no_grad()
    def generate(self,text,tokenizer,max_new_tokens = 100):
        self.eval()
        tokens = tokenizer.encode(text)

        input_ids = torch.tensor(
            [tokens],
            dtype=float,
            device=next(self.parameters()).device
        )
        for _ in range(max_new_tokens):

            # Keep only the context the model can handle
            context = input_ids[:, -self.max_seq_len:]

            # Forward pass
            logits = self(context)

            # We only need prediction for the LAST token
            logits = logits[:, -1, :]

            # Greedy decoding
            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

            # Append predicted token
            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1
            )

        # Token IDs → text
        return tokenizer.decode(
            input_ids[0].tolist()
        )