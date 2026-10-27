import torch
from pathlib import Path

from mini_gpt.models.gpt import GPT
from mini_gpt.trainer.input_producer import tokenizer

CHECKPOINT_PATH = Path("mini_gpt/checkpoints/minigpt_v1.pt")


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = GPT(
    vocabulary_size=tokenizer.vocab_size,
    embedding_dim=256,
    num_heads=8,
    num_layers=6,
    max_seq_len=128,
    ffn_hidden_dim=1024
).to(device)

checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

output = model.generate(
    "To be, or not to be, that is",
    tokenizer,
    max_new_tokens=200
)

print(output)