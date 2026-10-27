from torch.utils.data import DataLoader
from mini_gpt.models.gpt import GPT
import torch.nn as nn, torch
from mini_gpt.trainer.input_producer import train_dataset,tokenizer
from pathlib import Path 

CHECKPOINT_PATH = Path("mini_gpt/checkpoints/minigpt_v1.pt")

def main():
    loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=2
    )
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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4
    )

    criterion = nn.CrossEntropyLoss()
    epochs = 8

    for epoch in range(epochs):
        running_loss = 0.0

        for input_ids, targets in loader:
            input_ids = input_ids.to(
                device,
                non_blocking=True,
            )
            targets = targets.to(
                device,
                non_blocking=True
            )


            logits = model(input_ids)
            optimizer.zero_grad()

            loss = criterion(
                logits.view(-1, tokenizer.vocab_size),
                targets.view(-1)
            )

            loss.backward()

            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Loss: {avg_loss:.4f}"
        )
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state[_dict": optimizer.state_dict(),
            "loss": avg_loss,
        },
        "mini_gpt/checkpoints/minigpt_v1.pt"
    )
if __name__ == "__main__":
    main()