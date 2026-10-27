
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mini_gpt.models.gpt import GPT
from mini_gpt.trainer.input_producer import train_dataset, tokenizer


# --------------------------------------------------
# Configuration: keep identical to MiniGPT v1
# --------------------------------------------------

BATCH_SIZE = 32
WARMUP_BATCHES = 3
MEASURED_BATCHES = 20

EMBEDDING_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
CONTEXT_LENGTH = 128
FFN_HIDDEN_DIM = 1024

CHECKPOINT_PATH = Path("mini_gpt/checkpoints/minigpt_v1.pt")
REPORT_PATH = Path("mini_gpt/reports/minigpt_v1_baseline.json")


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}"
        )

    # 1. Select device.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Load the existing dataset.
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Reconstruct the same model architecture.
    model = GPT(
        vocabulary_size=tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=CONTEXT_LENGTH,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
    ).to(device)

    # 4. Restore the saved model parameters.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    print("Checkpoint loaded successfully.")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Prepare the benchmark.
    iterator = iter(loader)
    losses = []
    total_tokens = 0

    def next_batch():
        nonlocal iterator

        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            return next(iterator)

    def prepare_batch(batch):
        input_ids, targets = batch

        input_ids = input_ids.to(
            device, non_blocking=True
        )
        targets = targets.to(
            device, non_blocking=True
        )

        return input_ids, targets

    # 6. Warm up before measuring.
    # This reduces the impact of initial CUDA overhead.
    with torch.inference_mode():
        for _ in range(WARMUP_BATCHES):
            input_ids, targets = prepare_batch(next_batch())

            logits = model(input_ids)

            loss = criterion(
                logits.reshape(-1, tokenizer.vocab_size),
                targets.reshape(-1),
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    # 7. Measure steady-state throughput.
    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(MEASURED_BATCHES):
            input_ids, targets = prepare_batch(next_batch())

            logits = model(input_ids)

            loss = criterion(
                logits.reshape(-1, tokenizer.vocab_size),
                targets.reshape(-1),
            )

            losses.append(loss.detach())
            total_tokens += targets.numel()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_seconds = time.perf_counter() - start

    mean_loss = torch.stack(losses).mean().item()
    tokens_per_second = total_tokens / elapsed_seconds
    batches_per_second = MEASURED_BATCHES / elapsed_seconds

    # 8. Record a reproducible baseline report.
    report = {
        "checkpoint": str(CHECKPOINT_PATH),
        "evaluation_split": "training_data_smoke_benchmark",
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "vocabulary_size": tokenizer.vocab_size,
        "batch_size": BATCH_SIZE,
        "context_length": CONTEXT_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "ffn_hidden_dim": FFN_HIDDEN_DIM,
        "parameter_count": sum(
            p.numel() for p in model.parameters()
        ),
        "warmup_batches": WARMUP_BATCHES,
        "measured_batches": MEASURED_BATCHES,
        "tokens_processed": total_tokens,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "tokens_per_second": round(tokens_per_second, 2),
        "batches_per_second": round(batches_per_second, 3),
        "mean_training_batch_loss": round(mean_loss, 6),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n--- MiniGPT v1 Baseline ---")
    print(f"Mean training batch loss: {mean_loss:.6f}")
    print(f"Tokens processed:         {total_tokens:,}")
    print(f"Elapsed time:             {elapsed_seconds:.2f} seconds")
    print(f"Tokens/second:            {tokens_per_second:,.2f}")
    print(f"Batches/second:           {batches_per_second:.3f}")
    print(f"Report saved:             {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
