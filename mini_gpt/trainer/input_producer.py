from mini_gpt.tokenizer.char_tokenizer import CharacterTokenizer
from mini_gpt.datasets.dataset import TextDataset

with open("mini_gpt/data/input.txt","r",encoding="utf-8") as f:
    corpus = f.read()

tokenizer = CharacterTokenizer(corpus)
tokens = tokenizer.encode(corpus)


split = int(0.9 * len(tokens))

train_tokens = tokens[:split]
val_tokens = tokens[split:]

train_dataset = TextDataset(
    train_tokens,
    128,
    128
)
val_dataset = TextDataset(
    val_tokens,
    128,
    128
)
