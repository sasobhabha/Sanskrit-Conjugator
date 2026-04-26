#!/usr/bin/env python
"""Quick test of model forward pass."""
import sys
sys.path.insert(0, 'src')

import torch
from model import CharacterTokenizer, SanskritVerbConjugator, VerbConjugationDataset
from torch.utils.data import DataLoader

print("Loading small dataset...")
tokenizer = CharacterTokenizer(max_length=32)
dataset = VerbConjugationDataset("data/training_pairs.json", tokenizer)
print(f"Dataset size: {len(dataset)}")

# Take a tiny subset
small_dataset = torch.utils.data.Subset(dataset, range(20))
loader = DataLoader(small_dataset, batch_size=4, shuffle=False)

device = torch.device('cpu')
model = SanskritVerbConjugator(
    src_vocab_size=tokenizer.vocab_size,
    tgt_vocab_size=tokenizer.vocab_size,
    embed_dim=64,
    hidden_dim=128,
    num_layers=1,
    dropout=0.1
).to(device)

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
print("Running one batch forward pass...")

src, tgt = next(iter(loader))
print(f"  src shape: {src.shape}, tgt shape: {tgt.shape}")

output = model(src, tgt)
print(f"  output shape: {output.shape}")
print("OK!")
