#!/usr/bin/env python
"""Minimal training loop test for Sanskrit conjugation model."""
import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import CharacterTokenizer, SanskritVerbConjugator, VerbConjugationDataset

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded

print("Loading dataset...")
tokenizer = CharacterTokenizer(max_length=32)
dataset = VerbConjugationDataset("data/training_pairs.json", tokenizer)
print(f"Full dataset: {len(dataset)} samples")
# Small subset
small_dataset = torch.utils.data.Subset(dataset, range(100))
loader = DataLoader(small_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

device = torch.device('cpu')
model = SanskritVerbConjugator(
    src_vocab_size=tokenizer.vocab_size,
    tgt_vocab_size=tokenizer.vocab_size,
    embed_dim=64,
    hidden_dim=128,
    num_layers=1,
    dropout=0.1
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting minimal training (2 epochs)...")
for epoch in range(2):
    model.train()
    total_loss = 0
    for i, (src, tgt) in enumerate(loader):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)   # [B, T, V]
        loss = criterion(output[:,1:].reshape(-1, output.shape[-1]), tgt[:,1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print(f"  Batch {i}, loss={loss.item():.4f}")
    print(f"Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}")

print("Training completed OK!")
