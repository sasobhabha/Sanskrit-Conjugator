#!/usr/bin/env python
"""Quick train test to verify pipeline."""
import sys, os
sys.path.insert(0, 'src')
import torch
from fast_dataset import PreTokenizedDataset
from torch.utils.data import DataLoader, random_split
from model import CharacterTokenizer, SanskritVerbConjugator, save_model

# Load dataset
tensor_path = "data/real_training_tensors.pt"
dataset = PreTokenizedDataset(tensor_path)
train_len = int(0.8*len(dataset))
val_len = len(dataset)-train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

device = torch.device('cpu')
tokenizer = CharacterTokenizer(64)
model = SanskritVerbConjugator(tokenizer.vocab_size, tokenizer.vocab_size, embed_dim=128, hidden_dim=256, num_layers=1).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
crit = torch.nn.CrossEntropyLoss(ignore_index=0)

print(f"Train batches: {len(train_loader)}, val: {len(val_loader)}")
# Train 1 epoch quick
model.train()
for i, (src, tgt) in enumerate(train_loader):
    src, tgt = src.to(device), tgt.to(device)
    opt.zero_grad()
    out = model(src, tgt)
    loss = crit(out[:,1:].reshape(-1, out.shape[-1]), tgt[:,1:].reshape(-1))
    loss.backward()
    opt.step()
    if i % 50 == 0:
        print(f"  Batch {i}, loss {loss.item():.4f}")
    if i == 200:
        break

# Validate quick
model.eval()
val_loss = 0
with torch.no_grad():
    for j, (src, tgt) in enumerate(val_loader):
        out = model(src, tgt)
        loss = crit(out[:,1:].reshape(-1, out.shape[-1]), tgt[:,1:].reshape(-1))
        val_loss += loss.item()
        if j == 20: break
print(f"Avg val loss: {val_loss/20:.4f}")

save_model(model, "models/test_model.pt")
print("Test model saved")
