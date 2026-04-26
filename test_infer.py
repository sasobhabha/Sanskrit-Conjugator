#!/usr/bin/env python
"""Test a forward pass with model integration."""
import sys
sys.path.insert(0, 'src')
import torch
from model import CharacterTokenizer, SanskritVerbConjugator

tokenizer = CharacterTokenizer(max_length=32)
model = SanskritVerbConjugator(
    src_vocab_size=tokenizer.vocab_size,
    tgt_vocab_size=tokenizer.vocab_size,
    embed_dim=64, hidden_dim=128, num_layers=1
)
model.eval()

# Condition source: "gacch|lata|prathama_ekavachana"
src_text = "gacch|lata|prathama_ekavachana"
src = tokenizer.encode(src_text)
src_tensor = torch.tensor([src], dtype=torch.long)

preds = model.predict(src_tensor, max_len=32)
form = tokenizer.decode(preds[0].tolist())
print(f"Source: {src_text}")
print(f"Predicted form: {form}")
print("Done!")
