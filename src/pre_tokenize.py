#!/usr/bin/env python
"""Pre-tokenize real training pairs into tensors for fast loading."""

import torch
import json
import os
from model import CharacterTokenizer

def main():
    # Initialize tokenizer
    tokenizer = CharacterTokenizer(max_length=64)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load pairs JSON
    pairs_file = "data/real_training_pairs.json"
    print(f"Loading {pairs_file}...")
    with open(pairs_file, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} pairs")

    # Encode all source and target strings
    print("Encoding source strings...")
    src_tokens_list = [tokenizer.encode(pair['source']) for pair in pairs]

    print("Encoding target strings...")
    tgt_tokens_list = [tokenizer.encode(pair['target']) for pair in pairs]

    # Stack into tensors
    src_tensor = torch.tensor(src_tokens_list, dtype=torch.long)  # [N, max_len]
    tgt_tensor = torch.tensor(tgt_tokens_list, dtype=torch.long)  # [N, max_len]

    print(f"\nProcessed {len(pairs)} pairs")
    print(f"src_tensor shape: {src_tensor.shape}")
    print(f"tgt_tensor shape: {tgt_tensor.shape}")

    # Save to file
    output_file = "data/real_training_tensors.pt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save({
        'src': src_tensor,
        'tgt': tgt_tensor,
        'tokenizer_vocab_size': tokenizer.vocab_size,
    }, output_file)
    print(f"\nSaved tensors to {output_file}")

    # Verify the file
    loaded = torch.load(output_file)
    print(f"Verification - src shape: {loaded['src'].shape}, tgt shape: {loaded['tgt'].shape}")
    print(f"Verification - vocab size: {loaded['tokenizer_vocab_size']}")

if __name__ == "__main__":
    main()
