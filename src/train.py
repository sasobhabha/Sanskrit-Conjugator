# Training script for Sanskrit verb conjugation model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import argparse
import os
from tqdm import tqdm
import numpy as np

from model import CharacterTokenizer, SanskritVerbConjugator, VerbConjugationDataset, save_model, load_model
from data_generator import build_full_conjugations
from fast_dataset import PreTokenizedDataset

def collate_fn(batch):
    """Custom collate function for padding sequences"""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded

def train_epoch(model: SanskritVerbConjugator, dataloader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: torch.device) -> float:
    """Train model for one epoch"""
    model.train()
    total_loss = 0

    for src, tgt in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()
        output = model(src, tgt)

        # Reshape output and target for loss calculation
        output = output[:, 1:].reshape(-1, output.shape[-1])
        tgt = tgt[:, 1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(model: SanskritVerbConjugator, dataloader: DataLoader,
            criterion: nn.Module, device: torch.device) -> float:
    """Validate model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Validating"):
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt)
            output = output[:, 1:].reshape(-1, output.shape[-1])
            tgt = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def train_model(args):
    """Main training loop"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check for pre-tokenized tensors (fast path)
    tensor_path = "data/real_training_tensors.pt"
    if os.path.exists(tensor_path) and not args.regenerate:
        print(f"Loading pre-tokenized tensors from {tensor_path}")
        dataset = PreTokenizedDataset(tensor_path)
        # Split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"Training samples: {len(train_dataset)}  Validation: {len(val_dataset)}")
    else:
        # Fallback: use slower JSON-based dataset
        tokenizer = CharacterTokenizer(max_length=64)
        data_file = "data/real_training_pairs.json"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Training data not found: {data_file}. Run build_real_dataset.py first.")
        dataset = VerbConjugationDataset(data_file, tokenizer)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        print(f"Training samples: {len(train_dataset)}  Validation: {len(val_dataset)}")

    # Initialize model
    tokenizer_temp = CharacterTokenizer(max_length=64)  # for vocab size
    model = SanskritVerbConjugator(
        src_vocab_size=tokenizer_temp.vocab_size,
        tgt_vocab_size=tokenizer_temp.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Training loop
    best_val_loss = float('inf')
    os.makedirs("models", exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"models/verb_conjugator_best.pt"
            save_model(model, save_path)
            print(f"Saved best model to {save_path}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_path = f"models/verb_conjugator_epoch_{epoch+1}.pt"
            save_model(model, save_path)

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train Sanskrit verb conjugation model")
    parser.add_argument("--num-verbs", type=int, default=3009,
                       help="Number of verb roots (ignored, uses all real data)")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size")
    parser.add_argument("--embed-dim", type=int, default=256,
                       help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=512,
                       help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2,
                       help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--regenerate", action="store_true",
                       help="Regenerate training data from source")

    args = parser.parse_args()

    # Check for CUDA
    if torch.cuda.is_available():
        print("CUDA available, using GPU acceleration")
    else:
        print("CUDA not available, using CPU")

    model = train_model(args)

    # Quick inference test
    print("\n" + "="*60)
    print("Testing trained model on sample verbs:")
    test_verbs = ["gacch", "kri", "bhu", "vad", "pa"]
    for verb in test_verbs:
        print(f"\nVerb: {verb}")
        for lakara in ["lata", "lit", "lrt"]:
            try:
                conjugations = build_full_conjugations(verb)
                if lakara in conjugations["conjugations"]:
                    forms = conjugations["conjugations"][lakara]
                    print(f"  {lakara}:")
                    for pn, form in forms.items():
                        print(f"    {pn}: {form}")
            except:
                pass

if __name__ == "__main__":
    main()
