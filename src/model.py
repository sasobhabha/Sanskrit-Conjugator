# Neural Network Model for Sanskrit Verb Conjugation
# Sequence-to-Sequence Architecture with Attention

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class CharacterTokenizer:
    """Character-level tokenizer for Sanskrit (IAST and Devanagari)"""

    # Core IAST characters and common diacritics
    # Core IAST characters and common diacritics
    IAST_CHARS = [
        'a', 'ā', 'i', 'ī', 'u', 'ū', 'ṛ', 'ṝ', 'ḷ', 'ḹ',
        'e', 'ai', 'o', 'au', 'ṃ', 'ḥ',
        'k', 'kh', 'g', 'gh', 'ṅ',
        'c', 'ch', 'j', 'jh', 'ñ',
        'ṭ', 'ṭh', 'ḍ', 'ḍh', 'ṇ',
        't', 'th', 'd', 'dh', 'n',
        'p', 'ph', 'b', 'bh', 'm',
        'y', 'r', 'l', 'v',
        'ś', 'ṣ', 's', 'h',
        '|', '_'  # Separator and underscore for condition encoding
    ]

    # Special tokens
    PAD_TOKEN = 0
    SOS_TOKEN = 1  # Start of sequence
    EOS_TOKEN = 2  # End of sequence
    UNK_TOKEN = 3  # Unknown

    def __init__(self, max_length: int = 64):
        self.max_length = max_length
        # Build vocabulary ensuring deterministic ordering
        self.char_to_idx = {}
        self.char_to_idx['<pad>'] = self.PAD_TOKEN
        self.char_to_idx['<sos>'] = self.SOS_TOKEN
        self.char_to_idx['<eos>'] = self.EOS_TOKEN
        self.char_to_idx['<unk>'] = self.UNK_TOKEN

        # Add all IAST characters in a sorted order for consistency
        for c in sorted(self.IAST_CHARS):
            if c not in self.char_to_idx:
                self.char_to_idx[c] = len(self.char_to_idx)

        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using longest-match first."""
        tokens = [self.SOS_TOKEN]

        i = 0
        while i < len(text) and len(tokens) < self.max_length - 1:
            matched = False
            # Try longest tokens first (3-char down to 1-char)
            for length in [3, 2, 1]:
                if i + length <= len(text):
                    substr = text[i:i+length]
                    if substr in self.char_to_idx:
                        tokens.append(self.char_to_idx[substr])
                        i += length
                        matched = True
                        break
            if not matched:
                # Unknown character - treat as single char as UNK
                tokens.append(self.UNK_TOKEN)
                i += 1

        tokens.append(self.EOS_TOKEN)

        # Pad or truncate
        if len(tokens) < self.max_length:
            tokens.extend([self.PAD_TOKEN] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
            tokens[-1] = self.EOS_TOKEN

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back to text string."""
        chars = []
        for tok in tokens:
            if tok in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
                continue
            if tok in self.idx_to_char:
                chars.append(self.idx_to_char[tok])
        return ''.join(chars)

    def batch_encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of strings into a padded tensor."""
        encoded = [self.encode(t) for t in texts]
        return torch.tensor(encoded, dtype=torch.long)

class Encoder(nn.Module):
    """Bi-directional LSTM encoder."""

    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)

        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.fc(hidden).unsqueeze(0)
        cell = torch.cat([cell[-2], cell[-1]], dim=1)
        cell = self.fc(cell).unsqueeze(0)

        return outputs, (hidden, cell)

class Attention(nn.Module):
    """Bahdanau additive attention."""

    def __init__(self, enc_dim: int, dec_dim: int):
        super().__init__()
        self.attn = nn.Linear(enc_dim + dec_dim, dec_dim)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor,
                encoder_outputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # hidden: [batch, dec_dim]
        # encoder_outputs: [batch, src_len, enc_dim]
        batch, src_len, _ = encoder_outputs.shape
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        scores = self.v(energy).squeeze(2)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)

        return F.softmax(scores, dim=1)

class Decoder(nn.Module):
    """LSTM decoder with attention."""

    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim * 2,  # embed + context from attention (bidirectional)
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.attention = Attention(hidden_dim * 2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor,
                cell: torch.Tensor, encoder_outputs: torch.Tensor) -> \
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input_token: [batch]
        input_token = input_token.unsqueeze(1)                 # [batch, 1]
        embedded = self.dropout(self.embedding(input_token))   # [batch, 1, embed]

        attn_weights = self.attention(hidden[-1], encoder_outputs)   # [batch, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, enc_dim*2]

        lstm_input = torch.cat([embedded, context], dim=2)      # [batch, 1, embed+hidden]
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: [batch, 1, hidden_dim]

        prediction = self.fc_out(output.squeeze(1))              # [batch, vocab_size]
        return prediction, hidden, cell

class SanskritVerbConjugator(nn.Module):
    """Full encoder-decoder model for Sanskrit verb conjugation."""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 embed_dim: int = 256, hidden_dim: int = 512,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim,
                              num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim,
                              num_layers=1, dropout=dropout)  # Single-layer decoder
        self.hidden_dim = hidden_dim
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        batch_size, tgt_len = tgt.shape
        encoder_outputs, (hidden, cell) = self.encoder(src)

        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size).to(src.device)

        decoder_input = tgt[:, 0]   # SOS token

        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            decoder_input = tgt[:, t] if self.training else output.argmax(1)

        return outputs

    def predict(self, src: torch.Tensor, max_len: int = 32) -> torch.Tensor:
        """Inference mode: generate output sequence."""
        with torch.no_grad():
            batch_size = src.shape[0]
            encoder_outputs, (hidden, cell) = self.encoder(src)

            predictions = []
            decoder_input = torch.full((batch_size,), CharacterTokenizer.SOS_TOKEN,
                                      device=src.device, dtype=torch.long)

            for _ in range(max_len):
                output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
                pred = output.argmax(1)
                predictions.append(pred.unsqueeze(1))
                decoder_input = pred

                if (pred == CharacterTokenizer.EOS_TOKEN).all():
                    break

            return torch.cat(predictions, dim=1)

def save_model(model: SanskritVerbConjugator, path: str):
    """Save model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'src_vocab_size': model.src_vocab_size,
            'tgt_vocab_size': model.tgt_vocab_size,
            'embed_dim': model.encoder.embedding.embedding_dim,
            'hidden_dim': model.hidden_dim,
            'num_layers': model.encoder.lstm.num_layers,
        }
    }, path)

def load_model(path: str, device: str = 'cpu') -> SanskritVerbConjugator:
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['model_config']
    model = SanskritVerbConjugator(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

class VerbConjugationDataset(Dataset):
    """Dataset for source-target training pairs."""

    def __init__(self, data_file: str, tokenizer: CharacterTokenizer):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.pairs = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]
        src_text = pair['source']
        tgt_text = pair['target']
        src_tokens = self.tokenizer.encode(src_text)
        tgt_tokens = self.tokenizer.encode(tgt_text)
        return torch.tensor(src_tokens, dtype=torch.long), \
               torch.tensor(tgt_tokens, dtype=torch.long)
