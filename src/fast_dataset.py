import torch
from torch.utils.data import Dataset

class PreTokenizedDataset(Dataset):
    """Dataset that loads pre-tokenized source-target tensors."""

    def __init__(self, tensor_path: str):
        data = torch.load(tensor_path, map_location='cpu')
        self.src = data['src']
        self.tgt = data['tgt']
        assert len(self.src) == len(self.tgt)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]
