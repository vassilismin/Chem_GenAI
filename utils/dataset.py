import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SmilesDataset(Dataset):
    def __init__(self, smiles_list, char_dict):
        self.smiles = smiles_list
        self.char_dict = char_dict

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        # Get SMILES string
        smi = self.smiles[idx]

        # Add start/end tokens: "CCO" → "QCCO\n"
        encoded = self.char_dict.BEGIN + self.char_dict.encode(smi) + self.char_dict.END

        # Convert characters to indices: "QCCO\n" → [1, 19, 19, 9, 2]
        indices = [self.char_dict.char_idx[c] for c in encoded]

        return torch.tensor(indices, dtype=torch.long)


def get_collate_fn(pad_idx):
    def collate_fn(batch):
        # Pad sequences to the same length
        return pad_sequence(batch, batch_first=True, padding_value=pad_idx)

    return collate_fn


# def get_collate_fn(pad_idx):
#     """Returns a collate function with the specified pad index."""

#     def collate_fn(batch):
#         padded = pad_sequence(batch, batch_first=True, padding_value=pad_idx)
#         inputs = padded[:, :-1]
#         targets = padded[:, 1:]
#         return inputs, targets

#     return collate_fn
