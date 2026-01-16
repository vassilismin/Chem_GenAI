import pandas as pd
from torch.utils.data import DataLoader
from .dataset import SmilesDataset, get_collate_fn


def create_dataloader(data_path, char_dict, batch_size=64, logp_range=(1.5, 3.0)):
    # Load data
    df = pd.read_csv(data_path, sep="\t")

    # Filter by logP range
    min_logp, max_logp = logp_range
    filtered = df[(df["Y"] >= min_logp) & (df["Y"] <= max_logp)]
    smiles_list = filtered["Drug"].tolist()

    # Create dataset and loader
    dataset = SmilesDataset(smiles_list, char_dict)
    collate_fn = get_collate_fn(char_dict.pad_idx)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    return loader
