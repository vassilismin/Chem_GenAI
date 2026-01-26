from utils.data_loader import create_dataloader
from models.lstm_guacamol.smiles_char_dict import SmilesCharDictionary

# tdc_data_path = "data/lipophilicity_astrazeneca.tab"
chembl_data_path = "data/chembl_logp_0_2.csv"
char_dict = SmilesCharDictionary()

dataloader = create_dataloader(
    data_path=chembl_data_path,
    smiles_col="canonical_smiles",
    char_dict=char_dict,
    batch_size=64,
    logp_range=(0.0, 2.0),
)

# --- Validations ---

# 1. Check we got a dataloader with data
print(f"Number of batches: {len(dataloader)}")
print(f"Dataset size: {len(dataloader.dataset)}")

# 2. Get one batch and check shape
batch = next(iter(dataloader))
print(f"Batch shape: {batch.shape}")  # [batch_size, seq_len]

# 3. Check values are valid indices
print(f"Min index: {batch.min().item()}, Max index: {batch.max().item()}")
print(f"Vocab size: {char_dict.get_char_num()}")
assert batch.min() >= 0, "Negative index found!"
assert batch.max() < char_dict.get_char_num(), "Index exceeds vocab size!"

# 4. Decode first sample back to SMILES
first_sample = batch[0].tolist()
# Exclude PAD, BEGIN (Q) and END(\n) tokens for decoding
skipped_indices = {char_dict.pad_idx, char_dict.begin_idx, char_dict.end_idx}
decoded_chars = [
    char_dict.idx_char[idx] for idx in first_sample if idx not in skipped_indices
]
decoded_smiles = char_dict.decode("".join(decoded_chars))
print(f"Decoded first sample: {decoded_smiles}")

print("\nAll validations passed!")
