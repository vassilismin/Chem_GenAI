import pandas as pd
from models.lstm_guacamol.smiles_char_dict import SmilesCharDictionary

df_filtered = pd.read_csv("data/chembl_logp_0_2.csv")
df_filtered.head()

# filter out molecules with not allowed structures
char_dict = SmilesCharDictionary()


def is_valid_for_vocab(smiles, char_dict):
    """Check if all characters in encoded SMILES are in vocabulary."""
    try:
        encoded = char_dict.encode(smiles)  # Cl->X, Br->Y, etc.
        for c in encoded:
            if c not in char_dict.char_idx:
                return False
        return True
    except:
        return False


# Apply filter
df_filtered = df_filtered[
    df_filtered["canonical_smiles"].apply(lambda s: is_valid_for_vocab(s, char_dict))
]

df_filtered.to_csv("data/chembl_logp_0_2_filtered.csv", index=False)
