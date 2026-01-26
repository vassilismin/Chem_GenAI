# Download th bulk file
# wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_chemreps.txt.gz
# gunzip chembl_36_chemreps.txt.gz

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen
from models.lstm_guacamol.smiles_char_dict import SmilesCharDictionary


# Load the Chembl SMILES
df = pd.read_csv("chembl_36_chemreps.txt", sep="\t")


# Compute logP for each SMILES with RDKit and store in a new column
def cal_logp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Crippen.MolLogP(mol)
    return None


# Add LogP column
df["LogP"] = df["canonical_smiles"].apply(cal_logp)
df.head()

# Filter out rows where LogP could not be computed
df = df[df["LogP"].notnull()]

# Filter out molecules with characters not in vocabulary
char_dict = SmilesCharDictionary()


def is_valid_for_vocab(smiles):
    """Check if all characters in encoded SMILES are in vocabulary."""
    try:
        encoded = char_dict.encode(smiles)  # Cl->X, Br->Y, etc.
        for c in encoded:
            if c not in char_dict.char_idx:
                return False
        return True
    except:
        return False


df = df[df["canonical_smiles"].apply(is_valid_for_vocab)]
print(f"Molecules with valid vocabulary: {len(df)}")
# Save the filtered dataset to a new file
df.to_csv("data/chembl_logp.csv", index=False)
