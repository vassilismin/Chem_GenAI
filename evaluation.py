import torch
import json
import torch.nn.functional as F
from models.lstm_guacamol.rnn_model import SmilesRnn
from models.lstm_guacamol.smiles_char_dict import SmilesCharDictionary
from rdkit.Chem import Chem


model_dir = "models/lstm_guacamol/pretrained_model"
weights_path = f"{model_dir}/finetuned_model.pt"
# weights_path = f"{model_dir}/model_final_0.473.pt"
config_path = f"{model_dir}/model_final_0.473.json"

sd = SmilesCharDictionary()
vocab_size = sd.get_char_num()
print(f"Vocab size: {vocab_size}")

# Load config of the model
with open(config_path, "r") as f:
    config = json.load(f)
print(f"Model config: {config}")

# Initialize the model
model = SmilesRnn(**config)

# Load weights of the model
state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
result = model.load_state_dict(state_dict)

print("Missing keys:", result.missing_keys)
print("Unexpected keys:", result.unexpected_keys)

if not result.missing_keys and not result.unexpected_keys:
    print("Model loaded successfully with no missing or unexpected keys.")


def generate_smiles(model, char_dict, device="cpu", max_len=100):
    """Generate a single SMILES string."""
    model.eval()
    with torch.no_grad():
        hidden = model.init_hidden(1, device)
        token = torch.tensor([[char_dict.begin_idx]]).to(device)
        smiles = []

        for _ in range(max_len):
            output, hidden = model(token, hidden)
            probs = F.softmax(output[0, 0], dim=-1)
            token = torch.multinomial(probs, num_samples=1).unsqueeze(0)

            if token.item() == char_dict.end_idx:
                break
            smiles.append(char_dict.idx_char[token.item()])

        return char_dict.decode("".join(smiles))


def generate_molecules(model, char_dict, n_molecules=100, device="cpu"):
    """Generate n molecules and return list of SMILES."""
    generated_smiles = []

    for i in range(n_molecules):
        smi = generate_smiles(model, char_dict, device)
        generated_smiles.append(smi)

    return generated_smiles


# Generate molecules
n_molecules = 500
generated_list = generate_molecules(model, sd, n_molecules=n_molecules)

# Validate
valid_count = sum(1 for smi in generated_list if Chem.MolFromSmiles(smi) is not None)
print(f"Generated: {n_molecules}")
print(f"Valid: {valid_count} ({100 * valid_count / n_molecules:.1f}%)")

# Estimate MolLogP using RDKit for valid molecules
logp_values = []
for smi in generated_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        logp = Crippen.MolLogP(mol)
        logp_values.append(logp)

# Estimate summary statistics of MolLogP
avg_logp = sum(logp_values) / len(logp_values)
print(f"\nAverage MolLogP of valid molecules: {avg_logp:.2f}")
sd_logp = (sum((x - avg_logp) ** 2 for x in logp_values) / len(logp_values)) ** 0.5
print(f"Std Dev of MolLogP: {sd_logp:.2f}")
min_logp = min(logp_values)
max_logp = max(logp_values)
print(f"Min MolLogP: {min_logp:.2f}")
print(f"Max MolLogP: {max_logp:.2f}")
