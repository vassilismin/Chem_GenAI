import torch
import json
import numpy as np
import pandas as pd
import torch.nn.functional as F
from models.lstm_guacamol.rnn_model import SmilesRnn
from models.lstm_guacamol.smiles_char_dict import SmilesCharDictionary
from utils.evaluate import evaluate_generated
from rdkit import Chem
from rdkit.Chem import Crippen
# from guacamol.distribution_learning_benchmark import (
#     ValidityBenchmark,
#     UniquenessBenchmark,
#     NoveltyBenchmark,
#     KLDivBenchmark,
# )

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Generate a batch of SMILES strings
def generate_batch(model, char_dict, batch_size=64, device="cpu", max_len=100):
    model.eval()
    with torch.no_grad():
        # Initialize hidden state for ALL molecules at once
        # Shape: (n_layers, batch_size, hidden_size)
        hidden = model.init_hidden(batch_size, device)

        # Start token for ALL molecules (batch_size x 1)
        # Every molecule starts with 'Q' (begin token)
        tokens = torch.full((batch_size, 1), char_dict.begin_idx).to(device)

        # Store generated characters for each molecule
        results = [[] for _ in range(batch_size)]

        # Track which molecules have finished (hit end token)
        finished = [False] * batch_size

        for _ in range(max_len):
            output, hidden = model(tokens, hidden)
            probs = F.softmax(output[:, 0, :], dim=-1)
            tokens = torch.multinomial(probs, 1)

            for i in range(batch_size):
                if not finished[i]:
                    if tokens[i].item() == char_dict.end_idx:
                        finished[i] = True
                    else:
                        results[i].append(char_dict.idx_char[tokens[i].item()])

            if all(finished):
                break

        return [char_dict.decode("".join(r)) for r in results]


# Load model
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
model.to(device)

print("Missing keys:", result.missing_keys)
print("Unexpected keys:", result.unexpected_keys)

if not result.missing_keys and not result.unexpected_keys:
    print("Model loaded successfully with no missing or unexpected keys.")

# Generate molecules
n_batches = 10  # 10 batches x 64 = 640 molecules
all_smiles = []


for i in range(n_batches):
    batch = generate_batch(model, sd, batch_size=64, device=device)
    all_smiles.extend(batch)
    print(f"Batch {i + 1}/{n_batches} done")

print(f"\nGenerated: {len(all_smiles)} molecules")

# Validate and compute logP
valid_smiles = []
logp_values = []

for smi in all_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        valid_smiles.append(smi)
        logp = Crippen.MolLogP(mol)
        logp_values.append(logp)

# Statistics
print(f"Valid: {len(valid_smiles)} ({100 * len(valid_smiles) / len(all_smiles):.1f}%)")
print(f"Unique: {len(set(valid_smiles))}")

# logP statistics
logp_array = np.array(logp_values)
print(f"\nlogP statistics:")
print(f"  Mean: {logp_array.mean():.2f}")
print(f"  Std:  {logp_array.std():.2f}")
print(f"  Min:  {logp_array.min():.2f}")
print(f"  Max:  {logp_array.max():.2f}")


training_smiles = pd.read_csv("data/chembl_logp.csv")["canonical_smiles"].tolist()
benchmark_results = evaluate_generated(all_smiles, training_set=training_smiles)
print("\nBenchmark Results:")
print(f"Total molecules generated: {benchmark_results['total']}")
print(
    f"Validity: {benchmark_results['validity'] * 100:.2f}% ({benchmark_results['valid']}/{benchmark_results['total']})"
)
print(
    f"Uniqueness: {benchmark_results['uniqueness'] * 100:.2f}% ({benchmark_results['unique']}/{benchmark_results['valid']})"
)
if benchmark_results["novelty"] is not None:
    print(
        f"Novelty: {benchmark_results['novelty'] * 100:.2f}% ({benchmark_results['unique'] - (benchmark_results['total'] - benchmark_results['valid'])}/{benchmark_results['unique']})"
    )
