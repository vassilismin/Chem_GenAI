import torch
import json
import torch.nn.functional as F
from models.lstm_guacamol.rnn_model import SmilesRnn
from models.lstm_guacamol.smiles_char_dict import SmilesCharDictionary

# Paths
model_dir = "models/lstm_guacamol/pretrained_model"
weights_path = f"{model_dir}/model_final_0.473.pt"
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


# Generate a SMILES string
model.eval()  # Switch to evaluation mode (disables dropout)
with torch.no_grad():  # Disable gradient computation (saves memory, faster)
    hidden = model.init_hidden(1, "cpu")  # Create initial LSTM hidden state (zeros)
    # 1 = batch size, generating one molecule
    token = torch.tensor([[sd.begin_idx]])  # start token
    smiles = []

    for _ in range(100):
        output, hidden = model(
            token, hidden
        )  # Feed token → get prediction + new hidden state
        # The LSTM takes the current token and hidden state, outputs:
        #   - output: logits (scores) for each possible next character
        #   - hidden: updated memory state passed to next iteration
        probs = F.softmax(output[0, 0], dim=-1)
        token = torch.multinomial(probs, num_samples=1).unsqueeze(0)

        if token.item() == sd.end_idx:
            break
        smiles.append(sd.idx_char[token.item()])

    generated = sd.decode("".join(smiles))
    print(f"Generated SMILES: {generated}")

# Validate with RDKit
from rdkit import Chem

mol = Chem.MolFromSmiles(generated)
print("Valid molecule" if mol is not None else "Invalid molecule")
