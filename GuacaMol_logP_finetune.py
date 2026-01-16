import torch
import json
from utils.data_loader import create_dataloader
from utils.GuacaMol_trainer import GuacaMolTrainer
from models.lstm_guacamol.smiles_char_dict import SmilesCharDictionary
from models.lstm_guacamol.rnn_model import SmilesRnn

# Paths
model_dir = "models/lstm_guacamol/pretrained_model"
weights_path = f"{model_dir}/model_final_0.473.pt"
config_path = f"{model_dir}/model_final_0.473.json"

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 30
data_path = "data/lipophilicity_astrazeneca.tab"
logp_range = (1.5, 3.0)

# Load Vocabulary
char_dict = SmilesCharDictionary()
vocab_size = char_dict.get_char_num()
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
    print("Pre-trained model loaded")

# Create DataLoader with logP filtering
dataloader = create_dataloader(
    data_path=data_path,
    char_dict=char_dict,
    batch_size=batch_size,
    logp_range=logp_range,
)
print(f"DataLoader created with {len(dataloader)} batches.")
print(f"Dataset size: {len(dataloader.dataset)} molecules")
print(f"Batches per epoch: {len(dataloader)}")

# Initialize Trainer
trainer = GuacaMolTrainer(model, dataloader, device=device, lr=1e-4)
trainer.fit(epochs=3)
# Save fine-tuned model
trainer.save("models/lstm_guacamol/finetuned_model.pt")
print("Fine-tuned model saved")
