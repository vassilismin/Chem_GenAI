# Chem_GenAI

Experiments with generative models for molecular design using PyTorch.

## Goal

Fine-tune pre-trained SMILES generators to produce molecules with desired properties (e.g., lipophilicity).

## Current Progress

- [x] Downloaded lipophilicity dataset from TDC (Therapeutics Data Commons)
- [x] Downloaded pre-trained GuacaMol LSTM model and weights
- [x] Verified model loading and SMILES generation
- [ ] Prepare DataLoader for lipophilicity data
- [ ] Fine-tune model on lipophilicity SMILES
- [ ] Evaluate generated molecules

## Project Structure

```
Chem_GenAI/
├── data/
│   └── lipophilicity_astrazeneca.tab   # TDC lipophilicity dataset
├── models/
│   └── lstm_guacamol/
│       ├── rnn_model.py                # LSTM architecture
│       ├── smiles_char_dict.py         # Tokenizer/vocabulary
│       ├── rnn_utils.py                # Utility functions
│       ├── rnn_trainer.py              # Training utilities
│       └── pretrained_model/
│           ├── model_final_0.473.pt    # Pre-trained weights
│           └── model_final_0.473.json  # Model config
├── docs/
│   └── finetune_plan.md                # Fine-tuning strategy
├── retrieve_data.py                    # Script to download TDC data
└── verify_guacamol.py                  # Script to verify model loading
```

## Setup

```bash
# Install dependencies
pip install torch rdkit PyTDC pandas

# Verify model loads correctly
python verify_guacamol.py
```

## Usage

### Generate molecules from pre-trained model

```python
from models.lstm_guacamol.rnn_model import SmilesRnn
from models.lstm_guacamol.smiles_char_dict import SmilesCharDictionary
import torch
import torch.nn.functional as F
import json

# Load model
sd = SmilesCharDictionary()
with open('models/lstm_guacamol/pretrained_model/model_final_0.473.json') as f:
    config = json.load(f)
model = SmilesRnn(**config)
model.load_state_dict(torch.load('models/lstm_guacamol/pretrained_model/model_final_0.473.pt',
                                  map_location='cpu', weights_only=False))
model.eval()

# Generate
with torch.no_grad():
    hidden = model.init_hidden(1, 'cpu')
    token = torch.tensor([[sd.begin_idx]])
    smiles = []
    for _ in range(100):
        output, hidden = model(token, hidden)
        probs = F.softmax(output[0, 0], dim=-1)
        token = torch.multinomial(probs, 1).unsqueeze(0)
        if token.item() == sd.end_idx:
            break
        smiles.append(sd.idx_char[token.item()])
    generated = sd.decode(''.join(smiles))
    print(generated)
```

## References

- [GuacaMol](https://github.com/BenevolentAI/guacamol_baselines) - Pre-trained SMILES LSTM
- [TDC](https://tdcommons.ai/) - Therapeutics Data Commons
