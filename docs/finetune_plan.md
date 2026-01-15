# Fine-Tuning Pre-Trained Model on Lipophilicity Data

## Overview

Fine-tune a pre-trained SMILES generator on your lipophilicity dataset using transfer learning. The model learns to generate molecules similar to those in your dataset.

---

## Step 1: Select a Pre-Trained Model

### Options

| Model | Architecture | Pre-trained On | PyTorch Compatible |
|-------|-------------|----------------|-------------------|
| molecular-transformer | Transformer | ChEMBL | Yes |
| SMILES-RNN (GuacaMol) | LSTM | ChEMBL 1.6M | Yes |
| MolGPT | GPT | ZINC/ChEMBL | Yes (HuggingFace) |

### Recommendation
Start with a **SMILES-RNN** (LSTM/GRU) from GuacaMol - simpler to understand, well-documented weights available.

### Technical Tips
- Download pre-trained weights from GuacaMol GitHub or train your own on ChEMBL
- Ensure vocabulary compatibility (character set must match)
- Typical architecture: 3-layer LSTM, 512 hidden units, ~2M parameters

---

## Step 2: Tokenization

### Approach: Character-Level Tokenizer

```python
# Standard SMILES character vocabulary
VOCAB = ['<PAD>', '<SOS>', '<EOS>', 'C', 'c', 'N', 'n', 'O', 'o', 'S', 's',
         'F', 'Cl', 'Br', 'I', '(', ')', '[', ']', '=', '#', '@', '+', '-',
         '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.', '/', '\\']
```

### Technical Tips
- Use `<SOS>` (start) and `<EOS>` (end) tokens
- `<PAD>` for batch padding (use `pad_sequence` from PyTorch)
- Handle two-character tokens: `Cl`, `Br` → treat as single tokens
- Max sequence length: check your data, typically 100-150 chars is safe

### Code Pattern
```python
class SmilesTokenizer:
    def __init__(self, vocab):
        self.char_to_idx = {c: i for i, c in enumerate(vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(vocab)}

    def encode(self, smiles):
        # Handle multi-char tokens (Cl, Br)
        tokens = []
        i = 0
        while i < len(smiles):
            if smiles[i:i+2] in self.char_to_idx:
                tokens.append(self.char_to_idx[smiles[i:i+2]])
                i += 2
            else:
                tokens.append(self.char_to_idx[smiles[i]])
                i += 1
        return tokens
```

---

## Step 3: Prepare DataLoader

### Technical Tips
- Collate function should handle variable-length sequences
- Use `pack_padded_sequence` for efficient RNN training
- Shuffle training data each epoch

### Code Pattern
```python
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def collate_fn(batch):
    smiles = [item for item in batch]
    lengths = [len(s) for s in smiles]
    padded = pad_sequence(smiles, batch_first=True, padding_value=PAD_IDX)
    return padded, lengths
```

---

## Step 4: Fine-Tune with Teacher Forcing

### Objective
Adapt the pre-trained model to your dataset's SMILES distribution. The model learns to predict the next token given previous tokens.

### Training Loop
```python
for epoch in range(num_epochs):
    model.train()
    for batch, lengths in dataloader:
        # Input: <SOS> + SMILES[:-1]
        # Target: SMILES[1:] + <EOS>
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, vocab_size),
                               targets.view(-1),
                               ignore_index=PAD_IDX)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

### Technical Tips
- **Lower learning rate**: 1e-4 to 1e-5 (pre-trained weights are already good)
- **Fewer epochs**: 5-20 epochs usually sufficient
- **Gradient clipping**: Prevents exploding gradients (max_norm=1.0)
- **Freeze early layers** (optional): Only fine-tune last 1-2 layers initially
- **Monitor validation loss**: Stop when it plateaus or increases

### Hyperparameters
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
```

### Layer Freezing (Optional)
```python
# Freeze embedding and first LSTM layer
for param in model.embedding.parameters():
    param.requires_grad = False
for param in model.lstm.layers[0].parameters():
    param.requires_grad = False
```

---

## Step 5: Sampling from the Model

### Autoregressive Generation
```python
def sample(model, max_len=100, temperature=1.0):
    model.eval()
    with torch.no_grad():
        token = SOS_IDX
        hidden = None
        smiles = []

        for _ in range(max_len):
            input_tensor = torch.tensor([[token]])
            logits, hidden = model(input_tensor, hidden)

            # Temperature scaling
            probs = F.softmax(logits / temperature, dim=-1)
            token = torch.multinomial(probs[0, 0], 1).item()

            if token == EOS_IDX:
                break
            smiles.append(idx_to_char[token])

        return ''.join(smiles)
```

### Technical Tips
- **Temperature**:
  - 1.0 = normal sampling
  - < 1.0 = more conservative (higher probability tokens)
  - > 1.0 = more diverse (flatter distribution)
- **Top-k sampling**: Only sample from top k tokens for better quality
- **Beam search**: Deterministic, higher quality but less diverse

---

## Step 6: Evaluation

### Metrics to Track
1. **Validity**: % of generated SMILES that parse (RDKit)
2. **Uniqueness**: % of valid molecules that are unique
3. **Novelty**: % not in training set
4. **Internal diversity**: Tanimoto distance between generated molecules

### Code Pattern
```python
from rdkit import Chem

def evaluate_generator(model, n_samples=1000, training_smiles=None):
    smiles_list = [sample(model) for _ in range(n_samples)]

    valid = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    unique = list(set(valid))

    results = {
        'validity': len(valid) / n_samples,
        'uniqueness': len(unique) / len(valid) if valid else 0,
    }

    if training_smiles:
        novel = [s for s in unique if s not in training_smiles]
        results['novelty'] = len(novel) / len(unique) if unique else 0

    return results
```

### What to Expect
- **Validity**: Should be >90% if fine-tuning worked
- **Uniqueness**: Should be >80%
- **Novelty**: Depends on dataset size, typically 50-90%

---

## Summary Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Load pre-trained SMILES generator (LSTM/Transformer)        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. Build tokenizer (match pre-trained vocabulary)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. Create DataLoader with padding/collation                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. Fine-tune with teacher forcing                              │
│     - Low LR (1e-4), few epochs, monitor val loss               │
│     - Optional: freeze early layers                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. Sample molecules and evaluate                               │
│     - Validity, uniqueness, novelty                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Common Pitfalls

| Problem | Cause | Solution |
|---------|-------|----------|
| Validity drops | Overfitting or wrong tokenizer | Check vocab compatibility, early stopping |
| Loss doesn't decrease | LR too low or model frozen | Increase LR, check requires_grad |
| Loss explodes | LR too high | Reduce LR, add gradient clipping |
| Generates only training data | Overfitting | More regularization, fewer epochs |
| Gibberish output | Tokenizer mismatch | Ensure vocab matches pre-trained model |
