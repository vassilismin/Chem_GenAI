# Chem_GenAI

## Purpose

This project explores chemical generative models and reinforcement learning for molecular design. The goal is to train generative models (RNNs, Transformers) to generate molecules with desired properties.

## Approach

- **Framework**: PyTorch (no high-level wrappers like REINVENT) to maintain full control and learn the underlying mechanics
- **Generative models**: Character-level SMILES generation using RNNs or Transformers
- **Optimization**: Fine-tune generators using reinforcement learning (policy gradients) to bias output toward molecules with target properties
- **Transfer learning**: Start from pre-trained weights where available, then fine-tune

## Data

- Primary source: Therapeutics Data Commons (TDC)
- Format: SMILES strings with optional property labels
- Properties of interest: logP, QED, solubility, lipophilicity

## Project Structure

```
Chem_GenAI/
├── data/           # Datasets from TDC
├── models/         # PyTorch model definitions
├── training/       # Training loops, RL algorithms
├── utils/          # Tokenizers, data loaders, chemistry utilities
└── experiments/    # Notebooks and scripts for experiments
```

## Dependencies

- PyTorch
- RDKit (molecular property computation)
- TDC (Therapeutics Data Commons)
- pandas, numpy
