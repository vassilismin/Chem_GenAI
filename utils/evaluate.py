from rdkit import Chem


def evaluate_generated(smiles_list, training_set=None):
    """Evaluate generated SMILES - validity, uniqueness, novelty."""

    # Validity
    valid = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    validity = len(valid) / len(smiles_list) if smiles_list else 0

    # Uniqueness
    unique = list(set(valid))
    uniqueness = len(unique) / len(valid) if valid else 0

    # Novelty
    if training_set:
        training_set = set(training_set)
        novel = [s for s in unique if s not in training_set]
        novelty = len(novel) / len(unique) if unique else 0
    else:
        novelty = None

    return {
        "total": len(smiles_list),
        "valid": len(valid),
        "validity": validity,
        "unique": len(unique),
        "uniqueness": uniqueness,
        "novelty": novelty,
    }
