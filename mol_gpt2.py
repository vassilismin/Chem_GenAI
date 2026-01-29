from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "jonghyunlee/MolGPT_pretrained-by-ZINC15"
)
tokenizer.pad_token = "<pad>"
tokenizer.bos_token = "<bos>"
tokenizer.eos_token = "<eos>"

# Load pretrained model
model = GPT2LMHeadModel.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")


def generate_smiles(model, tokenizer, num_sequences=5, temperature=1.0):
    return model.generate(
        max_length=128,
        num_return_sequences=num_sequences,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        return_dict_in_generate=True,
    )


decoder_output = generate_smiles(model, tokenizer)

generated_smiles = [
    tokenizer.decode(g, skip_special_tokens=True) for g in decoder_output.sequences
]
print(generated_smiles)

for i, smi in enumerate(generated_smiles):
    generated_smiles[i] = smi[1:]
print(generated_smiles)

from rdkit import Chem

invalid_smiles = []
for smiles in generated_smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        invalid_smiles.append(smiles)
        print(f"Invalid SMILES: {smiles}")

print(f"number of invalid smiles = {len(invalid_smiles)}")
if len(invalid_smiles) != 0:
    print("Invalid SMILES:", invalid_smiles)
