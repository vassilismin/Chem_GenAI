import pandas as pd
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)
from utils.dataset import SmilesDataset


tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "jonghyunlee/MolGPT_pretrained-by-ZINC15"
)
tokenizer.pad_token = "<pad>"
tokenizer.bos_token = "<bos>"
tokenizer.eos_token = "<eos>"

# Load pretrained model
model = GPT2LMHeadModel.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")

# import dataset
logP_dataset = pd.read_csv("data/chembl_logp.csv")
logP_smiles = logP_dataset["canonical_smiles"][
    (logP_dataset["LogP"] < 1.0) & (logP_dataset["LogP"] > 0.0)
]

hf_dataset = Dataset.from_dict({"smiles": logP_smiles.tolist()})


def tokenize(batch):
    tokens = tokenizer(
        batch["smiles"], truncation=True, padding="max_length", max_length=128
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


dataset = hf_dataset.map(tokenize, batched=True, remove_columns=["smiles"])

training_args = TrainingArguments(
    output_dir="./molgpt2_finetuned_logP",
    num_train_epochs=5,
    per_device_train_batch_size=256,
    learning_rate=1e-4,
    save_strategy="epoch",
    logging_strategy="epoch",
    log_level="info",
)
logP_trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
logP_trainer.train()

logP_trainer.save_model("./molgpt2_finetuned_logP/final")
tokenizer.save_pretrained("./molgpt2_finetuned_logP/final")
