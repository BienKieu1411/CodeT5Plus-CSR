import os
import argparse
import logging
import numpy as np
import pandas as pd
import nltk
import evaluate
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed
)
import torch.nn.functional as F
from huggingface_hub import login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Bạn cần đăng nhập vào Hugging Face. Dùng `huggingface-cli login` hoặc set biến môi trường `HF_TOKEN`.")

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, required=True)
parser.add_argument("--valid_file", type=str, required=True)
parser.add_argument("--text_column", type=str, default=None)
parser.add_argument("--summary_column", type=str, default=None)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--model_name_or_path", type=str, default="Salesforce/codet5p-220m")
parser.add_argument("--max_source_length", type=int, default=512)
parser.add_argument("--max_target_length", type=int, default=64)
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--per_device_train_batch_size", type=int, default=8)
parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
parser.add_argument("--ignore_pad_token_for_loss", action="store_true")
parser.add_argument("--source_prefix", type=str, default="")
parser.add_argument("--push_to_hub", action="store_true")
parser.add_argument("--hub_model_id", type=str, default=None)
parser.add_argument("--contrastive_weight", type=float, default=1.0)
args = parser.parse_args()

set_seed(42)

df_train = pd.read_csv(args.train_file)
df_valid = pd.read_csv(args.valid_file)

raw_datasets = {
    "train": Dataset.from_pandas(df_train),
    "validation": Dataset.from_pandas(df_valid),
}

column_names = raw_datasets["train"].column_names
text_column = args.text_column or column_names[0]
summary_column = args.summary_column or column_names[1]

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

if model.config.decoder_start_token_id is None:
    raise ValueError("Decoder start token ID is not set in config.")

def preprocess_function(examples):
    inputs, targets = [], []
    for i in range(len(examples[text_column])):
        if examples[text_column][i] and examples[summary_column][i]:
            inputs.append(args.source_prefix + examples[text_column][i])
            targets.append(examples[summary_column][i])

    model_inputs = tokenizer(
        inputs, max_length=args.max_source_length, padding="max_length", truncation=True
    )
    labels = tokenizer(
        text_target=targets, max_length=args.max_target_length, padding="max_length", truncation=True
    )
    if args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = raw_datasets["train"].map(preprocess_function, batched=True, remove_columns=column_names)
tokenized_valid = raw_datasets["validation"].map(preprocess_function, batched=True, remove_columns=column_names)

rouge = evaluate.load("rouge")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    result["gen_len"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds])
    return result

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

class ContrastiveTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        loss_ce = outputs.loss

        encoder_hidden = outputs.encoder_last_hidden_state
        enc_repr = encoder_hidden.mean(dim=1)

        decoder_hidden = outputs.decoder_hidden_states[-1]
        mask = (labels != -100).unsqueeze(-1).float()
        dec_repr = (decoder_hidden * mask).sum(dim=1) / mask.sum(dim=1)

        temperature = 0.05
        enc_norm = F.normalize(enc_repr, p=2, dim=1)
        dec_norm = F.normalize(dec_repr, p=2, dim=1)
        sim = torch.matmul(enc_norm, dec_norm.T) / temperature
        labels_contrast = torch.arange(sim.size(0)).to(sim.device)
        loss_contrast = F.cross_entropy(sim, labels_contrast)

        total_loss = loss_ce + args.contrastive_weight * loss_contrast
        return (total_loss, outputs) if return_outputs else total_loss

training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=32,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    save_strategy="epoch",
    eval_strategy="no",
    logging_dir=f"{args.output_dir}/logs",
    logging_steps=100,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    push_to_hub=args.push_to_hub,
    hub_model_id=args.hub_model_id,
    hub_strategy="every_save" if args.push_to_hub else "end",
    # load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True
)

trainer = ContrastiveTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()
tokenizer.save_pretrained(args.output_dir)

if args.push_to_hub and args.hub_model_id:
    trainer.push_to_hub()
    tokenizer.push_to_hub(args.hub_model_id)
    print(f"Pushed to HuggingFace Hub: https://huggingface.co/{args.hub_model_id}")
