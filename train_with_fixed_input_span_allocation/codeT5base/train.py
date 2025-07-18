import os
import argparse
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, Seq2SeqTrainingArguments
)
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
import evaluate
import torch
from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Bạn cần đăng nhập vào Hugging Face. Chạy `huggingface-cli login` hoặc đặt biến môi trường `HF_TOKEN`.")

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, required=True)
parser.add_argument("--valid_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--init_model", type=str, default="Salesforce/codet5p-220m")
parser.add_argument("--num_train_epochs", type=int, default=None)
parser.add_argument("--push_to_hub", action="store_true")
parser.add_argument("--hub_model_id", type=str, default=None)
args = parser.parse_args()

print(f"Loading model from: {args.init_model}")
tokenizer = AutoTokenizer.from_pretrained(args.init_model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.init_model)

target_max_len = 64
max_input_tokens = 512
num_epochs = args.num_train_epochs if args.num_train_epochs else 4

print(f"Loading train data from: {args.train_file}")
df_train = pd.read_csv(args.train_file)
dataset_train = Dataset.from_pandas(df_train)

print(f"Loading valid data from: {args.valid_file}")
df_valid = pd.read_csv(args.valid_file)
dataset_valid = Dataset.from_pandas(df_valid)

def preprocess(example):
    lang = example["lang"].strip()
    desc_raw = str(example["desc"])
    code_raw = str(example["code"])
    title = str(example["title"])

    lang_token = tokenizer.tokenize(lang + ":")
    marker_token = tokenizer.tokenize("<code>")
    desc_token = tokenizer.tokenize(desc_raw)
    code_token = tokenizer.tokenize(code_raw)

    reserved = len(lang_token) + len(marker_token)
    available = max_input_tokens - reserved

    desc_max = available // 2
    desc_trunc = desc_token[:desc_max]
    code_trunc = code_token[:available - len(desc_trunc)]

    desc_str = tokenizer.convert_tokens_to_string(desc_trunc)
    code_str = tokenizer.convert_tokens_to_string(code_trunc)
    src = f"{lang}: {desc_str} <code> {code_str}"

    model_input = tokenizer(
        src,
        truncation=True,
        padding="max_length",
        max_length=max_input_tokens
    )

    labels = tokenizer(
        title,
        truncation=True,
        padding="max_length",
        max_length=target_max_len
    )
    model_input["labels"] = [
        t if t != tokenizer.pad_token_id else -100
        for t in labels["input_ids"]
    ]

    return model_input

print("Tokenizing...")
tokenized_dataset_train = dataset_train.map(preprocess)
tokenized_dataset_valid = dataset_valid.map(preprocess)

rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(pred.strip().split(". ")) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split(". ")) for label in decoded_labels]

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    return {
        "rouge1": result["rouge1"].mid.fmeasure,
        "rouge2": result["rouge2"].mid.fmeasure,
        "rougeL": result["rougeL"].mid.fmeasure
    }

training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    eval_strategy="no",
    save_strategy="epoch",
    eval_accumulation_steps=4,
    save_total_limit=2,
    logging_dir=f"{args.output_dir}/logs",
    logging_steps=100,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    push_to_hub=args.push_to_hub,
    hub_model_id=args.hub_model_id,
    hub_strategy="every_save" if args.push_to_hub else "end",
    load_best_model_at_end=False,
    metric_for_best_model="rougeL",
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print(f"Starting training | Epochs: {num_epochs}")
trainer.train()

print("Saving model to ./final_model/")
trainer.save_model("./final_model/")
tokenizer.save_pretrained("./final_model/") 

if args.push_to_hub:
    trainer.push_to_hub()
    tokenizer.push_to_hub(args.hub_model_id)
    print(f"Model & tokenizer pushed to: https://huggingface.co/{args.hub_model_id}")