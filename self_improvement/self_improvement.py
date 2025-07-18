import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge import Rouge

def build_test_input(example, tokenizer, max_input_tokens=512):
    lang = example["lang"].strip()
    desc_raw = str(example["desc"])
    code_raw = str(example["code"])

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
    return src

model_path = "BienKieu/codeT5Plus_Contrastive"     
input_file = "/kaggle/input/data-self/train.csv"           
output_file = "ouput_self.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LOAD ==========
df = pd.read_csv(input_file).fillna("")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
rouge = Rouge()

# ========== XỬ LÝ ==========
start = 0
end = df.shape[0] - 1

with open('ouput_self.txt', 'w', encoding='utf-8') as f:
    for index, row in tqdm(df.iloc[start:end+1].iterrows(), total=end - start + 1):
        if index % 100 == 0:
            print(f"Processing {index}")

        input_text = build_test_input(row, tokenizer)
        reference = row['title'].strip()

        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True
        ).to(DEVICE)

        # Generate candidate titles
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=30,
            num_return_sequences=30,
            min_length=2,
            max_length=64,
            top_p=0.9
        )

        candidates = tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Rerank using ROUGE-L F1
        best_output = candidates[0]
        best_score = rouge.get_scores(best_output, reference)[0]['rouge-l']['f']

        for output in candidates[1:]:
            score = rouge.get_scores(output, reference)[0]['rouge-l']['f']
            if score > best_score:
                best_output = output
                best_score = score

        f.write(best_output.strip() + '\n')