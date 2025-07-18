import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "BienKieu/codeT5Plus_Contrastive_SI_new"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.to(DEVICE)

num_gen = 200
max_input_tokens = 512 

def build_test_input(example, tokenizer):
    input_text = example["text"]
    
    tokenized = tokenizer(
        input_text,
        max_length=max_input_tokens,
        padding="max_length",
        truncation=True,
    )

    return tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True)

langs = ['C_sharp', 'Python', 'Java', 'JS']

for lang in langs:
    print(f"Generating for language: {lang}", flush=True)
    
    df_lang = pd.read_csv(f'../data_merged/test_{lang}.csv')
    
    output_file = f'../generate_results/{lang}_{num_gen}_candidate.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df_lang.iterrows():
            input_text = build_test_input(row, tokenizer)
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=max_input_tokens,
                padding="max_length",
                truncation=True
            ).to(DEVICE)

            summary_ids = model.generate(
                inputs["input_ids"],
                num_beams=num_gen,   
                num_return_sequences=num_gen,        
                min_length=4,
                max_length=64,
                length_penalty=0.0
            )
            for k in range(num_gen):
                candidate = tokenizer.decode(
                    summary_ids[k],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                f.write(candidate + '\n')
    print(f"Candidates for {lang} saved to {output_file}", flush=True)

# CUDA_VISIBLE_DEVICES=1 nohup python gen.py > gen.log 2>&1 &
