import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from rouge import FilesRouge

from sentence_transformers import CrossEncoder

max_input_tokens = 512
target_max_len = 64
languages = ['C_sharp', 'Python', 'Java', 'JS']

cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--use_textrank", action="store_true", help="Use reranking strategy")
    parser.add_argument("--num_candidates", type=int, default=30, help="Number of candidate sequences to generate")
    parser.add_argument("--data_dir", type=str, default=".", help="Path to directory with test CSV files")
    parser.add_argument("--rerank_method", type=str, default="cross-encoder",
                        choices=["textrank", "cross-encoder", "hybrid"], help="Reranking strategy")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for cross-encoder in hybrid rerank (0~1)")
    return parser.parse_args()

def build_test_input(example, tokenizer):
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

def textrank_rerank(candidates, alpha=0.23):
    if len(candidates) == 1:
        return candidates[0]
    try:
        vectorizer = TfidfVectorizer(lowercase=False).fit_transform(candidates)
        similarity_matrix = cosine_similarity(vectorizer)
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph, alpha=alpha, max_iter=100, tol=1e-6)
        best_idx = sorted(scores, key=scores.get, reverse=True)[0]
        return candidates[best_idx]
    except Exception as e:
        print("TextRank failed:", e)
        return candidates[0]

def cross_encoder_rerank(input_text, candidates):
    if len(candidates) == 1:
        return candidates[0]
    try:
        pairs = [(input_text, cand) for cand in candidates]
        scores = cross_encoder_model.predict(pairs)
        best_idx = scores.argmax()
        return candidates[best_idx]
    except Exception as e:
        print("Cross-Encoder rerank failed:", e)
        return candidates[0]

def hybrid_rerank(input_text, candidates, alpha=0.7):
    if len(candidates) == 1:
        return candidates[0]

    try:
        # TextRank scores
        vectorizer = TfidfVectorizer(lowercase=False).fit_transform(candidates)
        sim_matrix = cosine_similarity(vectorizer)
        graph = nx.from_numpy_array(sim_matrix)
        tr_scores_dict = nx.pagerank(graph, alpha=0.23)
        tr_scores = [tr_scores_dict[i] for i in range(len(candidates))]

        # Cross-Encoder scores
        pairs = [(input_text, cand) for cand in candidates]
        ce_scores = cross_encoder_model.predict(pairs)

        # Normalize scores
        ce_scores = (ce_scores - np.min(ce_scores)) / (np.max(ce_scores) - np.min(ce_scores) + 1e-8)
        tr_scores = (np.array(tr_scores) - np.min(tr_scores)) / (np.max(tr_scores) - np.min(tr_scores) + 1e-8)

        # Combine
        final_scores = alpha * ce_scores + (1 - alpha) * tr_scores
        best_idx = np.argmax(final_scores)
        return candidates[best_idx]

    except Exception as e:
        print("Hybrid rerank failed:", e)
        return candidates[0]

def generate_outputs(args, model, tokenizer, device):
    for lang in languages:
        print(f"Processing {lang}...")
        input_path = os.path.join(args.data_dir, f"test_{lang}.csv")
        df = pd.read_csv(input_path).fillna("")

        output_file = os.path.join(args.output_dir, f"{lang}_output.txt")
        with open(output_file, 'w', encoding='utf-8') as fout:
            for _, row in tqdm(df.iterrows(), total=len(df)):
                input_text = build_test_input(row, tokenizer)
                inputs = tokenizer(input_text, return_tensors="pt", max_length=max_input_tokens,
                                   padding="max_length", truncation=True).to(device)

                num_return = args.num_candidates if args.use_textrank else 1

                summary_ids = model.generate(
                    inputs["input_ids"],
                    num_beams=num_return,
                    num_return_sequences=num_return,
                    min_length=2,
                    max_length=target_max_len,
                    top_p=0.9,
                    top_k=5,
                    length_penalty=0.0
                )

                candidates = [
                    tokenizer.decode(summary_ids[k], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    for k in range(num_return)
                ]

                if args.use_textrank:
                    if args.rerank_method == "textrank":
                        best_output = textrank_rerank(candidates)
                    elif args.rerank_method == "cross-encoder":
                        best_output = cross_encoder_rerank(input_text, candidates)
                    elif args.rerank_method == "hybrid":
                        best_output = hybrid_rerank(input_text, candidates, alpha=args.alpha)
                else:
                    best_output = candidates[0]

                fout.write(best_output.strip() + '\n')

def evaluate_rouge(output_dir, data_dir):
    files_rouge = FilesRouge()
    results = []

    for lang in languages:
        print(f"Evaluating {lang}...")
        hyp_path = os.path.join(output_dir, f"{lang}_output.txt")
        ref_path = os.path.join(data_dir, f"test_{lang}.csv")

        df_ref = pd.read_csv(ref_path).fillna("")
        ref_file = os.path.join(output_dir, f"ref_{lang}.txt")
        df_ref["title"].astype(str).to_csv(ref_file, index=False, header=False)

        scores = files_rouge.get_scores(hyp_path, ref_file, avg=True)

        result = {
            "Language": lang,
            "ROUGE-1 (F1)": round(scores["rouge-1"]["f"], 4),
            "ROUGE-2 (F1)": round(scores["rouge-2"]["f"], 4),
            "ROUGE-L (F1)": round(scores["rouge-l"]["f"], 4)
        }
        results.append(result)

    df_result = pd.DataFrame(results)
    print("\n=== ROUGE Evaluation Results ===")
    print(df_result.to_string(index=False))
    df_result.to_csv(os.path.join(output_dir, "rouge_scores.csv"), index=False)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    generate_outputs(args, model, tokenizer, device)
    evaluate_rouge(args.output_dir, args.data_dir)

if __name__ == "__main__":
    main()