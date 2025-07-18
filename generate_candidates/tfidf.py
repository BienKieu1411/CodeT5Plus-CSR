import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from tqdm import tqdm
from rouge import FilesRouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import networkx as nx

def normalize(v):
    v = np.asarray(v)
    if len(v.shape) == 1:
        v = v.reshape(-1, 1)
    if np.all(v == v[0]):
        return np.zeros_like(v).flatten()
    return MinMaxScaler().fit_transform(v).flatten()

def rerank_by_pagerank_tfidf(candidates, alpha=1.0, pagerank_alpha=0.23):
    try:
        tfidf_matrix = TfidfVectorizer(lowercase=False).fit_transform(candidates)
        sim_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(sim_matrix, 0.0)
        graph = nx.from_numpy_array(sim_matrix)
        pagerank_scores = nx.pagerank(graph, alpha=pagerank_alpha)
        scores = np.array([pagerank_scores[i] for i in range(len(candidates))])
        scores = normalize(scores)
    except:
        scores = np.zeros(len(candidates))
    scores *= alpha
    return candidates[int(np.argmax(scores))]

def run_with_pagerank_tfidf_only(alpha=1.0, output_dir="../rerank_result/tfidf", group_size=200):
    langs = ['C_sharp', 'Python', 'Java', 'JS']
    os.makedirs(output_dir, exist_ok=True)

    for lang in langs:
        print(f"\nReranking for {lang}...")
        with open(f'../generate_results/{lang}_200_candidate.txt', 'r', encoding='utf-8') as f:
            candidates_all = [line.strip() for line in f if line.strip()]

        df_ref = pd.read_csv(f'../data_merged/test_{lang}.csv')
        refs = df_ref['title'].astype(str).tolist()

        assert len(refs) * group_size == len(candidates_all)

        preds = []
        for i in tqdm(range(len(refs))):
            group = candidates_all[i * group_size:(i + 1) * group_size]
            chosen = rerank_by_pagerank_tfidf(group, alpha=alpha)
            preds.append(chosen)

        out_path = os.path.join(output_dir, f'{lang}_best.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            for line in preds:
                f.write(line.strip() + '\n')

        ref_path = os.path.join(output_dir, f'{lang}_ref.txt')
        with open(ref_path, 'w', encoding='utf-8') as f:
            for line in refs:
                f.write(line.strip() + '\n')

        files_rouge = FilesRouge()
        scores = files_rouge.get_scores(hyp_path=out_path, ref_path=ref_path, avg=True)
        print(f"{lang} ROUGE-1: {scores['rouge-1']['f']:.4f} | ROUGE-2: {scores['rouge-2']['f']:.4f} | ROUGE-L: {scores['rouge-l']['f']:.4f}")

if __name__ == "__main__":
    run_with_pagerank_tfidf_only(alpha=1.0)

# CUDA_VISIBLE_DEVICES=1 nohup python tfidf.py > tfidf.log 2>&1 &
