import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from tqdm import tqdm
from rouge import Rouge, FilesRouge
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from multiprocessing import Pool, cpu_count

rouge = Rouge()

def normalize(v):
    v = np.asarray(v)
    if len(v.shape) == 1:
        v = v.reshape(-1, 1)
    if np.all(v == v[0]):
        return np.zeros_like(v).flatten()
    return MinMaxScaler().fit_transform(v).flatten()

def compute_rouge_pair(args):
    i, j, cands = args
    try:
        s1 = rouge.get_scores(cands[i], cands[j])[0]['rouge-l']['f']
        s2 = rouge.get_scores(cands[j], cands[i])[0]['rouge-l']['f']
        return (i, j, (s1 + s2) / 2)
    except:
        return (i, j, 0.0)

def rerank_by_pagerank_rougel(candidates, pagerank_alpha=0.23):
    N = len(candidates)
    sim_matrix = np.zeros((N, N))

    tasks = [(i, j, candidates) for i in range(N) for j in range(i + 1, N)]
    with Pool(processes=min(6, cpu_count())) as pool:
        results = pool.map(compute_rouge_pair, tasks)
    for i, j, score in results:
        sim_matrix[i][j] = sim_matrix[j][i] = score

    try:
        graph = nx.from_numpy_array(sim_matrix)
        pagerank_rougel_dict = nx.pagerank(graph, alpha=pagerank_alpha)
        pagerank_rougel_score = np.array([pagerank_rougel_dict[i] for i in range(N)])
        pagerank_rougel_score = normalize(pagerank_rougel_score)
    except:
        pagerank_rougel_score = np.zeros(N)

    return candidates[int(np.argmax(pagerank_rougel_score))]

def run_with_pagerank_rougel_only(output_dir="../rerank_result/rougel", group_size=200):
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
            chosen = rerank_by_pagerank_rougel(group)
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
        print(f"{lang} ROUGE-1: {scores['rouge-1']['f']:.4f} | "
              f"ROUGE-2: {scores['rouge-2']['f']:.4f} | "
              f"ROUGE-L: {scores['rouge-l']['f']:.4f}")

if __name__ == "__main__":
    run_with_pagerank_rougel_only(group_size=200)

# CUDA_VISIBLE_DEVICES=0 nohup python rougel.py > rougel.log 2>&1 &
# pkill -f rougel.py