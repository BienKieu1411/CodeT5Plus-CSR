# CodeT5Plus-CSR: A Title Generation Pipeline with Contrastive Learning, Self-Improvement and ROUGE-based Reranking

[![HuggingFace Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/BienKieu)  
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-blue)](https://huggingface.co/datasets/BienKieu)  
[![Paper](https://img.shields.io/badge/Paper-ESEM%202024-orange)](https://arxiv.org/abs/2406.15633)

---

## Overview

**CodeT5Plus-CSR** is a research project extending the [FILLER framework (ESEM 2024)](https://arxiv.org/abs/2406.15633) for Stack Overflow title generation. The project enhances the generation quality by combining:

- Fine-tuning **CodeT5+** on Stack Overflow posts  
- Applying **Self-Improvement** to mitigate exposure bias  
- Using a **Contrastive Encoder - Decoder** to align title semantics  
- Performing **Graph-Based Reranking** with **ROUGE-L-weighted PageRank**  

---

## Project Structure

```text
CodeT5Plus-CSR/
├── generate_candidates/
│   ├── gen.py
│   ├── tfidf.py       
│   └── rougl.py                    
│
├── self_improvement/
│   └── self_improvement.py             
│
├── train_with_fixed_input_span_allocation /
│   ├── Contrastive /
│       ├── train.py
│       ├── train.sh
│       ├── test.py
│       └── test.py
│   ├── codeT5Plus /
│       ├── train.py
│       ├── train.sh
│       ├── test.py
│       └── test.py
│   └── codeT5base /
│       ├── train.py
│       ├── train.sh
│       ├── test.py
│       └── test.py
│
├── train_without_fixed_input_span_allocation /
│   ├── Contrastive /
│       ├── train.py
│       ├── train.sh
│       ├── test.py
│       └── test.py
│   ├── codeT5Plus /
│       ├── train.py
│       ├── train.sh
│       ├── test.py
│       └── test.py
│   └── codeT5base /
│       ├── train.py
│       ├── train.sh
│       ├── test.py
│       └── test.py
│    
├── Result.xlsx                     
├── requirements.txt
└── README.md
```

---

## Datasets

Stack Overflow posts across 4 programming languages (Java, Python, JS, C#), with:

- `desc`: natural language problem description  
- `code`: source code snippet  
- `title`: ground-truth post title  

Available on Hugging Face:

- [BienKieu/data](https://huggingface.co/datasets/BienKieu/data)  
- [BienKieu/data_merged](https://huggingface.co/datasets/BienKieu/data_merged)  

---

## Pipeline

### 1. Fine-Tune CodeT5+

```bash
bash train_without_fixed_input_span_allocation/codeT5Plus/train.sh
```

### 2. Generate Title Candidates

```bash
python generate_candidates/gen.py
```

### 3. Rerank by ROUGE-L PageRank

```bash
python generate_candidates/rougl.py 
```

---

## Contrastive Learning

Contrastive encoder-decoder learning is integrated directly into the training phase. No separate reranking step is needed. To train the contrastive-enhanced CodeT5+ model:

```bash
bash train_without_fixed_input_span_allocation/Contrastive/train.sh
```

---

## Evaluation

Supported metrics:

- ROUGE-1, ROUGE-2, ROUGE-L (F1)
  
---

## Resources

- [FILLER Paper (ESEM 2024)](https://arxiv.org/abs/2406.15633)  
- Hugging Face: [BienKieu](https://huggingface.co/BienKieu)  
- Dataset: [BienKieu/data](https://huggingface.co/datasets/BienKieu/data)  

---

## Author

**Kieu Giang Bien**  
GitHub: [@BienKieu1411](https://github.com/BienKieu1411)  
Hugging Face: [BienKieu](https://huggingface.co/BienKieu)
