# MoLoRAG: Multi-modal Logical Retrieval Augmented Generation (Standard Version)

This folder contains the reproduction code for the standard MoLoRAG system as described in the original paper.

## 0. Links
- **Original Paper Repository**: [https://github.com/WxxShirley/MoLoRAG](https://github.com/WxxShirley/MoLoRAG)
- **Reproduction Repository**: [Your GitHub Link Here]

## 1. Dependencies
- Python 3.10+
- PyTorch 2.4+
- Transformers
- CLIP (OpenAI)
- PyMuPDF (fitz)
- NetworkX
- Pillow
- tqdm

Install all dependencies via:
```bash
pip install -r requirements.txt
```

## 2. Data Download Instructions
The system uses the **MMLongBench-Doc** and **LongDocURL** datasets.
1. Metadata and sample PDFs are already included in the `dataset/` folder at the root of the repository.
2. For the full dataset (6.5GB+), download the PDFs from the [official repository](https://github.com/WxxShirley/MoLoRAG) and place them in `dataset/MMLong` and `dataset/LongDocURL`.

## 3. Preprocessing (Indexing)
To reproduce the document graph and embeddings:
```bash
# This logic is integrated into the evaluation script for zero-shot reproduction.
# If using the original codebase:
python main.py --mode index --data_path dataset/MMLong
```

## 4. Evaluation Command
To run the MoLoRAG hierarchical retrieval engine on your local machine:
```bash
python molorag_local_eval.py
```
This script will output performance metrics (Recall, Precision, NDCG, MRR) for Top-1, 3, and 5.

## 5. Pretrained Models
- **Visual Encoder**: `openai/clip-vit-large-patch14`
- **Logical Reasoner**: `Qwen/Qwen2.5-VL-3B-Instruct` (Optimized for bfloat16 local inference)
