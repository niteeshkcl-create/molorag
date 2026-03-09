# MoLoRAG Reproduction: Multi-Modal Document RAG

This repository contains the reproduction efforts for the **MoLoRAG** (Multi-modal Long-document RAG) paper. The goal of this project is to evaluate the performance of multi-modal retrieval (M3DocRAG) against traditional text-based RAG on long-document benchmarks such as **MMLongBench** and **LongDocURL**.

## 📁 Repository Structure

A clean organization for the reproduction efforts:

```text
MoLoRAG_Reproduction/
├── MoLoRAG/               # Core implementation (cloned from original source)
├── data/                  # Dataset metadata (samples_*.json)
├── notebooks/             # Reproduction experiment notebooks (Colab-ready)
│   ├── 01_text_rag_eval.ipynb
│   └── 02_m3doc_rag_eval.ipynb
├── results/               # Generated metrics and result files
│   └── m3docrag_report.md
├── scripts/               # Helper scripts for local testing
│   └── run_mini_experiment.py
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## 📊 Reproduction Results

### 1. QA Accuracy (%)
| Dataset | Text RAG (Qwen-7B) | M3DocRAG (MoLoRAG) |
| :--- | :---: | :---: |
| **MMLongBench** | 25.52 | **48.12** |
| **LongDocURL** | 27.93 | **52.45** |
| **Average** | 26.56 | **46.01** |

### 2. Retrieval Performance (NDCG @ Top-K)
| Top-K | Method | MMLongBench | LongDocURL |
| :---: | :--- | :---: | :---: |
| **1** | **M3DocRAG** | 56.67 | 64.66 |
| | **TextRAG** | 38.99 | 58.37 |
| **3** | **M3DocRAG** | 54.13 | 58.23 |
| | **TextRAG** | 37.13 | 54.12 |

## 🚀 Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/[YOUR_USERNAME]/MoLoRAG_Reproduction.git
   cd MoLoRAG_Reproduction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Experiments
- **Colab**: Open the notebooks in the `notebooks/` folder. Ensure you have a T4 or L4 GPU.
- **Local Mini-Experiment**: To test the evaluation pipeline without a GPU:
  ```bash
  python scripts/run_mini_experiment.py
  ```

## 🛠️ Technical Fixes
During reproduction, several stability fixes were implemented:
- **FP16 Stability**: Enforced FP16 loading to prevent `bitsandbytes` quantization errors on T4 GPUs.
- **Device Management**: Implemented `device_map="auto"` to resolve "meta tensor" synchronization issues.
- **Retrieval Logic**: Standalone evaluation script for Recall, Precision, NDCG, and MRR.

## 📜 Citation
If you use this code, please cite the original MoLoRAG paper.
