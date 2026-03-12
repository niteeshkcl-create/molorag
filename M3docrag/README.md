# M3DocRAG Experimental Setup

This module contains the M3DocRAG baseline reproduction suite, primarily utilizing **ColPali** for multi-modal retrieval and **Qwen2.5-VL-3B** for logic-aware reasoning.

---

## 1. Dependencies
Install notebook-specific requirements:
```bash
pip install -r M3docrag/requirements.txt
```
*Note: Linux/Colab users may also need `sudo apt-get install poppler-utils` for PDF processing.*

## 2. Data Download Instructions
The notebooks are configured to download metadata and PDF subsets automatically from HuggingFace (`xxwu/MoLoRAG`). 

To download manually for local use:
```bash
# Metadata
huggingface-cli download xxwu/MoLoRAG --repo-type dataset --include "samples_*.json" --local-dir ./dataset

# PDFs
huggingface-cli download xxwu/MoLoRAG --repo-type dataset --include "MMLong/*.pdf" --local-dir ./dataset
```

## 3. Preprocessing (Indexing)
Visual encoding is handled by ColPali. You can run this via the `03_m3docrag_baseline_experiment.ipynb` notebook or the extracted script:
```bash
# From repository root
python M3docrag/index_local.py --dataset MMLong
```

## 4. Training (LoRA)
This baseline uses a pretrained **ColPali v1.2** retriever and **Qwen2.5-VL-3B-Instruct**. For fine-tuning tasks, refer to the [MoLoRAG+ setup](../molorag_plus).

## 5. Evaluation Command
Evaluation is performed in the final cells of the notebooks, computing EM, Accuracy, and F1. 
To run the full pipeline locally:
```bash
# Execute the reproduction script (extracted from notebook)
python M3docrag/MoLoRAG_Reproduction_Final.py
```

## 6. Pretrained Models
- **Retriever**: [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2)
- **VLM**: [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

---

## Notebooks Guide
- **`MoLoRAG_Reproduction_Final.ipynb`**: The master notebook for full E2E baseline reproduction.
- **`03_m3docrag_baseline_experiment.ipynb`**: Detailed retrieval experiments with memory optimizations for local systems.
- **`textRag.ipynb`**: Comparative analysis with traditional text-based RAG.
