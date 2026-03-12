# MoLoRAG: Multi-modal Logic-aware RAG

MoLoRAG is a high-performance retrieval system for long-form multi-modal documents. This repository contains the core implementation, evaluation suite, and training scripts.

## 1. Quick Start
### Dependencies
Ensure you have Python 3.10+ and install the required packages:
```bash
pip install torch torchvision transformers qwen_vl_utils peft accelerate bitsandbytes PyMuPDF networkx pillow tqdm scikit-learn
```

### Data Download
The system expects a `dataset/` folder at the root:
1. Download the **MMLongBench** and **LongDocURL** datasets.
2. Place the PDF documents in `dataset/MMLong/` and `dataset/LongDocURL/`.
3. Place the metadata JSON files (`samples_*.json`) in the `dataset/` root.

## 2. Project Structure

```text
molorag/
├── baseline/              # Original MoLoRAG core implementation
├── dataset/               # Consolidated dataset folder
├── M3docrag/              # M3DocRAG experimental notebooks
└── molorag/               # Research implementations
    ├── molorag_standard/  # Contains molorag_local_eval.py
    └── molorag_plus/      # Enhanced implementation files
```

## 3. Project Components
### [molorag_standard](./molorag/molorag_standard)
Original implementation of the hierarchical graph-based traversal.
- **Evaluation**: `python molorag/molorag_standard/molorag_local_eval.py`

### [molorag_plus](./molorag/molorag_plus)
Enhanced version featuring a fine-tuned VLM for logical relevance scoring.
- **Preprocessing**: Generate training data from documents.
  ```bash
  python molorag/molorag_plus/generate_data_qwen.py
  ```
- **Training**: Fine-tune the VLM using LoRA.
  ```bash
  python molorag/molorag_plus/train_qwen_lora.py
  ```
- **Evaluation**: Run evaluation with the fine-tuned adapter.
  ```bash
  python molorag/molorag_plus/molorag_v2_eval.py
  ```

### [baseline](./baseline)
Official reproduction scripts for paper baseline results.
- **Run QA**: `python baseline/main.py --dataset MMLong --model_name QwenVL-3B`
- **Evaluation**: `python baseline/main_eval.py --dataset MMLong`

### [M3docrag](./M3docrag)
Collection of Jupyter notebooks and standalone scripts for M3DocRAG experiments and baseline testing.
- **Preprocessing**: `python M3docrag/index_local.py --dataset MMLong`
- **Evaluation**: `python M3docrag/eval_local.py --dataset MMLong --top_k 3`

## 4. Models Used
- **Vision-Language Model**: `Qwen/Qwen2.5-VL-3B-Instruct`
- **Embedding Model**: `openai/clip-vit-large-patch14`

📜 **Citation**
If you use this code, please refer to the original MoLoRAG paper and this reproduction effort.
