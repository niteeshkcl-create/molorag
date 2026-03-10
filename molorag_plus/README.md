# MoLoRAG+: Distilled Logical Retrieval Augmented Generation

MoLoRAG+ is an optimized version of MoLoRAG that uses knowledge distillation to train a smaller, local retriever for efficient inference on consumer hardware.

## 0. Links
- **Original Paper Repository**: [https://github.com/WxxShirley/MoLoRAG](https://github.com/WxxShirley/MoLoRAG)
- **Reproduction Repository**: [Your GitHub Link Here]

## 1. Dependencies
- Python 3.10+
- PyTorch (with MPS support for Mac)
- Transformers & PEFT (HuggingFace)
- qwen_vl_utils
- fitz (PyMuPDF)
- datasets
- tqdm

Install all dependencies via:
```bash
pip install -r requirements.txt
```

## 2. Data Download Instructions
The system uses the **MMLongBench-Doc** and **LongDocURL** datasets for distillation and evaluation.
1. Metadata and sample PDFs are already included in the `dataset/` folder at the root of the repository.
2. For the full dataset (6.5GB+), download the PDFs from the [official repository](https://github.com/WxxShirley/MoLoRAG) and place them in `dataset/MMLong` and `dataset/LongDocURL`.

## 3. Data Generation (Distillation)
To generate the training triplets (Question, Image, Score) using a Qwen teacher model:
```bash
python generate_data_qwen.py
```
This script samples document pages and uses the teacher model to assign logical relevance scores.

## 4. Training Command (LoRA Fine-tuning)
To fine-tune the Qwen2.5-VL-3B model on your local MacBook:
```bash
python train_qwen_lora.py
```
This script utilizes LoRA (Rank 8) and is optimized for Apple Silicon (MPS).

## 5. Evaluation Command
To evaluate the fine-tuned MoLoRAG+ v2 model:
```bash
python molorag_v2_eval.py
```
This script tests the model on MMLongBench and LongDocURL datasets and outputs Recall, NDCG, and MRR.

## 6. Pretrained Models
- **Teacher**: `Qwen/Qwen2.5-VL-3B-Instruct` (Local distillation)
- **Student**: `Qwen/Qwen2.5-VL-3B` (Fine-tuned with LoRA adapters)
