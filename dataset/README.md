# MoLoRAG Dataset Breakdown

This folder contains the metadata and sample documents for reproducing the MoLoRAG benchmarks on **MMLongBench** and **LongDocURL**.

## 1. Dataset Summary & Info
MoLoRAG is evaluated on long-form, multi-modal documents. The two primary datasets used are:

### A. MMLongBench
- **Source**: Collected from a wide range of PDF documents including academic papers, technical manuals, and business reports.
- **Total Samples**: 1,000+ question-answer pairs (evaluated on the first 5 for local testing).
- **Format**: Multiple-choice and open-ended questions targeting specific pages or logical relationships across the document.
- **Logical Complexity**: Requires a "Hierarchical Traversal" strategy to follow logical links (e.g., "See Figure 2", "Continued on Page 45").

### B. LongDocURL
- **Source**: Documents with complex layouts, tables, and diverse URL-based references.
- **Total Samples**: 1,000+ question-answer pairs.
- **Focus**: Navigating and retrieving information from extremely long documents where semantic-only retrieval often fails.

## 2. Training vs. Testing Split
### Standard MoLoRAG (Zero-Shot)
The standard MoLoRAG engine is **Zero-Shot**. It uses an off-the-shelf CLIP encoder and the Qwen2.5-VL-3B logic reasoner. No training dataset is required for this version; it relies on the document graph constructed at inference time.

### MoLoRAG+ (Distilled Version)
The "Plus" version uses **Knowledge Distillation**. 
- **Training Source**: We utilize the *document pool* itself (the PDFs) as the source.
- **Process**: A larger teacher model (e.g., Qwen2.5-VL-7B or 3B) generates synthetic (Question, Image, Score) triplets from randomly sampled pages.
- **Result**: These triplets form a synthetic training set (e.g., `training_data_qwen.jsonl`) used to fine-tune the student model (Qwen2.5-VL-3B) with LoRA.
- **Testing**: Evaluation is performed on the official test queries provided in `samples_MMLong.json` and `samples_LongDocURL.json`.

## 3. Directory Structure
```text
dataset/
├── MMLong/                # Sample PDF documents from MMLongBench
├── LongDocURL/            # Sample PDF documents from LongDocURL
├── samples_MMLong.json    # Full metadata and ground-truth for MMLongBench
└── samples_LongDocURL.json # Full metadata and ground-truth for LongDocURL
```

## 4. Full Dataset Download
Due to the large size of the full PDFs (~6.5GB), we have included only a few samples for demonstration. To run full-scale benchmarks, please download the complete document sets from the official MoLoRAG repository:
- **Repo**: [https://github.com/WxxShirley/MoLoRAG](https://github.com/WxxShirley/MoLoRAG)
- **Path**: `dataset/MMLong` and `dataset/LongDocURL`
