# MoLoRAG Dataset Evaluation Report

## 1. Executive Summary
This report provides a detailed breakdown of the datasets used to evaluate the **MoLoRAG** and **MoLoRAG+** systems. The evaluation focuses on **MMLongBench-Doc** and **LongDocURL**, which represent the state-of-the-art in long-form, multi-modal document retrieval and reasoning.

To ensure reproducibility, we have included the full metadata and a representative subset of documents that are directly used by the local evaluation scripts.

---

## 2. Dataset Information

### A. MMLongBench-Doc
- **Description**: A comprehensive benchmark for multi-disciplinary, long-form PDF documents. Each document typically spans 20-50 pages and contains complex layouts, including tables, charts, and hierarchical headings.
- **Key Challenges**: Requires cross-page reasoning and following logical "pointers" (e.g., "See Section 4.2", "Refer to Figure 1").
- **Metadata**: `dataset/samples_MMLong.json`
- **Included Documents (Used in Local Eval)**:
    - `PH_2016.06.08_Economy-Final.pdf`: A socio-economic research report.
    - `Independents-Report.pdf`: A political analysis report.
    - `0e94b4197b10096b1f4c699701570fbf.pdf`: An academic tutorial/technical paper.

### B. LongDocURL
- **Description**: A dataset specifically designed to test the retrieval of information from extremely long documents where semantic-only retrieval (embedding matching) is prone to failure.
- **Key Challenges**: High document volume and the need for layout-aware retrieval.
- **Metadata**: `dataset/samples_LongDocURL.json`
- **Included Documents (Used in Local Eval)**:
    - `4026369.pdf`
    - `4125651.pdf`
    - `4145761.pdf`
    - `4091930.pdf`
    - `4088207.pdf`

---

## 3. Training vs. Testing Split

### Evaluation Mode (Zero-Shot)
The standard MoLoRAG evaluation is **Zero-Shot**. It uses the provided metadata JSON files as query sets.
- **Query Source**: `question` field in the JSON.
- **Ground Truth**: `answer` and `evidence_pages` fields.
- **Evaluation**: The system builds a graph index of the PDF and attempts to traverse it to find the answer.

### MoLoRAG+ (Distillation Mode)
In the "Plus" version, the **PDF document pool** itself serves as the training data source.
- **Synthetic Training**: We use a Teacher model (e.g., Qwen2.5-VL-72B) to generate (Question, Image, Relevance Score) triplets from random pages of the included PDFs.
- **Fine-Tuning**: A student model (Qwen2.5-VL-3B) is trained using LoRA on these synthetic triplets.
- **Final Test**: The fine-tuned model is then evaluated on the official test queries in the JSON files.

---

## 4. Usage Instructions
1. **Local Demo**: The included PDFs allow you to run the first 5 samples of each benchmark immediately using the provided evaluation scripts.
2. **Full Scale**: To run the full benchmark, please download the complete 6.5GB document sets from the [Official MoLoRAG Repository](https://github.com/WxxShirley/MoLoRAG).
