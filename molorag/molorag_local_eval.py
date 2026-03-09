"""
MoLoRAG Local Evaluation Script
This script provides a self-contained reproduction of the MoLoRAG retrieval engine.
It implements the hierarchical graph-based traversal (Algorithm 1) and evaluates 
performance on MMLongBench and LongDocURL.
"""
import os
import json
import torch
import gc
import re
import numpy as np
import fitz  # PyMuPDF
import networkx as nx
from PIL import Image
from math import log2
from transformers import (
    CLIPProcessor, CLIPModel, AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# --- 1. MacBook Optimization & Paths ---

# For Apple Silicon MacBook, "mps" is the target device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Local Workspace Paths
BASE_PATH = "/Users/niteeshkumar/Documents/molorag"
REPO_PATH = os.path.join(BASE_PATH, "molorag_project/repo_molorag")
DATASET_DIR = os.path.join(REPO_PATH, "dataset")

# Mapping local datasets
DATASETS = [
    ("MMLongBench", os.path.join(DATASET_DIR, "samples_MMLong.json"), os.path.join(DATASET_DIR, "MMLong")),
    ("LongDocURL", os.path.join(DATASET_DIR, "samples_LongDocURL.json"), os.path.join(DATASET_DIR, "LongDocURL")),
]

# --- 2. Retrieval Framework ---

class DocumentGraphIndex:
    """
    Handles the indexing of long PDFs by embedding pages into a shared visual space
    and building a proximity graph based on semantic similarity.
    """
    def __init__(self, model_name="openai/clip-vit-large-patch14", threshold=0.4, device=DEVICE):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.threshold = threshold
        self.graph = nx.Graph()
        self.page_images, self.embeddings = [], []

    def load_pdf(self, pdf_path):
        """Extracts images for each page of the input PDF."""
        doc = fitz.open(pdf_path)
        self.page_images = []
        for page in doc:
            pix = page.get_pixmap()
            self.page_images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        doc.close()

    def generate_embeddings(self):
        """Generates CLIP embeddings for each document page."""
        self.embeddings = []
        self.model.eval()
        with torch.no_grad():
            for img in self.page_images:
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                outputs = self.model.get_image_features(**inputs)
                features = outputs if torch.is_tensor(outputs) else outputs[0]
                if features.ndim > 2: features = features[:, 0, :]
                image_features = features / features.norm(p=2, dim=-1, keepdim=True)
                self.embeddings.append(image_features.cpu().numpy().flatten())

    def build_graph(self):
        """Constructs a graph where edges represent high semantic similarity between pages."""
        num_pages = len(self.embeddings)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(num_pages))
        for i in range(num_pages):
            for j in range(i + 1, num_pages):
                similarity = np.dot(self.embeddings[i], self.embeddings[j])
                if similarity >= self.threshold:
                    self.graph.add_edge(i, j, weight=float(similarity))

class LogicAwareRetriever:
    """
    Uses a Visual Language Model (Qwen2.5-VL) to assign a logical relevance score
    between a natural language query and a specific document page image.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", device=DEVICE):
        self.device = device
        # MacBook/MPS optimization: Use bfloat16 for high-speed local inference
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            device_map=self.device
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    def get_logical_score(self, question, page_image):
        """Asks the VLM to rank the relevance of the image on a scale of 1-5."""
        prompt = f"""# GOAL #
You are an Retrieval Expert, and your task is to evaluate how relevant the input document page is to the given query.
...
# QUERY #
{question}
Please generate just a single number (1-5) representing your relevance judgment."""

        messages = [{"role": "user", "content": [{"type": "image", "image": page_image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=5)
            out = self.processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        score_match = re.search(r'[1-5]', out)
        score = int(score_match.group(0)) if score_match else 3
        return (score - 1) / 4.0

class MoLoRAGTraversal:
    def __init__(self, index, retriever, w=3, n_hop=4):
        self.index, self.retriever = index, retriever
        self.w, self.n_hop = w, n_hop

    def run_traversal(self, query):
        # CLIP Text Embedding - Truncate to 77 tokens for CLIP limit
        inputs = self.index.processor(text=[query], return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.index.device)
        with torch.no_grad():
            text_features = self.index.model.get_text_features(**inputs)
            text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
            q_emb = text_features.cpu().numpy().flatten()

        sem_scores = [float(np.dot(q_emb, p_emb)) for p_emb in self.index.embeddings]
        exploration_set = np.argsort(sem_scores)[-self.w:].tolist()
        visited, results_list = set(exploration_set), []

        for _ in range(self.n_hop + 1):
            candidates = []
            for node in exploration_set:
                l_score = self.retriever.get_logical_score(query, self.index.page_images[node])
                results_list.append((node, (sem_scores[node] + l_score) / 2.0))
                for n in self.index.graph.neighbors(node):
                    if n not in visited:
                        visited.add(n); candidates.append(n)
            if not candidates: break
            exploration_set = candidates[:self.w]
        return sorted(list(set(results_list)), key=lambda x: x[1], reverse=True)

# --- 3. Evaluation Toolkit ---

def calculate_metrics(prediction, ground_truth):
    ranked = [p for p, s in prediction]
    res = {}
    for k in [1, 3, 5]:
        cur = ranked[:k]
        hits = len(set(cur) & set(ground_truth))
        recall = (hits / len(ground_truth)) * 100 if ground_truth else 0.0
        precision = (hits / k) * 100
        dcg = sum([1.0 / log2(i + 2) for i, p in enumerate(cur) if p in ground_truth])
        idcg = sum([1.0 / log2(i + 2) for i in range(min(k, len(ground_truth)))])
        ndcg = (dcg / idcg) * 100 if idcg > 0 else 0.0
        mrr = next(( (1.0 / (i + 1)) * 100 for i, p in enumerate(cur) if p in ground_truth), 0.0)
        res[k] = {"Recall": recall, "Precision": precision, "NDCG": ndcg, "MRR": mrr}
    return res

# --- 4. Main Execution ---

def main():
    print("Initialising MoLoRAG Local Engine...")
    retriever = LogicAwareRetriever()
    stats = {name: {k: {m: [] for m in ["Recall", "Precision", "NDCG", "MRR"]} for k in [1, 3, 5]} for name, _, _ in DATASETS}

    for name, json_path, pdf_root in DATASETS:
        if not os.path.exists(json_path):
            print(f"Skipping {name}: Metadata not found at {json_path}")
            continue

        print(f"\n--- Evaluating {name} ---")
        with open(json_path) as f:
            samples = json.load(f)[:5]  # Running small test set first

        for s in tqdm(samples):
            pdf_file = s.get('doc_id', s.get('pdf_path','').split('/')[-1])
            if not pdf_file.lower().endswith(".pdf"): pdf_file += ".pdf"
            pdf_path = os.path.join(pdf_root, pdf_file)
            
            if not os.path.exists(pdf_path):
                continue

            # Process Index
            idx = DocumentGraphIndex()
            idx.load_pdf(pdf_path)
            idx.generate_embeddings()
            idx.build_graph()

            # Run MoLoRAG
            trav = MoLoRAGTraversal(idx, retriever)
            pred = trav.run_traversal(s['question'])

            # Ground Truth Parsing
            gt_raw = s.get('evidence_pages', [])
            gt = [int(p)-1 for p in (eval(gt_raw) if isinstance(gt_raw, str) else gt_raw)]

            # Metrics
            sample_metrics = calculate_metrics(pred, gt)
            for k in [1, 3, 5]:
                for m in ["Recall", "Precision", "NDCG", "MRR"]:
                    stats[name][k][m].append(sample_metrics[k][m])

            # Cleanup
            del idx, trav
            gc.collect()
            if DEVICE == "cuda": torch.cuda.empty_cache()

    # Reporting
    print("\nTable 3: Local Retrieval Performance (in %)")
    header = f"{'K':<3} {'Method':<12}"
    for name, _, _ in DATASETS:
        header += f" {name + ' (R/P/N/M)':<35}"
    print(header)
    
    for k in [1, 3, 5]:
        row = f"  {k:<3} {'MoLoRAG':<12}"
        for name, _, _ in DATASETS:
            vals = [np.mean(stats[name][k][m] or [0]) for m in ["Recall", "Precision", "NDCG", "MRR"]]
            row += f" {'/'.join([f'{v:.2f}' for v in vals]):<35}"
        print(row)
    
    # Save results to file
    with open(os.path.join(BASE_PATH, "evaluation_results.json"), 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"\nFull results saved to {os.path.join(BASE_PATH, 'evaluation_results.json')}")

if __name__ == "__main__":
    main()
