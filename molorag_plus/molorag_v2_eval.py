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
from transformers import CLIPProcessor, CLIPModel
from retrieve_plus_v2 import MoLoRAGPlusV2Retriever
from tqdm import tqdm

# --- 1. MacBook Optimization & Paths ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# Local Workspace Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
# The script expects a 'dataset' folder in the current directory or the parent
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
if not os.path.exists(DATASET_DIR):
    DATASET_DIR = os.path.join(BASE_PATH, "dataset")

DATASETS = [
    ("MMLongBench", os.path.join(DATASET_DIR, "samples_MMLong.json"), os.path.join(DATASET_DIR, "MMLong")),
    ("LongDocURL", os.path.join(DATASET_DIR, "samples_LongDocURL.json"), os.path.join(DATASET_DIR, "LongDocURL")),
]

# --- 2. Existing Traversal & Graph Classes (Re-used) ---
class DocumentGraphIndex:
    def __init__(self, model_name="openai/clip-vit-large-patch14", threshold=0.4, device=DEVICE):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.threshold = threshold
        self.graph = nx.Graph()
        self.page_images, self.embeddings = [], []

    def load_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        self.page_images = []
        for page in doc:
            pix = page.get_pixmap()
            self.page_images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        doc.close()

    def generate_embeddings(self):
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
        num_pages = len(self.embeddings)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(num_pages))
        for i in range(num_pages):
            for j in range(i + 1, num_pages):
                similarity = np.dot(self.embeddings[i], self.embeddings[j])
                if similarity >= self.threshold:
                    self.graph.add_edge(i, j, weight=float(similarity))

class MoLoRAGTraversal:
    def __init__(self, index, retriever, w=3, n_hop=4):
        self.index, self.retriever = index, retriever
        self.w, self.n_hop = w, n_hop

    def run_traversal(self, query):
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

def main():
    print("Initialising MoLoRAG+ v2 Evaluation Engine...")
    retriever = MoLoRAGPlusV2Retriever()
    stats = {name: {k: {m: [] for m in ["Recall", "Precision", "NDCG", "MRR"]} for k in [1, 3, 5]} for name, _, _ in DATASETS}

    for name, json_path, pdf_root in DATASETS:
        print(f"\n--- Evaluating {name} (Plus v2) ---")
        with open(json_path) as f:
            samples = json.load(f)[:5]

        for s in tqdm(samples):
            pdf_file = s.get('doc_id', s.get('pdf_path','').split('/')[-1])
            if not pdf_file.lower().endswith(".pdf"): pdf_file += ".pdf"
            pdf_path = os.path.join(pdf_root, pdf_file)
            if not os.path.exists(pdf_path): continue

            idx = DocumentGraphIndex()
            idx.load_pdf(pdf_path)
            idx.generate_embeddings()
            idx.build_graph()

            trav = MoLoRAGTraversal(idx, retriever)
            pred = trav.run_traversal(s['question'])

            gt_raw = s.get('evidence_pages', [])
            gt = [int(p)-1 for p in (eval(gt_raw) if isinstance(gt_raw, str) else gt_raw)]

            sample_metrics = calculate_metrics(pred, gt)
            for k in [1, 3, 5]:
                for m in ["Recall", "Precision", "NDCG", "MRR"]:
                    stats[name][k][m].append(sample_metrics[k][m])
            del idx, trav; gc.collect()

    print("\nTable 4: MoLoRAG+ v2 Performance (%)")
    for name, _, _ in DATASETS:
        print(f"\nDataset: {name}")
        for k in [1, 3, 5]:
            vals = [np.mean(stats[name][k][m] or [0]) for m in ["Recall", "Precision", "NDCG", "MRR"]]
            print(f"K={k}: Recall={vals[0]:.2f}, Precision={vals[1]:.2f}, NDCG={vals[2]:.2f}, MRR={vals[3]:.2f}")

if __name__ == "__main__":
    main()
