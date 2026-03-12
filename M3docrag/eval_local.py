"""
M3DocRAG End-to-End Evaluation Script
-------------------------------------
Implements the multi-modal retrieval and QA pipeline using ColPali and Qwen-VL.
"""

import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor as QwenProcessor
from colpali_engine.models import ColPali, ColPaliProcessor
from qwen_vl_utils import process_vision_info

# --- Retrieval Logic ---
class DocumentRetriever:
    def __init__(self, encoder, processor, device):
        self.encoder = encoder 
        self.processor = processor 
        self.device = device 

    def compute_scores(self, query, all_embeds):
        queries = self.processor.process_queries(queries=[query]).to(self.device)
        with torch.no_grad():
            query_embeds = self.encoder(**queries)
        
        all_embeds = all_embeds.to(device=self.device, dtype=query_embeds.dtype)
        with torch.no_grad():
            scores = self.processor.score_multi_vector(query_embeds, all_embeds)
            if len(scores.shape) > 1:
                scores = scores[0]
        return scores.cpu()
    
    def retrieve(self, query, all_embeds, top_k=5):
        scores = self.compute_scores(query, all_embeds)
        top_indices = scores.argsort(dim=-1, descending=True)[:top_k].tolist()
        top_scores = scores[top_indices].tolist()
        return [idx+1 for idx in top_indices], top_scores

# --- QA Logic ---
def get_vlm_response(model, processor, question, image_paths, device):
    msgs = [dict(type='image', image=f"file://{os.path.abspath(p)}") for p in image_paths]
    msgs.append(dict(type='text', text=question))
    messages = [{"role": "user", "content": msgs}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors='pt').to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    generated_ids_trimmed = [ids[len(in_ids):] for in_ids, ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    return output_text

def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Running evaluation on {device}")

    # 1. Load Retriever
    print("[*] Loading ColPali Retriever...")
    retriever_model = ColPali.from_pretrained("vidore/colpaligemma-3b-pt-448-base", torch_dtype=torch.float16, device_map="auto")
    retriever_model.load_adapter("vidore/colpali-v1.2")
    retriever_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
    doc_retriever = DocumentRetriever(retriever_model, retriever_processor, device)

    # 2. Load VLM
    print("[*] Loading Qwen2.5-VL-3B...")
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.float16, device_map="auto")
    vlm_processor = QwenProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # 3. Load Data & Embeddings
    samples_file = f"dataset/samples_{args.dataset}.json"
    samples = json.load(open(samples_file, 'r'))
    
    results = []
    for sample in tqdm(samples, desc="Processing Pipeline"):
        doc_id = sample['doc_id'].replace('.pdf', '')
        emb_path = f"M3docrag/tmp/tmp_embs/{args.dataset}/{doc_id}.pt"
        
        if not os.path.exists(emb_path):
            continue

        # Retrieval
        doc_embs = torch.load(emb_path, map_location='cpu')
        ranked_pages, _ = doc_retriever.retrieve(sample['question'], doc_embs, top_k=args.top_k)
        
        # QA
        image_paths = [f"M3docrag/tmp/tmp_imgs/{args.dataset}/{doc_id}-{p}.png" for p in ranked_pages]
        response = get_vlm_response(vlm_model, vlm_processor, sample['question'], image_paths, device)
        
        sample['raw_response'] = response
        sample['pages_ranking'] = ranked_pages
        results.append(sample)

    # 4. Save Results
    os.makedirs("results", exist_ok=True)
    out_file = f"results/m3docrag_{args.dataset}_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"[DONE] Results saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MMLong")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()
    main(args)
