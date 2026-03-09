import torch 
from tqdm import tqdm 
from colpali_engine.models import ColPali, ColPaliProcessor
import argparse
import sys 
sys.path.append("../")
from utils import load_all_doc_embeddings
import json 
import os 

class DocumentRetriever:
    def __init__(self, encoder, processor, device, batch_size=128):
        self.encoder = encoder 
        self.processor = processor 
        self.device = device 
        self.batch_size = batch_size     

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
    
    def base_retrieve(self, query, all_embeds, top_k=5):
        scores = self.compute_scores(query, all_embeds)
        top_indices = scores.argsort(dim=-1, descending=True)[:top_k].tolist()
        top_scores = scores[top_indices].tolist()
        return [idx+1 for idx in top_indices], top_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MMLong")
    parser.add_argument("--emb_root", type=str, default="../tmp/tmp_embs")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    doc2emb = load_all_doc_embeddings(f"{args.emb_root}/{args.dataset}")
    
    model_name = "vidore/colpali-v1.2"
    print(f"Loading ColPali via PaliGemma base on {device}...")
    
    print(f"Loading ColPali via PaliGemma base on {device}...")
    
    # Switch to float16 to avoid bitsandbytes AssertionError on T4 GPUs
    # Removed device_map to avoid meta-tensor NotImplementedError during .to(device)
    model = ColPali.from_pretrained(
        "vidore/colpaligemma-3b-pt-448-base",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load the adapter into the ColPali instance
    print(f"Loading adapter: {model_name}")
    model.load_adapter(model_name)
    
    # model.to(device) removed to avoid Meta Tensor failure.
    # device_map='auto' already handled placement.
    model.eval()
    
    processor = ColPaliProcessor.from_pretrained(model_name)
    doc_retriever = DocumentRetriever(encoder=model, processor=processor, device=device)
    
    # Intensive cache clearing
    torch.cuda.empty_cache()
    import gc; gc.collect()
    
    samples_file = f"../dataset/samples_{args.dataset}.json"
    samples = json.load(open(samples_file, 'r'))
    print(f"Loaded {len(samples)} samples from {samples_file}")
    
    # Filter for first 5 docs
    emb_files = os.listdir(f"{args.emb_root}/{args.dataset}")
    subset_docs = [f.replace('.pt', '') for f in emb_files if f.endswith('.pt')]
    print(f"Found {len(subset_docs)} embedding file(s) in {args.emb_root}/{args.dataset}: {subset_docs}")
    
    filtered_samples = []
    for s in samples:
        clean_id = s['doc_id'].replace('.pdf', '')
        if clean_id in subset_docs:
            filtered_samples.append(s)
    
    samples = filtered_samples
    print(f"Samples remaining after filtering for subset: {len(samples)}")
    if len(samples) == 0:
        print("Warning: 0 samples matched the subset_docs ID. Checking first sample ID format...")
        if len(samples) > 0:
            print(f"Example target ID: {samples[0]['doc_id'].replace('.pdf', '')}")

    retrieve_file = f"../dataset/retrieved/samples_{args.dataset}_base_local.json"
    os.makedirs(os.path.dirname(retrieve_file), exist_ok=True)

    for sample in tqdm(samples, desc="Multi-modal Retrieval"):
        query = sample["question"]
        doc_id = sample["doc_id"].replace(".pdf", "")
        ranked_pages, page_scores = doc_retriever.base_retrieve(query, doc2emb[doc_id], top_k=args.top_k)
        
        sample["pages_ranking"] = str(ranked_pages)
        sample["pages_scores"] = str(page_scores)

    json.dump(samples, open(retrieve_file, 'w'), indent=4)
    print(f"Results saved to {retrieve_file}")
