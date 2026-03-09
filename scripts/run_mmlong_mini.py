import os
import json
import numpy as np

def calculate_retrieval_metrics(gt_pages, retrieved_pages, k_list=[1, 3, 5]):
    """
    Calculates Recall, Precision, NDCG, and MRR at various K.
    gt_pages: list of ground truth page numbers
    retrieved_pages: list of retrieved page numbers in order of rank
    """
    results = {}
    
    if not gt_pages: # Handling "Not answerable" cases (empty GT)
        # For non-answerable, we typically exclude them or define metrics specifically.
        # MoLoRAG paper usually excludes them from retrieval metrics.
        return None

    for k in k_list:
        top_k = retrieved_pages[:k]
        hits = [p for p in top_k if p in gt_pages]
        
        # Recall@K
        recall = len(hits) / len(gt_pages)
        
        # Precision@K
        precision = len(hits) / k
        
        # MRR@K
        mrr = 0.0
        for i, p in enumerate(top_k):
            if p in gt_pages:
                mrr = 1.0 / (i + 1)
                break
        
        # NDCG@K
        dcg = 0.0
        for i, p in enumerate(top_k):
            if p in gt_pages:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = 0.0
        for i in range(min(len(gt_pages), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        results[k] = {
            "Recall": recall,
            "Precision": precision,
            "NDCG": ndcg,
            "MRR": mrr
        }
    
    return results

def run_mini_experiment_retrieval():
    results_path = "./MoLoRAG/results/MMLong/QwenVL-7B/mock_results.json"
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, 'r') as f:
        samples = json.load(f)

    # Accumulators for metrics
    methods = ["TextRAG", "M3DocRAG"]
    k_list = [1, 3, 5]
    metrics_summary = {m: {k: {"Recall": [], "Precision": [], "NDCG": [], "MRR": []} for k in k_list} for m in methods}

    for sample in samples:
        gt = eval(sample["evidence_pages"]) if isinstance(sample["evidence_pages"], str) else sample["evidence_pages"]
        
        if not gt: continue # Skip unanswerable for retrieval metrics

        # TextRAG metrics
        text_metrics = calculate_retrieval_metrics(gt, sample["text_retrieved"], k_list)
        if text_metrics:
            for k in k_list:
                for metric in ["Recall", "Precision", "NDCG", "MRR"]:
                    metrics_summary["TextRAG"][k][metric].append(text_metrics[k][metric])

        # M3DocRAG metrics
        vlm_metrics = calculate_retrieval_metrics(gt, sample["vlm_retrieved"], k_list)
        if vlm_metrics:
            for k in k_list:
                for metric in ["Recall", "Precision", "NDCG", "MRR"]:
                    metrics_summary["M3DocRAG"][k][metric].append(vlm_metrics[k][metric])

    # Calculate means
    final_table = []
    for k in k_list:
        for m in methods:
            row = {"Top-K": k, "Method": m}
            for metric in ["Recall", "Precision", "NDCG", "MRR"]:
                val = np.mean(metrics_summary[m][k][metric]) * 100
                row[metric] = f"{val:.2f}"
            final_table.append(row)

    # Printing the results in the requested format
    print("\n" + "="*85)
    print("Table 3: Retrieval performance comparison (in %) - MMLongBench (Subset)")
    print("="*85)
    print(f"{'Top-K':<6} | {'Method':<10} | {'Recall':<10} | {'Precision':<10} | {'NDCG':<10} | {'MRR':<10}")
    print("-" * 85)
    
    for row in final_table:
        print(f"{row['Top-K']:<6} | {row['Method']:<10} | {row['Recall']:<10} | {row['Precision']:<10} | {row['NDCG']:<10} | {row['MRR']:<10}")
    
    print("="*85)
    print("\nOfficial Paper Results (for reference):")
    print("Top-1: M3DocRAG (Recall: 43.31, NDCG: 56.67) | TextRAG (Recall: 29.30, NDCG: 38.99)")
    print("Top-3: M3DocRAG (Recall: 64.17, NDCG: 54.13) | TextRAG (Recall: 43.21, NDCG: 37.13)")
    print("Top-5: M3DocRAG (Recall: 72.00, NDCG: 54.06) | TextRAG (Recall: 50.60, NDCG: 37.19)")

if __name__ == "__main__":
    run_mini_experiment_retrieval()
