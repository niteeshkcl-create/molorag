import os
from langchain_community.vectorstores import FAISS
import json
import argparse
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
import torch
from transformers import pipeline
import time

def retrieve_context(query, index_folder, embeddings, top_k=5, max_length=1600):
    if not os.path.exists(index_folder):
        return "" 
    
    faiss_index = FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)
    searched_docs = faiss_index.similarity_search(query, k=top_k)
    context = []
    for doc in searched_docs:
        content = doc.page_content[:max_length].replace('\n', ' ').strip()
        context.append(' '.join(content.split()))
    return " ".join(context)

def main_llm_QA(args):
    st_time = time.time()
    
    # Load Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=args.embed_model)
    
    # Load Local LLM
    print(f"Loading local LLM: {args.llm_model}")
    pipe = pipeline("text-generation", model=args.llm_model, device_map="auto", torch_dtype=torch.bfloat16)

    input_file = f"../dataset/samples_{args.dataset}.json"
    with open(input_file, 'r') as file:
        all_samples = json.load(file)
    
    # Filter for subset (only samples matching the first 5 PDFs)
    subset_docs = [f.replace('.pdf', '') for f in os.listdir(f"../dataset/{args.dataset}") if f.endswith('.pdf')][:5]
    samples = [s for s in all_samples if s['doc_id'].replace('.pdf', '') in subset_docs]

    for sample in tqdm(samples, desc="QA on Subset"):
        index_folder = f"../tmp/tmp_dbs/{args.dataset}/{sample['doc_id'].replace('.pdf', '')}"
        context_txt = retrieve_context(query=sample["question"], index_folder=index_folder, embeddings=embeddings, top_k=args.retrieve_topk)
        
        prompt = f"Context: {context_txt}\n\nQuestion: {sample['question']}\n\nAnswer concisely based on context:"
        
        result = pipe(prompt, max_new_tokens=128, do_sample=False)[0]['generated_text']
        response = result.replace(prompt, "").strip()
        sample[args.response_key] = response
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        json.dump(samples, file, indent=4)

    print(f"Cost time: {(time.time() - st_time)/60:.2f} Mins")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MMLong")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--retrieve_topk", type=int, default=5)
    parser.add_argument("--response_key", type=str, default="raw_response")
    args = parser.parse_args()

    output_file = f"../results/{args.dataset}/LLM/local_qwen1.5b_top{args.retrieve_topk}.json"
    main_llm_QA(args)
