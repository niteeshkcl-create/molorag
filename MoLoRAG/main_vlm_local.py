import os
import json
import argparse
import torch 
from tqdm import tqdm
import time 

def main_vlm_local_QA(args):
    st_time = time.time()
    
    from VLMModels.Qwen_VL_local import init_model, get_response_concat
    
    input_path = f"./dataset/retrieved/samples_{args.dataset}_base_local.json"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run retrieval first.")
        return

    samples = json.load(open(input_path, "r"))
    
    print(f"Loading local VLM: {args.model_name}")
    model = init_model(args.model_name, device="auto")

    for sample in tqdm(samples, desc="VLM QA"):
        doc_id = sample['doc_id'].replace('.pdf', '')
        ranked_pages = eval(sample["pages_ranking"])[:args.topk]
        
        # Paths to page images
        input_image_list = [f"./tmp/tmp_imgs/{args.dataset}/{doc_id}-{p}.png" for p in ranked_pages]
        
        try:
            query_prompt = f"Based on the images from the document, please answer the question: {sample['question']}"
            response = get_response_concat(model, query_prompt, input_image_list, max_new_tokens=128)
        except Exception as e:
            print(f"[ERROR] VLM prediction: {e}")
            response = "None"

        sample[args.response_key] = response 
     
    output_path = f"./results/{args.dataset}/VLM/qwen2.5_vl_base_local.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as file:
        json.dump(samples, file, indent=4)
        
    print(f"Completed in {(time.time() - st_time)/60:.2f} Mins")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MMLong")
    parser.add_argument("--model_name", type=str, default="QwenVL-3B")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--response_key", type=str, default="raw_response")
    args = parser.parse_args()
    main_vlm_local_QA(args)
