from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import PaliGemmaForConditionalGeneration, BitsAndBytesConfig
import os 
from pdf2image import convert_from_path
import argparse
import torch
from tqdm import tqdm
import sys 
sys.path.append("../")
from utils import prepare_files

def encode_document(doc_path, doc_id, batch_size=1, resolution=100, save_emb=True, save_img=True):
    # Reduced resolution to 100 DPI for safe T4 execution
    try:
        page_images = convert_from_path(doc_path, dpi=resolution)
    except Exception as e:
        print(f"Error converting {doc_path}: {e}")
        return
    
    if save_img: 
        os.makedirs(img_save_dir, exist_ok=True)
        for page_num, page_snapshot in enumerate(page_images):
            img_path = f"{img_save_dir}/{doc_id}-{page_num+1}.png"
            if not os.path.exists(img_path):
                page_snapshot.save(img_path) 

    if save_emb:
        total_image_embeds = torch.Tensor().to(device)
        for idx in range(0, len(page_images), batch_size):
            batch_images = page_images[idx: idx+batch_size]
            batch_images = processor.process_images(batch_images).to(device)
            with torch.no_grad():
                image_embeds = model(**batch_images)
            # Ensure we only store the embeddings on CPU to save VRAM
            total_image_embeds = torch.cat((total_image_embeds, image_embeds.to(device)), dim=0)
            
            torch.cuda.empty_cache()
            import gc; gc.collect()
        
        torch.save(total_image_embeds.cpu(), f"{save_dir}/{doc_id}.pt")
        print(f"Saved Embeddings {total_image_embeds.shape} to {save_dir}/{doc_id}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MMLong")
    parser.add_argument("--save_dir", type=str, default="../tmp/tmp_embs")
    parser.add_argument("--img_save_dir", type=str, default="../tmp/tmp_imgs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="vidore/colpali-v1.2")
    args = parser.parse_args()
    
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ColPali via PaliGemma base on {device}...")
    
    # Switch to float16 to avoid bitsandbytes AssertionError on T4 GPUs
    # Use device_map='auto' to let accelerate handle placement.
    # Manual .to(device) is REMOVED to avoid NotImplementedError (Meta Tensor).
    model = ColPali.from_pretrained(
        "vidore/colpaligemma-3b-pt-448-base",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load the adapter into the ColPali instance
    print(f"Loading adapter: {args.model_name}")
    model.load_adapter(args.model_name)
    
    # model.to(device) removed to avoid Meta Tensor failure.
    # device_map='auto' already handled placement.
    model.eval()
    
    processor = ColPaliProcessor.from_pretrained(args.model_name)
    

    documents = prepare_files(f"../dataset/{args.dataset}", suffix=".pdf")
    print(f"Found {len(documents)} PDF(s) in ../dataset/{args.dataset}")
    
    documents = documents[:5] # Subset logic
    print(f"Encoding subset: {documents}")
    
    save_dir, img_save_dir = f"{args.save_dir}/{args.dataset}", f"{args.img_save_dir}/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)

    for doc_path in tqdm(documents, desc="Multi-modal Encoding (Subset)"):
        doc_id = doc_path.replace(".pdf", "") 
        print(f"Processing {doc_id}...")
        encode_document(doc_path=f"../dataset/{args.dataset}/{doc_path}", doc_id=doc_id, save_img=True)
