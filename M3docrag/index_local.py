from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import PaliGemmaForConditionalGeneration, BitsAndBytesConfig
import os 
from pdf2image import convert_from_path
import argparse
import torch
from tqdm import tqdm
import sys 

# Add local path support
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))

def encode_document(doc_path, doc_id, model, processor, device, save_dir, img_save_dir, batch_size=1, resolution=100, save_emb=True, save_img=True):
    """Converts a PDF document into visual embeddings and snapshots."""
    try:
        page_images = convert_from_path(doc_path, dpi=resolution)
    except Exception as e:
        print(f"Error converting {doc_path}: {e}")
        return
    
    if save_img: 
        os.makedirs(img_save_dir, exist_ok=True)
        for page_num, page_snapshot in enumerate(page_images):
            img_path = os.path.join(img_save_dir, f"{doc_id}-{page_num+1}.png")
            if not os.path.exists(img_path):
                page_snapshot.save(img_path) 

    if save_emb:
        total_image_embeds = torch.Tensor().to(device)
        for idx in range(0, len(page_images), batch_size):
            batch_images = page_images[idx: idx+batch_size]
            batch_images = processor.process_images(batch_images).to(device)
            with torch.no_grad():
                image_embeds = model(**batch_images)
            total_image_embeds = torch.cat((total_image_embeds, image_embeds.to(device)), dim=0)
            
            torch.cuda.empty_cache()
        
        torch.save(total_image_embeds.cpu(), os.path.join(save_dir, f"{doc_id}.pt"))
        print(f"Saved Embeddings {total_image_embeds.shape} to {save_dir}/{doc_id}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MMLong")
    parser.add_argument("--save_dir", type=str, default="tmp/tmp_embs")
    parser.add_argument("--img_save_dir", type=str, default="tmp/tmp_imgs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="vidore/colpali-v1.2")
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading ColPali on {device}...")
    
    model = ColPali.from_pretrained(
        "vidore/colpaligemma-3b-pt-448-base",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.load_adapter(args.model_name)
    model.eval()
    
    processor = ColPaliProcessor.from_pretrained(args.model_name)
    
    # Path logic
    dataset_root = os.path.join(SCRIPT_DIR, "..", "dataset", args.dataset)
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset not found at {dataset_root}")
        sys.exit(1)

    pdfs = [f for f in os.listdir(dataset_root) if f.endswith(".pdf")]
    print(f"Found {len(pdfs)} PDF(s) in {dataset_root}")
    
    save_dir = os.path.join(SCRIPT_DIR, args.save_dir, args.dataset)
    img_save_dir = os.path.join(SCRIPT_DIR, args.img_save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    for pdf in tqdm(pdfs, desc="Indexing Documents"):
        doc_id = pdf.replace(".pdf", "") 
        encode_document(
            doc_path=os.path.join(dataset_root, pdf), 
            doc_id=doc_id, 
            model=model, 
            processor=processor, 
            device=device,
            save_dir=save_dir,
            img_save_dir=img_save_dir
        )
