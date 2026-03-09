import os
import json
import torch
import random
import re
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import fitz  # PyMuPDF

# --- Configuration ---
# TEACHER_MODEL: Using 3B version for MacBook memory compatibility
TEACHER_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct" 
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Local Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
# The script expects a 'dataset' folder in the current directory or the parent
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
if not os.path.exists(DATASET_DIR):
    DATASET_DIR = os.path.join(BASE_PATH, "dataset")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "training_data_qwen.jsonl")

# Datasets to sample from
DATASETS = [
    ("MMLongBench", os.path.join(DATASET_DIR, "MMLong")),
    ("LongDocURL", os.path.join(DATASET_DIR, "LongDocURL")),
]

def load_teacher():
    print(f"Loading Teacher Model: {TEACHER_MODEL_ID} on {DEVICE}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        TEACHER_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE
    ).eval()
    processor = AutoProcessor.from_pretrained(TEACHER_MODEL_ID)
    return model, processor

def generate_question(model, processor, image, target_score):
    """Generate a question that matches the target relevance score."""
    score_prompts = {
        5: "Generate a question that can be answered DIRECTLY and completely by looking at this image.",
        4: "Generate a question where most of the answer is in this image, but maybe one minor detail is missing.",
        3: "Generate a question that is MODERATELY related to this image but requires some outside inference.",
        2: "Generate a question that is SLIGHTLY related to the topic of this image but NOT answerable from it.",
        1: "Generate a question that is completely IRRELEVANT to the content of this image."
    }
    
    prompt = f"""# TASK #
{score_prompts[target_score]}
# INSTRUCTION #
Output ONLY the question text. Do not include any prefixes or explanations."""

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=100)
        question = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    
    return question

def verify_score(model, processor, question, image):
    """Automated Quality Control: Teacher predicts score for the generated question."""
    prompt = f"""# TASK #
Evaluate how relevant the image is to this query: "{question}"
Rate 1-5. Output ONLY the number."""

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=5)
        out = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    score_match = re.search(r'[1-5]', out)
    return int(score_match.group(0)) if score_match else 3

def main():
    model, processor = load_teacher()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    dataset_paths = []
    for _, path in DATASETS:
        if os.path.exists(path):
            pdfs = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pdf")]
            dataset_paths.extend(pdfs)

    print(f"Sampling from {len(dataset_paths)} PDFs...")
    
    generated_count = 0
    target_total = 10 # Adjust as needed for full training set (paper uses 3,519)

    with open(OUTPUT_FILE, "a") as f:
        while generated_count < target_total:
            pdf_path = random.choice(dataset_paths)
            try:
                doc = fitz.open(pdf_path)
                page_num = random.randint(0, len(doc)-1)
                page = doc[page_num]
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                doc.close()
                
                target_score = random.randint(1, 5)
                question = generate_question(model, processor, img, target_score)
                
                # Automated QC: |s - s'| <= 1
                predicted_score = verify_score(model, processor, question, img)
                if abs(target_score - predicted_score) <= 1:
                    data_point = {
                        "question": question,
                        "image_path": f"{os.path.basename(pdf_path)}_p{page_num}", # Placeholder, you'd save the image or ref
                        "target_score": target_score,
                        "verified_score": predicted_score
                    }
                    f.write(json.dumps(data_point) + "\n")
                    generated_count += 1
                    print(f"[{generated_count}/{target_total}] Generated Q with score {target_score} (Verified: {predicted_score})")
                else:
                    print(f"Sample rejected: Target {target_score}, Predicted {predicted_score}")
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    main()
