import os
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# --- Configuration ---
# Configuration
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Expect training data in the same directory
DATA_FILE = os.path.join(SCRIPT_DIR, "training_data_qwen.jsonl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

# MacBook/MPS Setup
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def train():
    print(f"Loading Base Model: {MODEL_ID} for LoRA Fine-Tuning...")
    
    # On Mac, we use bfloat16 directly as bitsandbytes isn't fully supported for training on MPS yet
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # LoRA Config (Official Rank 8)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load and Preprocess Dataset
    # Format for Qwen-VL Chat training is complex; this is a simplified version
    # mapping (Question, Target_Score) -> Model Response (Single Digit)
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    def preprocess_function(examples):
        # This would normally prepare the multi-modal Conversation format
        # For simplicity, we are showing the structure. 
        # Fine-tuning VLMs requires proper vision-text collation.
        texts = [f"Relevance: {s}" for s in examples["target_score"]]
        return processor(text=texts, padding="max_length", truncation=True, max_length=256)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Training Arguments (Aligned with MoLoRAG+ Specs - Simplified for Demo)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_steps=5, # Just 5 steps for demo
        logging_steps=1,
        save_strategy="no",
        fp16=False,
        bf16=True,
        use_mps_device=True if DEVICE == "mps" else False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(processor.tokenizer, mlm=False),
    )

    print("Starting Training...")
    trainer.train()
    
    # Save the adapter
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    print(f"Training Complete! Adapter saved to {OUTPUT_DIR}/final_adapter")

if __name__ == "__main__":
    train()
