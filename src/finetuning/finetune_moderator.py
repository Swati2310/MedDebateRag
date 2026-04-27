"""
Fine-tune Llama 3.1 8B as medical debate moderator using QLoRA.

Run on Google Colab A100 GPU (~2-3 hours).

Usage:
    python -m src.finetuning.finetune_moderator
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "models/moderator-qlora"
FINAL_DIR  = "models/moderator-qlora-final"
DATA_FILE  = "data/moderator_training_data.json"


def main():
    # ── 1. 4-bit quantization config (QLoRA) ─────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── 2. Load base model ────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # ── 3. LoRA config ────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 4. Load + format training data ───────────────────────────────────
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    def format_example(example):
        return {
            "text": (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n{example['output']}"
            )
        }

    dataset = dataset.map(format_example)

    # ── 5. Training config ────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="wandb",
        run_name="medebate-moderator-qlora",
    )

    # ── 6. Train ──────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=2048,
    )

    trainer.train()
    trainer.save_model(FINAL_DIR)
    print(f"Fine-tuning complete! Model saved to {FINAL_DIR}")


if __name__ == "__main__":
    main()
