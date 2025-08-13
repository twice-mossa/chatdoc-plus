import os
from typing import Dict
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def format_example(ex: Dict) -> Dict:
    prompt = ex["prompt"]
    response = ex["response"]
    ex["text"] = f"<|system|>你是一个有帮助的中文助手。<|end|>\n<|user|>{prompt}<|end|>\n<|assistant|>{response}<|end|>"
    return ex


def main():
    model_dir = os.getenv("CHATDOC_MODEL_DIR", "/root/chatdoc-plus/models/hf/Qwen1.5-7B-Chat/qwen/Qwen1.5-7B-Chat")
    data_path = os.getenv("CHATDOC_SFT_DATA", "/root/chatdoc-plus/data/sft.jsonl")
    output_dir = os.getenv("CHATDOC_SFT_OUT", "/root/chatdoc-plus/models/sft-qwen")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    base = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    base = prepare_model_for_kbit_training(base)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)

    ds = load_dataset("json", data_files=data_path, split="train")
    ds = ds.map(format_example)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            num_train_epochs=1,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_steps=200,
            bf16=torch.cuda.is_available(),
        ),
        max_seq_length=2048,
        packing=True,
        dataset_text_field="text",
    )

    trainer.train()
    # 保存可直接被推理端加载的 adapter
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()


