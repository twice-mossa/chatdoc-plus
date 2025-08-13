import os
import json
from typing import List, Dict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def reward_citation(outputs: List[str], prompts: List[str]) -> List[float]:
    # +1 if contains [1] style citations, else 0
    scores = []
    for out in outputs:
        scores.append(1.0 if "[1]" in out or "[2]" in out else 0.0)
    return scores


def reward_conciseness(outputs: List[str], prompts: List[str]) -> List[float]:
    # Encourage concise answers (50-400 chars window)
    scores = []
    for out in outputs:
        n = len(out)
        if n < 50:
            scores.append(0.2)
        elif n > 600:
            scores.append(0.3)
        else:
            scores.append(1.0)
    return scores


def format_prompt(ex: Dict) -> str:
    # Simple system prompt to encourage citation usage
    sys = "你是严谨的中文金融助手，必须依据检索片段回答，并在末尾用 [1][2]… 标注引用。"
    return f"<|system|>{sys}<|end|>\n<|user|>{ex['prompt']}<|end|>\n<|assistant|>"


def main():
    model_dir = os.getenv("CHATDOC_MODEL_DIR", "/root/chatdoc-plus/models/hf/Qwen1.5-7B-Chat/qwen/Qwen1.5-7B-Chat")
    data_path = os.getenv("CHATDOC_GRPO_DATA", "/root/chatdoc-plus/data/grpo.jsonl")
    output_dir = os.getenv("CHATDOC_GRPO_OUT", "/root/chatdoc-plus/models/grpo-qwen")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    base = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    base = prepare_model_for_kbit_training(base)

    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lora_cfg)

    # Build dataset with prompts
    raw_ds = load_dataset("json", data_files=data_path, split="train")
    ds = raw_ds.map(lambda ex: {"prompt": format_prompt(ex)})

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[reward_citation, reward_conciseness],
        args=GRPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-6,
            num_train_epochs=1,
            logging_steps=10,
            save_steps=200,
            bf16=torch.cuda.is_available(),
            max_prompt_length=512,
            max_completion_length=512,
        ),
        train_dataset=ds,
        dataset_text_field="prompt",
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()


