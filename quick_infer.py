# quick_infer.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_DIR = "/root/chatdoc-plus/models/hf/Qwen1.5-7B-Chat/qwen/Qwen1.5-7B-Chat"  # ← 改成你的真实目录

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

# 一句中文测试：你可以替换成任何问题
messages = [
    {"role": "system", "content": "你是一个谨慎的中文金融顾问，回答尽量简洁并给出要点。"},
    {"role": "user", "content": "基金定投需要注意哪些核心风险？"},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
    )

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# 只打印最后一轮助手回答（简单切分）
if "assistant" in text:
    print(text.split("assistant")[-1].strip())
else:
    print(text)
