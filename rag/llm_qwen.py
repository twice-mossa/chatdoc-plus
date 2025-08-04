# rag/llm_qwen.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_DIR = "/root/chatdoc-plus/models/hf/Qwen1.5-7B-Chat/qwen/Qwen1.5-7B-Chat"  # 改成你的真实路径

_tok = None
_llm = None

def get_llm():
    global _tok, _llm
    if _llm is None:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        _tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)
        _llm = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto",
                                                    torch_dtype=torch.float16, quantization_config=bnb,
                                                    trust_remote_code=True)
    return _tok, _llm

def generate_with_cot(prompt, max_new_tokens=300):
    tok, llm = get_llm()
    inputs = tok([prompt], return_tensors="pt").to(llm.device)
    with torch.inference_mode():
        out = llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    return tok.decode(out[0], skip_special_tokens=True)
