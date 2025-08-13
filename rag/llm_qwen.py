# rag/llm_qwen.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread
from config import MODEL_DIR, MAX_NEW_TOKENS
from peft import PeftModel

_tok = None
_llm = None


def get_llm():
    global _tok, _llm
    if _llm is None:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)
        _llm = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=bnb,
            trust_remote_code=True,
        )
        adapter_dir = os.getenv("CHATDOC_ADAPTER_DIR")
        if adapter_dir and os.path.isdir(adapter_dir):
            try:
                _llm = PeftModel.from_pretrained(_llm, adapter_dir)
                _llm.eval()
                print(f"Loaded LoRA adapter from {adapter_dir}")
            except Exception as e:
                print(f"Failed to load adapter {adapter_dir}: {e}")
    return _tok, _llm


def apply_chat_template(messages, tok):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_with_cot(prompt, max_new_tokens: int = MAX_NEW_TOKENS):
    tok, llm = get_llm()
    inputs = tok([prompt], return_tensors="pt").to(llm.device)
    with torch.inference_mode():
        out = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
        )
    return tok.decode(out[0], skip_special_tokens=True)


def stream_generate_with_cot(prompt, max_new_tokens: int = MAX_NEW_TOKENS):
    tok, llm = get_llm()
    inputs = tok([prompt], return_tensors="pt").to(llm.device)
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        streamer=streamer,
    )

    def _run():
        with torch.inference_mode():
            llm.generate(**gen_kwargs)

    thread = Thread(target=_run)
    thread.start()

    accumulated = ""
    for text in streamer:
        accumulated += text
        yield accumulated
