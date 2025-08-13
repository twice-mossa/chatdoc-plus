ChatDoc-Plus
============

功能概览
--------
- 上传 PDF → 构建/更新 FAISS 向量索引（`bge-small-zh` 本地嵌入）
- RAG 检索 + 本地 Qwen-7B-Chat（4bit）回答
- 多轮对话（可控历史轮数）
- 证据溯源（文件名+页码）与可选重排序器（bge-reranker）
- 简单检索评测（Hit@k、MRR、延迟）
- SFT 与 DPO 的最小可运行训练脚本（QLoRA/PEFT）

快速开始
--------
1) 准备环境
```
pip install -r requirements.txt
```

2) 下载模型
- Qwen（示例）: 运行 `models/download_ms.py` 或手动放到 `models/hf/Qwen1.5-7B-Chat/qwen/Qwen1.5-7B-Chat`
- 嵌入：将 `bge-small-zh-v1.5` 放到 `models/embeddings/bge-small-zh/AI-ModelScope/bge-small-zh-v1.5`

3) 环境变量（可选）
- `CHATDOC_MODEL_DIR`：LLM 路径
- `CHATDOC_EMBEDDING_DIR`：嵌入模型路径
- `CHATDOC_INDEX_DIR`：索引目录（默认 `index/faiss_index`）
- `CHATDOC_USE_RERANKER`：是否使用重排（"1"/"0"）
- `CHATDOC_MAX_HISTORY_TURNS`：对话历史轮数

4) 启动 Demo
```
python rag/app.py
```
访问 `http://127.0.0.1:7860`，上传 PDF，开始问答。

评测
----
准备一个如 `data/eval_retrieval.json` 的数据：
```
[
  {"question": "书籍简介是什么？", "references": ["demo.pdf", "demo.pdf:1"]}
]
```
运行：
```
# 单配置对比（baseline vs tuned）
python rag/eval.py data/eval_retrieval.json -k 4 --mode hybrid --mmr --rerank

# 多配置表格（导出 CSV）
python rag/eval.py data/eval_retrieval.json -k 4 --grid --out reports/metrics.csv
```

对比表使用建议：挑选“命中率最高（其次 MRR），延迟可接受”的一行作为推荐配置，并在 README/Elevator Pitch 中写一句结论，例如：

> 在金融 PDF 数据集上，使用 hybrid（BM25+向量+RRF）+MMR+CrossEncoder 相比 dense-only，Hit@4 提升 +18%，MRR 提升 +15%，平均延迟增加 0.35s（权衡可接受）。

SFT（监督微调）
--------------
数据格式（`data/sft.jsonl`）：每行 JSON
```
{"prompt": "……", "response": "……"}
```
训练（LoRA + TRL SFTTrainer）：
```
python sft/train_sft.py

# 推理端加载 LoRA 适配器
export CHATDOC_ADAPTER_DIR="/root/chatdoc-plus/models/sft-qwen"
python rag/app.py
```

DPO（偏好优化）
--------------
数据格式（`data/dpo.jsonl`）：每行 JSON
```
{"prompt": "……", "chosen": "……", "rejected": "……"}
```
训练（LoRA + TRL DPOTrainer）：
```
python dpo/train_dpo.py

# 推理端加载 LoRA 适配器
export CHATDOC_ADAPTER_DIR="/root/chatdoc-plus/models/dpo-qwen"
python rag/app.py
```

GRPO（强化式偏好优化）
--------------------
数据格式（`data/grpo.jsonl`）：每行 JSON
```
{"prompt": "……"}  # prompt 内部可包含你拼接的检索片段模板
```
训练（LoRA + TRL GRPOTrainer，使用内置奖励：引用存在 + 简洁度）：
```
python grpo/train_grpo.py

# 推理端加载 LoRA 适配器
export CHATDOC_ADAPTER_DIR="/root/chatdoc-plus/models/grpo-qwen"
python rag/app.py
```
对比建议：
- DPO：稳定、偏好对齐明确，需要“同问优/劣答案”配对
- GRPO：无需成对答案，能针对“是否有引用/是否简洁”等行为目标优化
可在相同验证集上比较“引用率、Hit@k（RAG）、平均长度、人工偏好”。

目录结构
--------
```
chatdoc-plus/
├── rag/
│   ├── app.py             # Gradio 对话 UI（ChatGPT 样式/流式/反馈）
│   ├── core.py            # 加载/切分/嵌入/索引/检索/重排
│   ├── config.py          # 配置与环境变量
│   ├── llm_qwen.py        # 本地 Qwen 4bit 推理 + LoRA 适配器加载
│   ├── hybrid.py          # BM25 + RRF 融合
│   ├── packing.py         # token 预算装配与相似度去重
│   ├── query.py           # 闲聊旁路与轻量 rewrite
│   ├── graph.py           # 最小状态机（rewrite→retrieve→pack→verify→generate）
│   ├── telemetry.py       # 轻量监控（阶段耗时/引用质量写入日志）
│   ├── eval.py            # 检索评测（单配置/网格 CSV）
│   └── assets/            # 头像等资源
├── sft/train_sft.py       # SFT（LoRA + TRL）
├── dpo/train_dpo.py       # DPO（LoRA + TRL）
├── grpo/train_grpo.py     # GRPO（LoRA + TRL）
├── data/
│   ├── sft.jsonl
│   ├── dpo.jsonl
│   ├── grpo.jsonl
│   ├── eval_retrieval.json
│   ├── feedback_to_sft.py # 反馈→SFT 数据
│   └── feedback_to_dpo.py # 反馈→DPO 数据
├── models/download_ms.py  # 模型下载（ModelScope）
├── index/                 # 本地索引（已在 .gitignore 中排除）
├── quick_infer.py         # 纯 LLM 推理样例
└── requirements.txt
```

备注
----
- 本项目默认离线运行，请确保相应模型已下载到本地路径。
- 如果没有 GPU，也可在 CPU 上运行，但推理与重排的速度会较慢。


