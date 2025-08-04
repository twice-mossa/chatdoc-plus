# rag/app.py
import gradio as gr
from core import build_or_update_index, retrieve, has_index
from llm_qwen import generate_with_cot

# —— 系统提示词（有/无检索两种）——
SYS_RAG = (
    "你是严谨的中文金融助手，必须依据提供的【检索片段】回答；"
    "若片段中没有相关信息，明确说明“根据材料无法确定”。"
    "回答要点化，并在末尾用 [1][2]… 标注引用片段编号。"
)
SYS_PLAIN = (
    "你是中文金融助手。当前没有检索片段，请基于常识给出简洁、要点化回答，"
    "并提示内容可能不够精准。"
)

def build_index_ui(file):
    build_or_update_index(file.name)
    return "✔️ 索引已更新"

def ask(query):
    docs = retrieve(query, k=4)          # 没索引时返回 []
    if docs:
        context = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
        prompt = f"{SYS_RAG}\n\n【检索片段】\n{context}\n\n【问题】{query}\n【回答】"
    else:
        prompt = f"{SYS_PLAIN}\n\n【问题】{query}\n【回答】"
    try:
        return generate_with_cot(prompt, max_new_tokens=300)
    except Exception as e:
        return f"❌ 推理出错：{e}"

with gr.Blocks() as demo:
    gr.Markdown("### ChatDoc-Plus · 金融文档问答（RAG + Qwen）")
    with gr.Row():
        up = gr.File(label="上传PDF")
        status = gr.Textbox(label="索引状态", interactive=False)
    up.upload(build_index_ui, inputs=up, outputs=status)

    q = gr.Textbox(label="问题", placeholder="例：介绍金融知识普及读本")
    out = gr.Textbox(label="回答", lines=12)
    btn = gr.Button("提问")
    btn.click(ask, inputs=q, outputs=out)
    q.submit(ask, inputs=q, outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
