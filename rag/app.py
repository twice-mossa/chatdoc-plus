# ~/chatdoc-plus/rag/app.py
import gradio as gr
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI  # 先用占位；换成本地模型见下方注释
from pathlib import Path
from .core import build_or_load_db

INDEX_DIR = "../index/faiss_index"

def build_chain(pdf_dir="../data/pdfs"):
    pdf_paths = [str(p) for p in Path(pdf_dir).glob("*.pdf")]
    db = build_or_load_db(pdf_paths, INDEX_DIR)

    # —— 先用 OpenAI 兼容，等你把本地 LLM（如 vLLM/llama-cpp）接上再替换 ——
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True
    )
    return chain

chain = None

def build_index(_):
    global chain
    chain = build_chain()
    return "✅ 索引构建完成，可以开始问答。"

def ask(q):
    if chain is None:
        return "请先点击【构建/加载索引】", None
    res = chain({"question": q})
    ans = res["answer"]
    srcs = res.get("source_documents", [])
    refs = "\n".join([f"- {s.metadata.get('source','')}: p.{s.metadata.get('page', '?')}"
                      for s in srcs])
    return ans, refs or "（无检索引用）"

with gr.Blocks() as demo:
    gr.Markdown("### ChatDoc-Plus | 金融 PDF RAG 多轮问答（MVP）")
    with gr.Row():
        pdf_btn = gr.Button("构建/加载索引")
    with gr.Row():
        q = gr.Textbox(label="你的问题（支持连续追问）")
    with gr.Row():
        a = gr.Textbox(label="回答", lines=8)
    with gr.Row():
        refs = gr.Textbox(label="引用片段", lines=6)
    pdf_btn.click(build_index, outputs=a)
    q.submit(ask, inputs=q, outputs=[a, refs])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
