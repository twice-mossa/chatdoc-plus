# rag/app.py
import os
import json
from datetime import datetime
import gradio as gr
from core import build_or_update_index, retrieve, build_context_and_citations, has_index
from llm_qwen import generate_with_cot, stream_generate_with_cot
from config import MAX_HISTORY_TURNS, MAX_NEW_TOKENS, MAX_CONTEXT_TOKENS, DEDUP_SIM_THRESHOLD, REWRITE_ENABLED, ALLOW_NO_RETRIEVAL_ANSWER, GRAPH_POLICY
from packing import pack_documents
from query import is_small_talk, is_simple_math, rewrite_with_history
from graph import run_graph, GraphOutput
from telemetry import span, write_telemetry

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


def build_indexes_ui(files):
    if not files:
        return "未选择文件", []
    try:
        file_list = files if isinstance(files, list) else [files]
        for f in file_list:
            build_or_update_index(f.name)
        return "✔️ 索引已更新", file_list
    except Exception as e:
        return f"❌ 索引失败：{e}", []

def chat_ask(message, history):
    # history: list[tuple[str,str]] from Chatbot
    # 旁路：小聊/简单计算不检索，直接生成
    if ALLOW_NO_RETRIEVAL_ANSWER and (is_small_talk(message) or is_simple_math(message)):
        history = history or []
        history_tail = history[-MAX_HISTORY_TURNS:]
        history_str = "\n\n".join([f"用户：{q}\n助手：{a}" for q, a in history_tail]) if history_tail else ""
        prompt = (
            f"{SYS_PLAIN}\n\n" +
            (f"【历史对话】\n{history_str}\n\n" if history_str else "") +
            f"【问题】{message}\n【回答】"
        )
        try:
            partial = ""
            for chunk in stream_generate_with_cot(prompt, max_new_tokens=MAX_NEW_TOKENS):
                partial = chunk
                yield (history + [(message, partial)], "", "", (message, partial), "")
            return
        except Exception as e:
            err = f"❌ 推理出错：{e}"
            yield ((history or []) + [(message, err)], "", "", (message, err), "")
            return

    # 正常：可选重写后检索（若配置 graph，则走最小状态机）
    if GRAPH_POLICY == "graph":
        for out in run_graph(message, history or []):
            if isinstance(out, GraphOutput):
                # 流式中间态
                cites_md = out.cites_md
                if cites_md:
                    cites_md = f"{cites_md}\n\n引用条数：{out.cite_num}｜平均置信度：{out.avg_score:.2f}"
                yield ((history or []) + [(message, out.answer)], cites_md, "", (message, out.answer), cites_md)
        return

    rewritten = rewrite_with_history(message, history or []) if REWRITE_ENABLED else message
    trace = {"query": message}
    with span("retrieve1", trace):
        docs = retrieve(rewritten, k=4)
    history = history or []
    history_tail = history[-MAX_HISTORY_TURNS:]
    history_str = "\n\n".join([f"用户：{q}\n助手：{a}" for q, a in history_tail]) if history_tail else ""

    if docs:
        with span("pack", trace):
            packed_docs, avg_score, cite_num = pack_documents(
                message, docs, max_context_tokens=MAX_CONTEXT_TOKENS, dedup_sim_threshold=DEDUP_SIM_THRESHOLD
            )
            context, cites_md = build_context_and_citations(packed_docs)
            if cites_md:
                cites_md = f"{cites_md}\n\n引用条数：{cite_num}｜平均置信度：{avg_score:.2f}"
        prompt = (
            f"{SYS_RAG}\n\n" +
            (f"【历史对话】\n{history_str}\n\n" if history_str else "") +
            f"【检索片段】\n{context}\n\n【问题】{message}\n【回答】"
        )
    else:
        cites_md = ""
        prompt = (
            f"{SYS_PLAIN}\n\n" +
            (f"【历史对话】\n{history_str}\n\n" if history_str else "") +
            f"【问题】{message}\n【回答】"
        )
    try:
        partial = ""
        with span("generate", trace):
            for chunk in stream_generate_with_cot(prompt, max_new_tokens=MAX_NEW_TOKENS):
                partial = chunk
                yield (history + [(message, partial)], cites_md, "", (message, partial), cites_md)
        answer = partial
    except Exception as e:
        answer = f"❌ 推理出错：{e}"
        cites_md = ""
        yield ((history or []) + [(message, answer)], cites_md, "", (message, answer), cites_md)
    trace["cite_num"] = cite_num if docs else 0
    trace["avg_score"] = float(avg_score) if docs else 0.0
    write_telemetry(trace)

avatar_user = "rag/assets/user.png"
avatar_bot = "rag/assets/bot.png"
avatars = (avatar_user if os.path.exists(avatar_user) else None, avatar_bot if os.path.exists(avatar_bot) else None)

custom_css = """
.topbar{display:flex;gap:8px;align-items:center}
.attach-btn button,.link-btn button{min-width:32px;width:36px;height:36px}
.send-btn button{min-width:32px;width:40px;height:40px}
.clear-btn button{min-width:40px;width:56px;height:40px}
.msgbox textarea{min-height:44px;font-size:15px}
.composer{position:sticky;bottom:0;background:transparent;padding:6px 0}
.gr-chatbot{background:#0e0f13}
.gr-chatbot .message.user{justify-content:flex-end}
.gr-chatbot .message{display:flex}
.gr-chatbot .message .message-content{max-width:78%;border-radius:16px;padding:12px 14px}
.gr-chatbot .message.user .message-content{background:#2563eb;color:#fff}
.gr-chatbot .message.bot .message-content{background:#0b1221;border:1px solid #1f2937}
.cites{font-size:13px;opacity:0.9}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("### ChatDoc-Plus · 金融文档问答（RAG + Qwen）")

    with gr.Row(elem_classes=["topbar"]):
        status = gr.Textbox(label="索引状态", interactive=False)
    files_box = gr.Files(label="已上传", interactive=False, height=60)

    chatbot = gr.Chatbot(label=None, height=560, avatar_images=avatars, show_copy_button=True)
    with gr.Accordion("证据与引用", open=False):
        cites = gr.Markdown(elem_classes=["cites"])
        with gr.Row():
            thumb_up = gr.Button("👍", scale=0)
            thumb_down = gr.Button("👎", scale=0)
            fix_btn = gr.Button("纠错", scale=0)
    last_pair = gr.State(("", ""))  # (q,a)
    last_cites = gr.State("")

    with gr.Row(elem_classes=["composer"]):
        attach2 = gr.UploadButton("📎", file_types=[".pdf"], file_count="multiple", elem_classes=["link-btn"], scale=0)
        msg = gr.Textbox(placeholder="在这里输入，按 Enter 发送…", lines=2, show_label=False, elem_classes=["msgbox"])
        send = gr.Button("➤", variant="primary", elem_classes=["send-btn"], scale=0)
        clear = gr.Button("🗑", variant="secondary", elem_classes=["clear-btn"], scale=0)

    attach2.upload(build_indexes_ui, inputs=attach2, outputs=[status, files_box])

    def _clear_chat():
        return [], ""

    send.click(chat_ask, inputs=[msg, chatbot], outputs=[chatbot, cites, msg, last_pair, last_cites])
    msg.submit(chat_ask, inputs=[msg, chatbot], outputs=[chatbot, cites, msg, last_pair, last_cites])
    clear.click(_clear_chat, outputs=[chatbot, cites])

    def _log_feedback(polarity: int, pair: tuple[str, str], cites_text: str):
        try:
            os.makedirs("logs", exist_ok=True)
            rec = {
                "time": datetime.utcnow().isoformat(),
                "question": pair[0],
                "answer": pair[1],
                "citations": cites_text,
                "feedback": polarity,  # 1=up, -1=down, 0=fix
            }
            with open("logs/feedback.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return "已记录反馈"
        except Exception as e:
            return f"记录失败：{e}"

    fb_status = gr.Markdown()
    thumb_up.click(lambda p, c: _log_feedback(1, p, c), inputs=[last_pair, last_cites], outputs=fb_status)
    thumb_down.click(lambda p, c: _log_feedback(-1, p, c), inputs=[last_pair, last_cites], outputs=fb_status)
    def _fix(pair, cites_text):
        return _log_feedback(0, pair, cites_text)
    fix_btn.click(_fix, inputs=[last_pair, last_cites], outputs=fb_status)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
