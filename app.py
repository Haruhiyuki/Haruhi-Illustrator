import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import pickle
import numpy as np
import gradio as gr
from BCEmbedding import EmbeddingModel, RerankerModel
from pathlib import Path
import asyncio
import traceback
import sys

import faiss
import google.generativeai as genai
from dotenv import load_dotenv

# ---------- 路径 & 参数 ----------
IMG_DIR = Path("my_images/")
FAISS_INDEX_PATH = "dataset/captions.faiss"
DOCS_PATH = "dataset/captions.docs.pkl"

# --- 模型配置 ---
LLM_MODEL = "gemini-2.0-flash"
EMBED_MODEL_NAME = "maidalun1020/bce-embedding-base_v1"
RERANK_MODEL_NAME = "maidalun1020/bce-reranker-base_v1"
VECTOR_DIM = 768

# --- 检索参数 ---
TOP_K_INITIAL = 25
TOP_K_RERANK = 9
RERANK_QUERY_MAX_LENGTH = 400

# --- 平台感知优化 ---
IS_MAC = sys.platform == "darwin"
USE_FP16 = not IS_MAC  # 在macOS上禁用FP16，在其他平台启用

# ---------- 初始化 ----------
# 1. 加载本地模型和索引
print("Loading local models and FAISS index...")
print(f"Platform: {sys.platform}. Using FP16: {USE_FP16}")
try:
    embed_model = EmbeddingModel(model_name_or_path=EMBED_MODEL_NAME, use_fp16=USE_FP16)
    rerank_model = RerankerModel(model_name_or_path=RERANK_MODEL_NAME, use_fp16=USE_FP16)
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)
    print(f"[✓] Local models and index with {len(documents)} documents loaded.")
except Exception as e:
    raise SystemExit(f"[!] Failed to load local models or FAISS index. Error: {e}")

# 2. 初始化 Gemini API
print("Configuring Gemini API...")
load_dotenv()
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"], transport="rest")
    llm = genai.GenerativeModel(LLM_MODEL)
    print("[✓] Gemini API configured.")
except Exception as e:
    raise SystemExit(f"[!] Failed to configure Gemini API. Check your GOOGLE_API_KEY. Error: {e}")


# ---------- 核心功能函数 ----------

async def extract_scenes_from_article(article_text: str):
    """
    阶段一：仅调用LLM从文章中提取适合配图的视觉化片段。
    """
    if not article_text or not article_text.strip():
        gr.Warning("请输入文章内容！")
        return gr.update(choices=[], value=None), [], ""  # 清空所有相关输出

    gr.Info("正在分析文章并提取场景，请稍候...")
    print("Extracting scenes from article...")

    system_prompt = """
    你是一个专业的文学编辑，你的任务是从给定的的小说或文章中，提取出最具有画面感、最适合用图片来表现的句子或段落。

    你要遵循以下规则：
    1.  只选择那些包含具体动作、场景描写、人物外貌或强烈情绪氛围的片段。
    2.  忽略那些纯粹的对话、心理活动或抽象的论述。
    3.  每个提取的片段不宜过长，最好是一到三句话。
    4.  选择片段的时候，片段在文章中的分布均匀一些，不要让文章中连续太多内容缺少配图，同时不要太同质化。
    5.  你的输出必须是一个严格的JSON格式的字符串列表。例如：["片段1", "片段2", "片段3"]
    """

    try:
        response = await asyncio.to_thread(
            llm.generate_content,
            [system_prompt, "请处理以下文章：\n\n" + article_text],
            generation_config={"response_mime_type": "application/json"}
        )

        scenes = json.loads(response.text)
        first_scene = scenes[0]
        print(f"  ...Extracted {len(scenes)} scenes.")

        if not scenes:
            gr.Info("未能从文章中提取出适合配图的场景。")
            return gr.update(choices=[], value=None), [], ""

        return gr.update(choices=scenes, value=first_scene), [], ""

    except Exception as e:
        gr.Error(f"文章分析失败: {e}")
        traceback.print_exc()
        return gr.update(choices=[], value=None), [], ""


async def search_and_rerank(query_text: str):
    """
    阶段二：根据单个文本片段，进行向量搜索、重排序，并返回结果。
    """
    if not query_text or not query_text.strip():
        return [], ""

    try:
        gr.Info(f"正在为片段 “{query_text[:30]}...” 检索图片...")
        print(f"Embedding query: '{query_text[:50]}...'")
        query_vector = await asyncio.to_thread(embed_model.encode, [query_text], normalize_embeddings=True)

        print(f"Searching top {TOP_K_INITIAL} candidates...")
        distances, indices = await asyncio.to_thread(faiss_index.search, query_vector, TOP_K_INITIAL)

        initial_candidates = [documents[i] for i in indices[0]]
        candidate_texts = [doc["text"] for doc in initial_candidates]

        print(f"Reranking {len(candidate_texts)} candidates...")
        truncated_query = query_text[:RERANK_QUERY_MAX_LENGTH]
        rerank_output = await asyncio.to_thread(rerank_model.rerank, truncated_query, candidate_texts)

        if isinstance(rerank_output, dict) and 'rerank_passages' in rerank_output:
            reranked_results = list(zip(rerank_output['rerank_passages'], rerank_output['rerank_scores']))
        else:
            reranked_results = []

        if not reranked_results:
            gr.Info("没有找到匹配的图片。")
            return [], ""

        final_docs_map = {doc["text"]: doc for doc in initial_candidates}
        gallery_results = []
        markdown_results = "### 搜索结果详情 (已重排序):\n\n"

        for i, (text, score) in enumerate(reranked_results[:TOP_K_RERANK]):
            doc = final_docs_map.get(text)
            if not doc: continue

            image_path = IMG_DIR / doc['path']
            if image_path.exists():
                gallery_results.append((str(image_path), f"Rerank Score: {score:.3f}"))
                markdown_results += f"**结果 {i + 1}: {os.path.basename(image_path)}** (重排分数: {score:.3f})\n> {doc['text']}\n\n---\n"
            else:
                print(f"[Warning] Image path not found: {image_path}")

        return gallery_results, markdown_results

    except Exception as e:
        gr.Error(f"搜索过程中发生错误: {e}")
        traceback.print_exc()
        return [], ""


# ---------- Gradio UI ----------
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {max-width: 95% !important;}") as app:
    gr.Markdown("# 凉宫同人小说配图器 (BY 凉宫春日应援团)")

    with gr.Tabs():
        with gr.TabItem("文章智能配图"):
            with gr.Row():
                with gr.Column(scale=2):
                    article_box = gr.Textbox(
                        label="第一步：粘贴完整文章",
                        placeholder="在此处粘贴你的小说章节或文章内容...",
                        lines=28
                    )
                    analyze_button = gr.Button("分析文章并提取场景", variant="primary")

                    scenes_radio = gr.Radio(
                        label="第二步：选择一个片段进行配图",
                        choices=[],
                        type="value"
                    )

                with gr.Column(scale=3):
                    article_gallery_output = gr.Gallery(
                        label="检索结果",
                        show_label=False,
                        columns=3,
                        object_fit="contain",
                        height="auto"
                    )
                    article_details_output = gr.Markdown("在这里查看结果详情...")

            # --- 关键修改：分析按钮现在只负责提取场景和清空结果区 ---
            analyze_button.click(
                fn=extract_scenes_from_article,
                inputs=[article_box],
                outputs=[scenes_radio, article_gallery_output, article_details_output]
            )

            scenes_radio.change(
                fn=search_and_rerank,
                inputs=[scenes_radio],
                outputs=[article_gallery_output, article_details_output]
            )

        with gr.TabItem("单片段配图"):
            with gr.Row():
                with gr.Column(scale=2):
                    snippet_box = gr.Textbox(
                        label="输入单个描述片段",
                        placeholder="例如：春日在教室里，阳光正好，她回头微笑着...",
                        lines=10
                    )
                    snippet_search_button = gr.Button("直接检索此片段", variant="primary")

                with gr.Column(scale=3):
                    snippet_gallery_output = gr.Gallery(
                        label="检索结果",
                        show_label=False,
                        columns=3,
                        object_fit="contain",
                        height="auto"
                    )
                    snippet_details_output = gr.Markdown("在这里查看结果详情...")

            snippet_search_button.click(
                fn=search_and_rerank,
                inputs=[snippet_box],
                outputs=[snippet_gallery_output, snippet_details_output]
            )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=True)
