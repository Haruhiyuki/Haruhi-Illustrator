import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from BCEmbedding import EmbeddingModel

# 尝试导入FAISS，如果失败则给出提示
try:
    import faiss
except ImportError:
    raise SystemExit("[!] 缺少依赖，请运行: pip install faiss-cpu (或 faiss-gpu)")

# ---------- 路径 & 参数 ----------
CAPTIONS_FILE = "dataset/captions.json"
FAISS_INDEX_PATH = "dataset/captions.faiss"
DOCS_PATH = "dataset/captions.docs.pkl"

# --- 模型配置 ---
# 参考: https://huggingface.co/maidalun1020/bce-embedding-base_v1
EMBED_MODEL_NAME = "maidalun1020/bce-embedding-base_v1"
VECTOR_DIM = 768  # bce-embedding-base_v1 的维度是 768
BATCH_SIZE = 128  # 根据你的硬件（CPU/GPU内存）调整

# ---------- 初始化 ----------
# 1. 加载嵌入模型
print(f"Loading embedding model: {EMBED_MODEL_NAME}...")
# use_fp16=True 在支持的GPU上可以加速
embed_model = EmbeddingModel(model_name_or_path=EMBED_MODEL_NAME, use_fp16=True)
print("[✓] Embedding model loaded.")

# 2. 加载所有本地的描述数据
try:
    with open(CAPTIONS_FILE, "r", encoding="utf8") as f:
        captions = json.load(f)
    # --- 关键修复：使用正确的变量名 'captions' ---
    all_local_paths = set(captions.keys())
except FileNotFoundError:
    raise SystemExit(f"[!] Caption file not found: {CAPTIONS_FILE}")
except json.JSONDecodeError:
    raise SystemExit(f"[!] Failed to parse JSON from: {CAPTIONS_FILE}")

# 3. 加载或创建FAISS索引和文档列表
faiss_index = None
documents = []
existing_paths = set()

if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCS_PATH):
    try:
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            documents = pickle.load(f)
        existing_paths = {doc["path"] for doc in documents}
        print(f"[i] Found {len(existing_paths)} existing items in index.")
    except Exception as e:
        print(f"[!] Warning: Failed to load existing index. Rebuilding from scratch. Error: {e}")
        faiss_index = None
        documents = []
        existing_paths = set()

if faiss_index is None:
    print("Creating a new FAISS index...")
    # IndexFlatIP 对应余弦相似度，因为BCEmbedding推荐使用归一化向量
    faiss_index = faiss.IndexFlatIP(VECTOR_DIM)

# 4. 筛选出需要处理的新数据
paths_to_process = sorted([p for p in all_local_paths if p not in existing_paths])
if not paths_to_process:
    raise SystemExit(f"[✓] All {len(all_local_paths)} captions are already indexed. Nothing to do.")
print(f"[*] Found {len(paths_to_process)} new items to index.")

# 5. 批量处理并添加到索引
for i in tqdm(range(0, len(paths_to_process), BATCH_SIZE), desc="Embedding and Indexing"):
    batch_paths = paths_to_process[i:i + BATCH_SIZE]
    batch_texts = [captions[k] for k in batch_paths]

    try:
        # 调用 BCEmbedding API 获取 embeddings
        embeddings = embed_model.encode(batch_texts, batch_size=len(batch_texts), normalize_embeddings=True)

        # 添加到FAISS索引
        faiss_index.add(embeddings)

        # 添加文档元数据
        for path, text in zip(batch_paths, batch_texts):
            documents.append({"path": path, "text": text})

    except Exception as e:
        print(f"\n[!] An unexpected error occurred on batch starting with '{batch_paths[0]}'. Skipping. Error: {e}")

# 6. 保存更新后的索引和文档
try:
    print("\nSaving index and documents to disk...")
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)
    print(f"[✓] Successfully indexed {len(paths_to_process)} new items.")
    print(f"    - FAISS index saved to: {FAISS_INDEX_PATH}")
    print(f"    - Documents saved to: {DOCS_PATH}")
    print(f"    - Total items in index: {faiss_index.ntotal}")
except Exception as e:
    print(f"\n[!] Failed to save index or documents. Error: {e}")
