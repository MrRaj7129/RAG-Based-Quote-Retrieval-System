# streamlit_app/utils.py
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple

# ---------------- CONFIG (adjust paths if needed) ----------------
DATA_DIR = os.getenv("DATA_DIR", "data")              # place your files in /app/data or data/
DF_PATH = os.path.join(DATA_DIR, "quotes.parquet")    # expected processed dataframe
EMB_PATH = os.path.join(DATA_DIR, "quote_embeddings.npy")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
EMBED_MODEL_NAME = "all-mpnet-base-v2"
EMBED_DEVICE = "cpu"   # change to "cuda" if you deploy on GPU-enabled host

# ---------------- Utilities ----------------
def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ---------------- Load data frame ----------------
def load_df(parquet_path: str = DF_PATH) -> pd.DataFrame:
    """
    Load processed dataframe containing columns:
     - id, quote_clean, author_clean, tags_clean, combined_text
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"{parquet_path} not found. Place processed dataframe at this path.")
    df = pd.read_parquet(parquet_path)
    # minimal validation
    required = {"id", "quote_clean", "author_clean", "tags_clean", "combined_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataframe missing required columns: {missing}")
    return df.reset_index(drop=True)

# ---------------- Embeddings ----------------
def load_embeddings(emb_path: str = EMB_PATH, mmap_mode: str = "r") -> np.ndarray:
    """
    Memory-map embeddings so they don't all load into RAM at once.
    Expects float32 numpy array of shape (N, dim).
    """
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings file not found at {emb_path}")
    emb = np.load(emb_path, mmap_mode=mmap_mode)
    # ensure float32
    if emb.dtype != np.float32:
        # create a memmapped copy if needed (safe fallback)
        emb = emb.astype("float32")
    return emb

# ---------------- FAISS index ----------------
def load_or_build_faiss(embeddings: np.ndarray, index_path: str = FAISS_INDEX_PATH) -> faiss.Index:
    """
    Try to load a saved FAISS index file. If it does not exist, build a simple IndexFlatIP
    on the provided embeddings (assumes embeddings are already normalized for cosine).
    """
    d = embeddings.shape[1]
    if os.path.exists(index_path):
        try:
            idx = faiss.read_index(index_path)
            return idx
        except Exception as e:
            # if reading fails, fall back to building index below
            print("Warning: failed to read FAISS index (will rebuild):", e)

    # Build a simple IndexFlatIP (fast to construct but memory heavy)
    index = faiss.IndexFlatIP(d)
    # make contiguous float32
    arr = np.ascontiguousarray(embeddings.astype("float32"))
    index.add(arr)
    # try saving (best-effort)
    try:
        _ensure_dir(index_path)
        faiss.write_index(index, index_path)
    except Exception:
        pass
    return index

# ---------------- Embedding model loader ----------------
def load_embed_model(model_name: str = EMBED_MODEL_NAME, device: str = EMBED_DEVICE) -> SentenceTransformer:
    """
    Load the sentence-transformers model (kept in memory).
    """
    model = SentenceTransformer(model_name, device=device)
    return model

# ---------------- init resources ----------------
_cached_resources = None

def init_resources(df_path: str = DF_PATH,
                   emb_path: str = EMB_PATH,
                   index_path: str = FAISS_INDEX_PATH,
                   model_name: str = EMBED_MODEL_NAME,
                   device: str = EMBED_DEVICE):
    """
    Loads/returns (df, embeddings, embed_model, faiss_index). Cached for reuse.
    """
    global _cached_resources
    if _cached_resources is not None:
        return _cached_resources

    df = load_df(df_path)
    embeddings = load_embeddings(emb_path)
    # Note: we expect embeddings to be normalized (L2). If they are not, the retrieval code below will normalize queries and perform inner-product anyway.
    index = load_or_build_faiss(embeddings, index_path)
    embed_model = load_embed_model(model_name, device=device)

    _cached_resources = (df, embeddings, embed_model, index)
    return _cached_resources

# ---------------- Retrieval ----------------
def _normalize_query_vector(vec: np.ndarray) -> np.ndarray:
    """L2 normalize the vector(s) for cosine via inner product."""
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vec / norms

def retrieve_top_k(query: str,
                   df,
                   embed_model: SentenceTransformer,
                   index: faiss.Index,
                   embeddings: np.ndarray,
                   k: int = 5) -> List[Dict]:
    """
    Encode query, search FAISS index, and return top-k hits as dicts:
      {id, quote, author, tags, combined_text, score, index}
    """
    # encode and normalize
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = _normalize_query_vector(q_emb.astype("float32"))
    # search
    sims, ids = index.search(q_emb, k)
    sims = sims[0].tolist()
    ids = ids[0].tolist()

    results = []
    for score, idx in zip(sims, ids):
        if idx < 0 or idx >= len(df):
            continue
        row = df.iloc[idx]
        results.append({
            "id": row["id"],
            "quote": row.get("quote_clean", row.get("quote", "")),
            "author": row.get("author_clean", row.get("author", "Unknown")),
            "tags": row.get("tags_clean", []),
            "combined_text": row.get("combined_text", ""),
            "score": float(score),
            "index": int(idx)
        })
    return results

# ---------------- Example helper for Streamlit ----------------
def search_and_format(query: str, top_k: int = 5, device: str = EMBED_DEVICE):
    """
    Convenience function to initialize resources (if needed) and run a search.
    Returns (df, results)
    """
    df, embeddings, embed_model, index = init_resources(device=device)
    results = retrieve_top_k(query, df, embed_model, index, embeddings, k=top_k)
    return df, results
