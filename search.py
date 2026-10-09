# search.py - Vector retrieval with per-modality relevance calibration

import numpy as np
import config
import database
from embedder import get_text_embedding

PHOTO_PREFIXES = ("a photo of", "a picture of", "an image of", "a photograph of")


def _best_per_file(rows, query_vec):
    """Score all chunks in one matrix multiply, keep best chunk per file."""
    dim = query_vec.shape[0]
    rows = [r for r in rows if len(r["embedding"]) == dim * 4]  # skip stale vectors
    if not rows:
        return []
    matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    scores = matrix @ query_vec

    best = {}
    for row, score in zip(rows, scores):
        path = row["file_path"]
        if path not in best or score > best[path]["score"]:
            best[path] = {
                "file_path": path,
                "content": row["content"],
                "updated_at": row["updated_at"],
                "type": row["modality"],
                "score": float(score),
            }
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


def semantic_search(query_text, top_k=5):
    print(f"[Search] Querying database for: '{query_text}'")

    conn = database.get_db_connection()
    rows = conn.execute(
        "SELECT file_path, content, updated_at, embedding, modality FROM file_index"
    ).fetchall()
    conn.close()

    if not rows:
        print("[Search] Database is empty. Index some files first.")
        return []

    text_rows = [r for r in rows if r["modality"] == "text"]
    image_rows = [r for r in rows if r["modality"] == "image"]

    text_hits = _best_per_file(text_rows, get_text_embedding(query_text)) if text_rows else []

    # CLIP matches images best with a caption-style prompt (don't double the prefix)
    img_query = query_text.strip()
    if not img_query.lower().startswith(PHOTO_PREFIXES):
        img_query = f"a photo of {img_query}"
    image_hits = _best_per_file(image_rows, get_text_embedding(img_query)) if image_rows else []

    if getattr(config, "SEARCH_DEBUG", False):
        for name, hits in (("text", text_hits), ("image", image_hits)):
            print(f"[Debug] top {name} scores:",
                  [(h['file_path'].split('\\')[-1][:25], round(h['score'], 3)) for h in hits[:5]])

    # Each modality has its own score scale, so judge relevance against its own
    # threshold, then rank everything by score / threshold (comparable across types).
    for h in text_hits:
        h["relevance"] = h["score"] / config.TEXT_MIN_SCORE
    for h in image_hits:
        h["relevance"] = h["score"] / config.IMAGE_MIN_SCORE

    combined = text_hits + image_hits
    strong = [h for h in combined if h["relevance"] >= 1.0]
    strong.sort(key=lambda h: h["relevance"], reverse=True)

    if strong:
        for h in strong:
            h["weak"] = False
        return strong[:top_k]

    # Nothing cleared the bar: show the closest few, clearly flagged as weak
    print("[Search] No confident matches; showing closest results (weak).")
    combined.sort(key=lambda h: h["relevance"], reverse=True)
    for h in combined[:3]:
        h["weak"] = True
    return combined[:3]


if __name__ == "__main__":
    database.init_db()
    query = input("Enter natural language search query: ").strip()
    if query:
        matches = semantic_search(query)
        print(f"\n--- Top {len(matches)} Semantic Matches ---")
        for i, m in enumerate(matches, 1):
            flag = " (weak)" if m.get("weak") else ""
            print(f"{i}. [{m['type']}] Score: {m['score']:.4f}{flag} | File: {m['file_path']}")
            print(f"   Preview: {m['content'][:150]}...\n")
