# embedder.py - ONNX MobileCLIP2-S0 integration layer (fixed)

import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

import database
import extractor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_ENCODER_PATH = os.path.join(BASE_DIR, "models", "text.onnx")
IMAGE_ENCODER_PATH = os.path.join(BASE_DIR, "models", "visual.onnx")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.json")

CONTEXT_LENGTH = 77
DEFAULT_IMAGE_SIZE = 256  # MobileCLIP2-S0 uses 256x256

# ---------------------------------------------------------------- sessions
# FIX: sessions were re-created on EVERY call (once per text chunk!). Cache them.
_sessions = {}


def _get_session(path):
    if path not in _sessions:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing ONNX model at {path}")
        _sessions[path] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _sessions[path]


# --------------------------------------------------------------- tokenizer
_tokenizer = None
SOT_ID, EOT_ID = 49406, 49407  # CLIP BPE start/end-of-text defaults


def _get_tokenizer():
    global _tokenizer, SOT_ID, EOT_ID
    if _tokenizer is None:
        if not os.path.exists(TOKENIZER_PATH):
            # FIX: the old ord(c) fallback produced garbage vectors silently.
            raise FileNotFoundError(f"Missing tokenizer at {TOKENIZER_PATH}")
        _tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        _tokenizer.no_padding()
        _tokenizer.no_truncation()
        SOT_ID = _tokenizer.token_to_id("<|startoftext|>") or SOT_ID
        EOT_ID = _tokenizer.token_to_id("<|endoftext|>") or EOT_ID
    return _tokenizer


def tokenize_text(text):
    """
    OpenCLIP format: [SOT] tokens... [EOT] 0 0 0 ... (length 77).
    The text tower pools at argmax(token_id), i.e. at the EOT token, so
    SOT/EOT MUST be present or the embedding is meaningless.
    """
    tok = _get_tokenizer()
    text = " ".join(text.lower().split())
    ids = tok.encode(text, add_special_tokens=False).ids[: CONTEXT_LENGTH - 2]
    ids = [SOT_ID] + ids + [EOT_ID]
    ids += [0] * (CONTEXT_LENGTH - len(ids))

    session = _get_session(TEXT_ENCODER_PATH)
    dtype = np.int32 if "int32" in session.get_inputs()[0].type else np.int64
    return np.array([ids], dtype=dtype)


def _normalize(vec):
    vec = vec.astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# -------------------------------------------------------------- embeddings
def get_text_embedding(text_content):
    session = _get_session(TEXT_ENCODER_PATH)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tokenize_text(text_content)})
    return _normalize(outputs[0][0])


def preprocess_image(image_obj: Image.Image, size=DEFAULT_IMAGE_SIZE):
    """
    MobileCLIP preprocessing: resize shortest side -> center crop,
    scale to [0,1], NO mean/std normalization (mean=0, std=1).
    FIX: old code squashed the aspect ratio and used (x-0.5)/0.5.
    """
    img = image_obj.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))),
                     Image.Resampling.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(arr, 0)


def get_image_embedding(image_obj: Image.Image):
    session = _get_session(IMAGE_ENCODER_PATH)
    inp = session.get_inputs()[0]
    size = inp.shape[2] if isinstance(inp.shape[2], int) else DEFAULT_IMAGE_SIZE
    outputs = session.run(None, {inp.name: preprocess_image(image_obj, size)})
    return _normalize(outputs[0][0])


# ----------------------------------------------------------------- storage
def remove_file_from_index(file_path):
    conn = database.get_db_connection()
    conn.execute("DELETE FROM file_index WHERE file_path = ?", (file_path,))
    conn.commit()
    conn.close()
    print(f"[Embedder] Removed from index: {os.path.basename(file_path)}")


def process_and_store_file(file_path):
    print(f"[Embedder] Processing file: {file_path}")
    # Compute everything FIRST so a failure doesn't leave the file un-indexed.
    # Every segment is processed: all text pages AND all image-only pages of a PDF.
    rows = []
    chunk_idx = 0  # running index, unique per file
    chunk_size, stride = 300, 150
    try:
        for kind, payload, label in extractor.extract_segments(file_path):
            tag = f"[{label}] " if label else ""
            if kind == "text":
                for i in range(0, max(1, len(payload)), stride):
                    chunk = payload[i:i + chunk_size]
                    if not chunk.strip():
                        continue
                    vec = get_text_embedding(chunk)
                    rows.append((file_path, chunk_idx, (tag + chunk)[:200], vec.tobytes(), "text"))
                    chunk_idx += 1
            elif kind == "image":
                vec = get_image_embedding(payload)
                desc = f"[Visual Asset / Rendered Slide] {label}".strip()
                rows.append((file_path, chunk_idx, desc, vec.tobytes(), "image"))
                chunk_idx += 1
    except Exception as e:
        print(f"[Embedder Error] Could not process {file_path}: {e}")
        return

    conn = database.get_db_connection()
    try:
        conn.execute("DELETE FROM file_index WHERE file_path = ?", (file_path,))
        conn.executemany(
            "INSERT INTO file_index (file_path, chunk_index, content, embedding, modality, updated_at) "
            "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)", rows)
        conn.commit()
    finally:
        conn.close()
    print(f"[Embedder Success] Indexed {len(rows)} chunk(s) for: {os.path.basename(file_path)}")


if __name__ == "__main__":
    database.init_db()
    test_file = input("Enter file path to index and embed: ").strip().strip('"')
    if os.path.exists(test_file):
        process_and_store_file(test_file)
    else:
        print("File path does not exist.")
