# SemantiFS

**A privacy-first, fully local semantic file search engine.**

Search your files by *meaning*, not keywords. Type "dog playing on the beach" or "meeting about the budget" and SemantiFS finds matching text files, PDFs, images and audio recordings, all on your own machine with no cloud services.

---

## Features

- **Semantic search** across text, PDFs, images and audio using one shared vector space
- **Text-to-image retrieval**: describe a picture in words and find it
- **Audio search**: recordings are transcribed with Whisper and searched like text
- **PDF handling**: text pages are embedded as text; scanned or slide-style pages are rendered and embedded as images
- **Hybrid ranking**: semantic score plus a boost for filename matches
- **Live indexing**: a background watchdog daemon indexes new, changed, moved and deleted files automatically (with debouncing)
- **Two interfaces**: a command-line app and a Windows-Explorer-style GUI with switchable themes
- **Weak-match fallback**: if nothing clears the confidence threshold, the closest results are shown and flagged `(weak)`

---

## How it works

```
INDEXING
  file -> extractor.py -> text / image segments -> embedder.py -> vectors -> SQLite

SEARCHING
  query -> embedder (text tower) -> search.py (cosine similarity + thresholds + filename boost) -> ranked results
```

1. **Extraction** (`extractor.py`): each file is split into segments. Text files become text, audio is transcribed by faster-whisper, PDF pages become text or images, and image files stay images.
2. **Embedding** (`embedder.py`): MobileCLIP2-S0 (ONNX, CPU) turns text chunks and images into L2-normalized vectors. Text is chunked at 300 characters with a stride of 150.
3. **Storage** (`database.py`): vectors are saved as float32 BLOBs in SQLite (WAL mode) in the `file_index` table.
4. **Search** (`search.py`): the query is embedded, all vectors are scored with a single matrix multiply, the best chunk per file is kept, and results are ranked by `score / threshold` so text and image scores are comparable.

---

## Project structure

| File | Purpose |
|---|---|
| `config.py` | Watch folders, supported extensions, thresholds, debounce time |
| `database.py` | SQLite connection (WAL) and schema/migration |
| `extractor.py` | Converts files into text/image segments |
| `embedder.py` | MobileCLIP2 ONNX embeddings, chunking, index storage |
| `search.py` | Vector retrieval, filtering, hybrid ranking |
| `watchdog_daemon.py` | Debounced auto-indexing of watched folders |
| `cli_app.py` | Command-line interface |
| `ui.py` | CustomTkinter GUI |
| `themes.json` | Color palettes (Dark, Cyberpunk) |
| `audio_extractor.py` | Standalone Whisper helper (not used by the main pipeline) |
| `models/` | ONNX models and tokenizer (not included in the repo) |

---

## Supported file types

| Category | Extensions |
|---|---|
| Text | `.txt .md .py .json .csv .log` |
| Documents | `.pdf` |
| Images | `.png .jpg .jpeg .webp` |
| Audio | `.mp3 .wav .m4a .flac .ogg .mpeg` |

---

## Installation

**Requirements:** Python 3.10+ (the UI uses `Path | None` type syntax).

```bash
pip install numpy onnxruntime pillow tokenizers pymupdf watchdog faster-whisper customtkinter
```

### Model files

Place the MobileCLIP2-S0 ONNX exports and tokenizer in a `models/` folder next to the code:

```
models/
├── text.onnx
├── visual.onnx
└── tokenizer.json
```

Note: faster-whisper downloads the `small` model from Hugging Face on first use, so the first audio transcription needs internet access. After that it is cached and works offline.

---

## Configuration

Edit `config.py`:

```python
WATCH_FOLDERS = [r"C:\Users\Kartik\Documents\SemantiFS\Test_folder"]
IMAGE_MIN_SCORE = 0.20   # text->image cosine threshold
TEXT_MIN_SCORE = 0.60    # text->text cosine threshold
DEBOUNCE_SECONDS = 2.0
PDF_EMBED_ALL_PAGES_AS_IMAGES = False
SEARCH_DEBUG = False     # True prints raw top scores per modality
```

Set `SEARCH_DEBUG = True` while tuning thresholds.

---

## Usage

### GUI

```bash
python ui.py
```

Pages: **All Files**, **Recent**, **SemantiFS** (indexed files), **Search Index**. Use the sidebar dropdown to switch themes. Double-click a result to open its location in the file explorer.

### CLI

```bash
python cli_app.py
```

Choose to index a file manually or run a natural-language search with an optional filter (text, images, audio, PDFs).

### Auto-indexing daemon

```bash
python watchdog_daemon.py
```

Watches every folder in `WATCH_FOLDERS`. A file is indexed after it has been unchanged for `DEBOUNCE_SECONDS`. Moves are handled as delete-old plus index-new, and deletions remove the file from the index.

---

## Database schema

```sql
CREATE TABLE file_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT,            -- first 200 chars, used as a display snippet
    embedding BLOB,          -- float32 vector
    modality TEXT DEFAULT 'text',
    updated_at TIMESTAMP,
    UNIQUE(file_path, chunk_index)
);
```

---

## Known limitations

- No OCR, so scanned PDFs are matched visually only
- CLIP is weaker at text-to-text matching than dedicated sentence-embedding models
- The daemon does not scan existing files at startup; only changes are indexed
- Indexing is sequential, so long audio files delay the queue
- Search is brute-force NumPy, which suits personal-scale collections
- The filename boost checks the full path rather than just the file name
- Nav-bar back, up and search buttons in the GUI are placeholders

## Future work

OCR for scanned PDFs, an initial folder scan, FAISS/HNSW vector index, a dedicated text embedding model, Word/PowerPoint/video support, result thumbnails, and search by example image.

---

## Tech stack

Python, MobileCLIP2-S0, ONNX Runtime, HuggingFace Tokenizers, faster-whisper, PyMuPDF, SQLite, NumPy, watchdog, CustomTkinter
