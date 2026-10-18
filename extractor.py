# extractor.py - Universal extractor: yields segments for EVERY part of a file
#
# extract_segments(path) is a generator of (kind, payload, label):
#   kind     : 'text' | 'image'
#   payload  : str (text) | PIL.Image (image)
#   label    : e.g. "p.3" for PDF pages, "" otherwise
# A PDF can yield a mix: text pages as text, text-less pages (scans / slides
# exported as images) as images. Pages are yielded one at a time so large PDFs
# don't hold every rendered page in memory.

import os
from PIL import Image
import pymupdf as fitz

import config

TEXT_EXTS = {'.txt', '.md', '.py', '.json', '.csv', '.log'}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}
AUDIO_EXTS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg','.mpeg'}
MIN_PAGE_TEXT_CHARS = 20   # pages with less text than this are treated as images
PDF_RENDER_DPI = 100       # model input is 256px, so 150 dpi was wasted work

# Lazy-loaded whisper model instance to avoid heavy startup overhead
_audio_model = None

def _get_whisper_model():
    global _audio_model
    if _audio_model is None:
        from faster_whisper import WhisperModel
        _audio_model = WhisperModel("small", device="cpu", compute_type="int8")
    return _audio_model


def extract_segments(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in TEXT_EXTS:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            yield ('text', f.read(), "")

    elif ext in AUDIO_EXTS:
        model = _get_whisper_model()
        try:
            segments, _ = model.transcribe(file_path, beam_size=5)
            transcript = " ".join([segment.text for segment in segments]).strip()
            if transcript:
                yield ('text', transcript, "")
        except Exception as e:
            print(f"Error transcribing audio file {file_path}: {e}")

    elif ext == '.pdf':
        with fitz.open(file_path) as doc:
            for page_index, page in enumerate(doc):
                label = f"p.{page_index + 1}"
                page_text = page.get_text().strip()
                has_text = len(page_text) >= MIN_PAGE_TEXT_CHARS

                if has_text:
                    yield ('text', page_text, label)

                # Render the page as an image if it has no real text, or if the
                # config asks for every page (useful for diagrams/charts in slides)
                if (not has_text) or getattr(config, "PDF_EMBED_ALL_PAGES_AS_IMAGES", False):
                    pix = page.get_pixmap(dpi=PDF_RENDER_DPI)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    yield ('image', img, label)

    elif ext in IMAGE_EXTS:
        yield ('image', Image.open(file_path).convert("RGB"), "")

    else:
        raise ValueError(f"Unsupported extension: {ext}")


if __name__ == "__main__":
    test_path = input("Enter a file path to test extraction: ").strip().strip('"')
    if os.path.exists(test_path):
        for kind, payload, label in extract_segments(test_path):
            if kind == 'text':
                print(f"[text  {label:>6}] {len(payload)} chars | {payload[:80]!r}")
            else:
                print(f"[image {label:>6}] size={payload.size}")
    else:
        print("File path does not exist.")
