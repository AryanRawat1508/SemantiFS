import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "semantifs.db")
WATCH_FOLDERS = [r"F:\TEST FOLDER"]  # raw string: "\T" in a normal string is a latent bug
ALL_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv", ".log",
                  ".pdf", ".png", ".jpg", ".jpeg", ".webp"}
IGNORE_PATTERNS = {"node_modules", ".git", "__pycache__"}
DEBOUNCE_SECONDS = 2.0
PDF_EMBED_ALL_PAGES_AS_IMAGES = False  # True: also embed every text page as an image (diagrams/charts in slides)

# --- Search relevance (starting values; tune using SEARCH_DEBUG output) ---
IMAGE_MIN_SCORE = 0.20   # CLIP text->image cosine; real matches are usually ~0.22-0.35
TEXT_MIN_SCORE = 0.70    # CLIP text->text cosine; unrelated text already sits ~0.5-0.6
SEARCH_DEBUG = False     # True: print top raw scores per modality
