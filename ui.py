import os
import sys
import json
import subprocess
import threading
import time
from pathlib import Path
from tkinter import filedialog, ttk

import customtkinter as ctk

# =============================================================================
# BACKEND - SemantiFS Integration
# =============================================================================
import config
import database
from embedder import process_and_store_file
from search import semantic_search

def get_file_icon(filename: str) -> str:
    """Returns a visual icon based on the file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        return "🖼️"
    elif ext in {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mpeg"}:
        return "🎵"
    elif ext == ".pdf":
        return "📕"
    elif ext in {".txt", ".md", ".py", ".json", ".csv", ".log"}:
        return "📄"
    else:
        return "📄"

class FileBackend:
    def __init__(self, root: Path | None = None):
        # Default to the Test Folder specified in config.py
        default_root = config.WATCH_FOLDERS[0] if config.WATCH_FOLDERS else Path.home() / "Documents"
        self.root = Path(root) if root else Path(default_root)
        database.init_db()

    def get_types(self) -> list[str]:
        return ["All", "Text Documents", "Images", "Audio Transcripts", "PDFs"]

    def _format_size(self, size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

    def _format_date(self, mtime):
        return time.strftime('%m/%d/%Y', time.localtime(mtime))

    def list_all_files(self, limit: int = 2000):
        rows = []
        if not self.root.exists():
            return rows
        for dirpath, _dirs, files in os.walk(self.root):
            for f in files:
                path = Path(dirpath) / f
                display_name = f"{get_file_icon(f)} {f}"
                try:
                    stat = path.stat()
                    rows.append((display_name, self._format_date(stat.st_mtime), self._format_size(stat.st_size), str(path)))
                except Exception:
                    rows.append((display_name, "--", "--", str(path)))
                if len(rows) >= limit:
                    return rows
        return rows

    def list_recent_files(self, limit: int = 50):
        files = []
        if not self.root.exists():
            return files
        for dirpath, _dirs, filenames in os.walk(self.root):
            for f in filenames:
                path = Path(dirpath) / f
                try:
                    stat = path.stat()
                    # Use st_mtime as proxy for recently modified/accessed files
                    files.append((stat.st_mtime, f, self._format_date(stat.st_mtime), self._format_size(stat.st_size), str(path)))
                except Exception:
                    pass
        # Sort descending by time
        files.sort(reverse=True, key=lambda x: x[0])
        # Add icons and format for the UI table
        return [(f"{get_file_icon(f[1])} {f[1]}", f[2], f[3], f[4]) for f in files[:limit]]

    def list_indexed_files(self):
        conn = database.get_db_connection()
        rows = conn.execute("SELECT file_path, COUNT(*) as chunks FROM file_index GROUP BY file_path").fetchall()
        conn.close()
        
        indexed = []
        for r in rows:
            filename = Path(r["file_path"]).name
            display_name = f"{get_file_icon(filename)} {filename}"
            indexed.append((display_name, f"{r['chunks']} chunks", r["file_path"]))
        return indexed

    def index_file(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File path does not exist: {path}")
        process_and_store_file(path)

    def search(self, query: str, type_: str = "All"):
        q = query.strip()
        if not q:
            return []

        filter_map = {"Text Documents": "text", "Images": "image", "Audio Transcripts": "audio", "PDFs": "pdf"}
        results = semantic_search(q, filter_type=filter_map.get(type_))
        
        formatted_results = []
        for res in results:
            filename = Path(res["file_path"]).name
            display_name = f"{get_file_icon(filename)} {filename}"
            score_str = f"{res['score']:.4f}" + (" (weak)" if res.get("weak") else "")
            formatted_results.append((display_name, score_str, res["file_path"]))
            
        return formatted_results


# =============================================================================
# THEME ENGINE (JSON BASED)
# =============================================================================
class ThemeManager:
    themes = {}
    current_theme_name = "Dark"

    @classmethod
    def load_themes(cls):
        theme_path = Path("themes.json")
        if not theme_path.exists():
            cls.themes = {
                "Dark": {"BG": "#202020", "SIDEBAR": "#181818", "SURFACE": "#2b2b2b", "SURFACE_ALT": "#333333", "BORDER": "#383838", "TEXT": "#ffffff", "MUTED": "#a0a0a0", "ACCENT": "#60cdff", "ACCENT_HOVER": "#4ab8e6", "ON_ACCENT": "#000000"},
                "Cyberpunk": {"BG": "#0a0a0c", "SIDEBAR": "#050505", "SURFACE": "#141417", "SURFACE_ALT": "#1f1f24", "BORDER": "#ff007f", "TEXT": "#ffffff", "MUTED": "#00f0ff", "ACCENT": "#7000ff", "ACCENT_HOVER": "#9d4dff", "ON_ACCENT": "#ffffff"}
            }
            with open(theme_path, "w") as f:
                json.dump(cls.themes, f, indent=2)
        else:
            with open(theme_path, "r") as f:
                cls.themes = json.load(f)

    @classmethod
    def apply(cls, name):
        if name in cls.themes:
            cls.current_theme_name = name
            t = cls.themes[name]
            T.BG, T.SIDEBAR = t["BG"], t["SIDEBAR"]
            T.SURFACE, T.SURFACE_ALT = t["SURFACE"], t["SURFACE_ALT"]
            T.BORDER, T.TEXT, T.MUTED = t["BORDER"], t["TEXT"], t["MUTED"]
            T.ACCENT, T.ACCENT_HOVER, T.ON_ACCENT = t["ACCENT"], t["ACCENT_HOVER"], t["ON_ACCENT"]

class T:
    FONT = "Segoe UI" if sys.platform.startswith("win") else "Helvetica"
    BG = SIDEBAR = SURFACE = SURFACE_ALT = BORDER = TEXT = MUTED = ACCENT = ACCENT_HOVER = ON_ACCENT = ""

def font(size=13, weight="normal"): 
    return ctk.CTkFont(family=T.FONT, size=size, weight=weight)


# =============================================================================
# WIDGETS
# =============================================================================
class ExplorerButton(ctk.CTkButton):
    def __init__(self, master, text, command=None, primary=False, width=100, **kw):
        super().__init__(master, text=text, command=command, width=width, height=30, corner_radius=4, border_width=1 if not primary else 0, fg_color=T.ACCENT if primary else T.SURFACE, hover_color=T.ACCENT_HOVER if primary else T.SURFACE_ALT, text_color=T.ON_ACCENT if primary else T.TEXT, border_color=T.BORDER, font=font(12, "bold" if primary else "normal"), **kw)

class ExplorerEntry(ctk.CTkEntry):
    def __init__(self, master, **kw):
        super().__init__(master, height=30, corner_radius=4, border_width=1, fg_color=T.SURFACE, border_color=T.BORDER, text_color=T.TEXT, placeholder_text_color=T.MUTED, font=font(12), **kw)

class ExplorerCombo(ctk.CTkComboBox):
    def __init__(self, master, values, **kw):
        super().__init__(master, values=values, height=30, corner_radius=4, border_width=1, fg_color=T.SURFACE, border_color=T.BORDER, text_color=T.TEXT, button_color=T.SURFACE_ALT, button_hover_color=T.BORDER, dropdown_fg_color=T.SURFACE, dropdown_hover_color=T.SURFACE_ALT, dropdown_text_color=T.TEXT, font=font(12), dropdown_font=font(12), state="readonly", **kw)

class DataTable(ctk.CTkFrame):
    def __init__(self, master, columns, on_open=None):
        super().__init__(master, corner_radius=4, fg_color=T.SURFACE, border_width=1, border_color=T.BORDER)
        self.on_open = on_open
        self.hovered_item = None
        self.original_tags = []  # Track original tags to restore later
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(self, columns=[c[0] for c in columns], show="headings", style="Explorer.Treeview", selectmode="browse")
        for name, width, stretch, anchor in columns:
            self.tree.heading(name, text=name, anchor="w" if anchor == "w" else anchor)
            self.tree.column(name, width=width, minwidth=60, stretch=stretch, anchor=anchor)

        bar = ctk.CTkScrollbar(self, orientation="vertical", corner_radius=4, width=12, command=self.tree.yview, fg_color=T.SURFACE, button_color=T.BORDER, button_hover_color=T.MUTED)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.grid(row=0, column=0, sticky="nsew", padx=(1, 0), pady=1)
        bar.grid(row=0, column=1, sticky="ns", padx=(0, 1), pady=1)
        
        self.tree.bind("<Double-1>", self._double_click)
        self.tree.bind("<Motion>", self._on_hover)
        self.tree.bind("<Leave>", self._on_leave)
        
        self.restyle()

    def set_rows(self, rows):
        self.tree.delete(*self.tree.get_children())
        self.hovered_item = None
        self.original_tags = []
        for i, row in enumerate(rows):
            self.tree.insert("", "end", values=row, tags=("odd" if i % 2 else "even",))

    def _double_click(self, _event):
        sel = self.tree.selection()
        if sel and self.on_open:
            self.on_open(self.tree.item(sel[0], "values"))

    def _get_tags(self, item):
        """Safely parses Tkinter tags, avoiding the string-to-list character splitting bug."""
        tags = self.tree.item(item, "tags")
        if isinstance(tags, str):
            return tags.split()
        return list(tags) if tags else []

    def _on_hover(self, event):
        item = self.tree.identify_row(event.y)
        if self.hovered_item != item:
            self._clear_hover()
            self.hovered_item = item
            if item:
                # Save the current tags (even/odd) and temporarily replace with ONLY hover
                self.original_tags = self._get_tags(item)
                self.tree.item(item, tags=("hover",))

    def _on_leave(self, event):
        self._clear_hover()

    def _clear_hover(self):
        if self.hovered_item and self.tree.exists(self.hovered_item):
            # Restore the original even/odd tags when the mouse leaves
            self.tree.item(self.hovered_item, tags=self.original_tags)
        self.hovered_item = None
        self.original_tags = []

    def restyle(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        # 1. Base style configuration
        style.configure("Explorer.Treeview", background=T.SURFACE, fieldbackground=T.SURFACE, foreground=T.TEXT, rowheight=28, borderwidth=0, relief="flat", font=(T.FONT, 9))
        style.configure("Explorer.Treeview.Heading", background=T.SURFACE_ALT, foreground=T.TEXT, relief="flat", borderwidth=0, padding=(8, 6), font=(T.FONT, 10, "bold"))
        
        # 2. State mappings (Ensures the selection accent color still works)
        style.map("Explorer.Treeview.Heading", background=[("active", T.BORDER)])
        style.map("Explorer.Treeview", background=[("selected", T.ACCENT)], foreground=[("selected", T.ON_ACCENT)])
        style.layout("Explorer.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        
        # 3. Tag configurations
        self.tree.tag_configure("even", background=T.SURFACE)
        self.tree.tag_configure("odd", background=T.SURFACE_ALT)
        
        # T.BORDER provides a distinct background color, T.TEXT keeps the font fully readable
        self.tree.tag_configure("hover", background=T.BORDER, foreground=T.TEXT)
def open_in_explorer(path: str):
    try:
        if sys.platform.startswith("win"): subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        elif sys.platform == "darwin": subprocess.Popen(["open", "-R", path])
        else: subprocess.Popen(["xdg-open", str(Path(path).parent)])
    except Exception: pass

def run_async(app, work, done):
    def target():
        try: result = work()
        except Exception as exc: result = exc
        app.after(0, lambda: done(result))
    threading.Thread(target=target, daemon=True).start()

# =============================================================================
# PAGES
# =============================================================================
class Page(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, corner_radius=0, fg_color=T.BG)
        self.app = app
        self.backend = app.backend
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.nav_bar = ctk.CTkFrame(self, height=45, corner_radius=0, fg_color=T.SURFACE)
        self.nav_bar.grid(row=0, column=0, sticky="ew")
        self.nav_bar.grid_columnconfigure(2, weight=1)

        self.btn_back = ExplorerButton(self.nav_bar, text="⬅", width=40)
        self.btn_back.grid(row=0, column=0, padx=(10, 5), pady=10)
        self.btn_up = ExplorerButton(self.nav_bar, text="⬆", width=40)
        self.btn_up.grid(row=0, column=1, padx=5, pady=10)
        self.path_entry = ExplorerEntry(self.nav_bar, placeholder_text="/virtual/root/directory...")
        self.path_entry.grid(row=0, column=2, padx=10, pady=10, sticky="ew")
        self.path_entry.insert(0, str(self.backend.root))
        self.btn_search_nav = ExplorerButton(self.nav_bar, text="🔍 Search", width=70)
        self.btn_search_nav.grid(row=0, column=3, padx=(5, 10), pady=10)

        self.toolbar = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.toolbar.grid(row=1, column=0, sticky="ew", padx=10, pady=(10, 5))

        self.status = ctk.CTkLabel(self, text="", font=font(11), text_color=T.MUTED, anchor="w")
        self.status.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))

    def on_show(self): pass
    def set_status(self, text): self.status.configure(text=text)

class AllFilesPage(Page):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.toolbar.grid_columnconfigure(0, weight=1)
        self.filter = ExplorerEntry(self.toolbar, placeholder_text="Filter items...")
        self.filter.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.filter.bind("<KeyRelease>", lambda _e: self._apply_filter())
        ExplorerButton(self.toolbar, "Refresh", self.refresh, width=90).grid(row=0, column=1)

        self.table = DataTable(self, [("Name", 300, True, "w"), ("Date Modified", 120, False, "w"), ("Size", 80, False, "e"), ("Path", 400, False, "w")], on_open=lambda v: open_in_explorer(v[3]))
        self.table.grid(row=2, column=0, sticky="nsew", padx=10)
        self.app.tables.append(self.table)
        self._rows = []

    def on_show(self):
        if not self._rows: self.refresh()

    def refresh(self):
        self.set_status("Loading files...")
        run_async(self.app, self.backend.list_all_files, self._loaded)

    def _loaded(self, result):
        if isinstance(result, Exception): self.set_status(f"Error: {result}")
        else:
            self._rows = result
            self._apply_filter()

    def _apply_filter(self):
        f = self.filter.get().strip().lower()
        rows = [r for r in self._rows if f in r[0].lower() or f in r[3].lower()] if f else self._rows
        self.table.set_rows(rows)
        self.set_status(f"{len(rows)} of {len(self._rows)} item(s)")

class RecentFilesPage(Page):
    def __init__(self, master, app):
        super().__init__(master, app)
        ExplorerButton(self.toolbar, "Refresh Recents", self.refresh, width=120).pack(side="left")
        self.table = DataTable(self, [("Name", 300, True, "w"), ("Date Modified", 120, False, "w"), ("Size", 80, False, "e"), ("Path", 400, False, "w")], on_open=lambda v: open_in_explorer(v[3]))
        self.table.grid(row=2, column=0, sticky="nsew", padx=10)
        self.app.tables.append(self.table)

    def on_show(self): self.refresh()
    def refresh(self): run_async(self.app, self.backend.list_recent_files, lambda r: self.table.set_rows(r))

class IndexedFilesPage(Page):
    def __init__(self, master, app):
        super().__init__(master, app)
        ExplorerButton(self.toolbar, "Index File...", self.index_file, primary=True, width=110).pack(side="left")
        ExplorerButton(self.toolbar, "Refresh", self.refresh, width=90).pack(side="left", padx=(8, 0))
        self.table = DataTable(self, [("Name", 300, True, "w"), ("Chunks", 100, False, "e"), ("Path", 480, False, "w")], on_open=lambda v: open_in_explorer(v[2]))
        self.table.grid(row=2, column=0, sticky="nsew", padx=10)
        self.app.tables.append(self.table)

    def on_show(self): self.refresh()
    def refresh(self): run_async(self.app, self.backend.list_indexed_files, lambda r: self.table.set_rows(r))
    def index_file(self):
        paths = filedialog.askopenfilenames(title="Select files to index")
        if not paths: return
        self.set_status(f"Indexing {len(paths)} file(s)...")

        def work():
            success, errors = 0, []
            for p in paths:
                try:
                    self.backend.index_file(p)
                    success += 1
                except Exception as e: errors.append(f"{Path(p).name}: {e}")
            if errors: raise Exception(" | ".join(errors))
            return success

        def done(result):
            self.refresh()
            if isinstance(result, Exception): self.set_status(f"Indexed with errors: {result}")
            else: self.set_status(f"Successfully indexed {result} file(s)")
        run_async(self.app, work, done)

class SearchPage(Page):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.toolbar.grid_columnconfigure(0, weight=1)
        self.query = ExplorerEntry(self.toolbar, placeholder_text="Enter natural language query to search semantics...")
        self.query.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.type_box = ExplorerCombo(self.toolbar, self.backend.get_types(), width=130)
        self.type_box.set(self.backend.get_types()[0])
        self.type_box.grid(row=0, column=1, padx=(0, 8))
        ExplorerButton(self.toolbar, "Search", self.run_search, primary=True, width=90).grid(row=0, column=2)
        self.table = DataTable(self, [("Name", 300, True, "w"), ("Score", 100, False, "e"), ("Path", 480, False, "w")], on_open=lambda v: open_in_explorer(v[2]))
        self.table.grid(row=2, column=0, sticky="nsew", padx=10)
        self.app.tables.append(self.table)

    def run_search(self):
        q, t = self.query.get(), self.type_box.get()
        if q.strip(): run_async(self.app, lambda: self.backend.search(q, t), lambda r: self.table.set_rows(r) if not isinstance(r, Exception) else None)

# =============================================================================
# APP SHELL
# =============================================================================
class App(ctk.CTk):
    def __init__(self, backend: FileBackend | None = None):
        super().__init__()
        self.title("SemantiFS Explorer")
        self.geometry("1100x680")
        self.minsize(850, 520)
        
        self.backend = backend or FileBackend()
        
        # Load and apply initial theme
        ThemeManager.load_themes()
        ThemeManager.apply(ThemeManager.current_theme_name)
        
        self.current_page = "all"
        self.build_ui()

    def build_ui(self):
        """Destroys current UI and rebuilds it instantly to apply new global theme colors."""
        if hasattr(self, 'main_container'):
            self.main_container.destroy()

        # Update root window background
        self.configure(fg_color=T.BG)
        ctk.set_appearance_mode("dark" if ThemeManager.current_theme_name != "Light" else "light")
        
        self.tables = [] # Reset tables reference
        
        self.main_container = ctk.CTkFrame(self, fg_color=T.BG, corner_radius=0)
        self.main_container.pack(fill="both", expand=True)
        self.main_container.grid_columnconfigure(1, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self.main_container, width=180, corner_radius=0, fg_color=T.SIDEBAR, border_width=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(self.sidebar, text="Quick Access", font=font(14, "bold"), text_color=T.TEXT, anchor="w").grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        ctk.CTkFrame(self.main_container, width=1, corner_radius=0, fg_color=T.BORDER).grid(row=0, column=0, sticky="nse")

        self.nav_buttons = {}
        
        def create_nav_btn(row, key, text):
            btn = ctk.CTkButton(self.sidebar, text=text, anchor="w", height=36, corner_radius=4, fg_color="transparent", hover_color=T.SURFACE_ALT, text_color=T.TEXT, font=font(12), command=lambda k=key: self.show_page(k))
            btn.grid(row=row, column=0, sticky="ew", padx=10, pady=2)
            self.nav_buttons[key] = btn

        create_nav_btn(1, "all", "🏠 All Files")
        create_nav_btn(2, "recent", "⭐ Recent")
        create_nav_btn(3, "indexed", "☁️ SemantiFS")
        create_nav_btn(4, "search", "🔍 Search Index")

        # Theme Selector
        ctk.CTkLabel(self.sidebar, text="Theme:", font=font(11), text_color=T.MUTED, anchor="w").grid(row=6, column=0, sticky="ew", padx=15, pady=(0, 0))
        self.theme_combo = ExplorerCombo(self.sidebar, values=list(ThemeManager.themes.keys()), command=self.change_theme)
        self.theme_combo.set(ThemeManager.current_theme_name)
        self.theme_combo.grid(row=7, column=0, sticky="ew", padx=12, pady=(2, 12))

        # Content Area
        self.content = ctk.CTkFrame(self.main_container, corner_radius=0, fg_color=T.BG)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self.pages = {
            "all": AllFilesPage(self.content, self),
            "recent": RecentFilesPage(self.content, self),
            "indexed": IndexedFilesPage(self.content, self),
            "search": SearchPage(self.content, self)
        }
        
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")

        # Restore last viewed page
        self.show_page(self.current_page)

    def show_page(self, key: str):
        self.current_page = key
        for k, btn in self.nav_buttons.items():
            btn.configure(fg_color=T.SURFACE_ALT if k == key else "transparent", font=font(12, "bold" if k == key else "normal"))
        page = self.pages[key]
        page.tkraise()
        page.on_show()

    def change_theme(self, selection):
        ThemeManager.apply(selection)
        self.build_ui() # Instantly rebuilds UI with new colors

if __name__ == "__main__":
    App().mainloop()