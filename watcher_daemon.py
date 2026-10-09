import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import config
import database
from embedder import process_and_store_file, remove_file_from_index

# Only react to real changes. Newer watchdog versions also emit "opened"/"closed"
# events, and reading a file while indexing could re-trigger indexing in a loop.
RELEVANT_EVENTS = {"created", "modified", "moved", "deleted"}


class DebouncedHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.pending_events = {}  # path -> (timestamp, action)

    def _wanted(self, path):
        if os.path.splitext(path)[1].lower() not in config.ALL_EXTENSIONS:
            return False
        return not any(p in path for p in config.IGNORE_PATTERNS)

    def on_any_event(self, event):
        if event.is_directory or event.event_type not in RELEVANT_EVENTS:
            return

        now = time.time()
        if event.event_type == "moved":
            if self._wanted(event.src_path):
                self.pending_events[event.src_path] = (now, "delete")
            if self._wanted(event.dest_path):
                self.pending_events[event.dest_path] = (now, "index")
        elif event.event_type == "deleted":
            if self._wanted(event.src_path):
                self.pending_events[event.src_path] = (now, "delete")
        else:
            if self._wanted(event.src_path):
                self.pending_events[event.src_path] = (now, "index")

    def process_debounced_files(self):
        while True:
            now = time.time()
            ready = [(p, a) for p, (t, a) in list(self.pending_events.items())
                     if now - t >= config.DEBOUNCE_SECONDS]
            for path, _ in ready:
                self.pending_events.pop(path, None)

            for path, action in ready:
                name = os.path.basename(path)
                try:
                    if action == "delete" or not os.path.exists(path):
                        remove_file_from_index(path)
                    else:
                        print(f"[Daemon] File settled. Auto-indexing: {name}")
                        process_and_store_file(path)
                except Exception as e:
                    print(f"[Daemon Error] Failed on {name}: {e}")
            time.sleep(0.5)


def start_watcher():
    database.init_db()
    handler = DebouncedHandler()
    observer = Observer()
    for folder in config.WATCH_FOLDERS:
        if os.path.exists(folder):
            observer.schedule(handler, path=folder, recursive=True)
            print(f"Watching directory: {folder}")
        else:
            print(f"[Warning] Watch folder not found: {folder}")
    observer.start()
    try:
        handler.process_debounced_files()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    start_watcher()
