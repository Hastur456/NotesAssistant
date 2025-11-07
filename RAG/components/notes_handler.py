import time
import hashlib
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging 

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RAG.logging_config import logger


class NotesHandler(FileSystemEventHandler):
    def __init__(self, update_callback):
        self.update_callback = update_callback
        self.file_hashes = {}
     
    def on_any_event(self, event):
        if not event.is_directory:
            logging.info(f"Событие: {event.event_type}, Файл: {event.src_path}")

    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith(('.md', '.txt')):
            return
        
        self.update_callback(event.src_path, 'modified')
    
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(('.md', '.txt')):
            return
        
        self.update_callback(event.src_path, 'created')
    
    def on_deleted(self, event):
        if event.is_directory or not event.src_path.endswith(('.md', '.txt')):
            return
        
        self.update_callback(event.src_path, 'deleted')


def update_database_callback(filepath, event_type):
    print(f"  → Обновление БД: {filepath} ({event_type})")


def start_monitoring(notes_dir, update_callback):
    event_handler = NotesHandler(update_callback)
    observer = Observer()
    observer.schedule(event_handler, notes_dir, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Мониторинг за директорией {} заверешен.".format(notes_dir))
    observer.join()
