import time
import hashlib
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging 

import os
from dotenv import load_dotenv

load_dotenv()
path_to_notes = os.getenv("NOTES_PATH")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.basicConfig(filename='debug.log', level=logging.DEBUG, filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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


