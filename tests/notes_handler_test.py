import pytest
from unittest.mock import MagicMock
from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileDeletedEvent

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from components.notes_handler import NotesHandler

@pytest.fixture
def notes_handler():
    callback = MagicMock()
    handler = NotesHandler(callback)
    return handler, callback

def test_on_created(notes_handler):
    handler, callback = notes_handler
    event = FileCreatedEvent("/tmp/test_note.md")
    handler.on_created(event)
    callback.assert_called_with("/tmp/test_note.md", 'created')

def test_on_modified(notes_handler):
    handler, callback = notes_handler
    event = FileModifiedEvent("/tmp/test_note.txt")
    handler.on_modified(event)
    callback.assert_called_with("/tmp/test_note.txt", 'modified')

def test_on_deleted(notes_handler):
    handler, callback = notes_handler
    event = FileDeletedEvent("/tmp/test_note.md")
    handler.on_deleted(event)
    callback.assert_called_with("/tmp/test_note.md", 'deleted')

def test_ignore_non_txt_md_files(notes_handler):
    handler, callback = notes_handler
    event = FileCreatedEvent("/tmp/test_note.pdf")
    handler.on_created(event)
    callback.assert_not_called()
