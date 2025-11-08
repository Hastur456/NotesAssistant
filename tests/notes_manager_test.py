import os
from dotenv import load_dotenv
import pytest
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AGENT.file_manager.notes_manager import NotesManager

load_dotenv()
path_to_notes = os.getenv("NOTES_PATH")


@pytest.fixture
def temp_dir():
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def manager(temp_dir):
    return NotesManager(temp_dir)


def test_init_creates_directory(temp_dir):
    notes_dir = Path(temp_dir) / "notes"
    NotesManager(str(notes_dir))
    assert notes_dir.exists()


def test_create_note(manager):
    result = manager.create_note("test", "content")
    assert result["status"] == "success"
    assert (manager.notes_dir / "test.md").exists()


def test_create_note_file_format(manager):
    manager.create_note("title", "text")
    with open(manager.notes_dir / "title.md") as f:
        assert f.read().startswith("# title\n\n")


def test_edit_existing_note(manager):
    manager.create_note("note", "old")
    manager.edit_note("note.md", "new")
    with open(manager.notes_dir / "note.md") as f:
        assert "new" in f.read()


def test_edit_with_new_title(manager):
    manager.create_note("old", "content")
    manager.edit_note("old.md", "new content", title="new_title")
    with open(manager.notes_dir / "old.md") as f:
        assert "# new_title" in f.read()


def test_edit_nonexistent_note(manager):
    result = manager.edit_note("missing.md", "content")
    assert result["status"] == "error"


def test_delete_existing_note(manager):
    manager.create_note("delete", "content")
    result = manager.delete_note("delete.md")
    assert result["status"] == "success"
    assert not (manager.notes_dir / "delete.md").exists()


def test_delete_nonexistent_note(manager):
    result = manager.delete_note("missing.md")
    assert result["status"] == "error"


def test_workflow_create_edit_delete(manager):
    manager.create_note("note", "v1")
    manager.edit_note("note.md", "v2")
    assert (manager.notes_dir / "note.md").exists()
    result = manager.delete_note("note.md")
    assert result["status"] == "success"
    assert not (manager.notes_dir / "note.md").exists()


def test_multiple_notes(manager):
    for i in range(3):
        manager.create_note(f"note{i}", f"content{i}")
    assert len(list(manager.notes_dir.glob("*.md"))) == 3


def test_special_characters_in_title(manager):
    result = manager.create_note("note_@#$", "content")
    assert result["status"] == "success"


def test_dir_structure_returns_dict(manager):
    manager.create_note("test", "content")
    structure = manager.get_dir_structure()
    assert isinstance(structure, dict)
    assert "name" in structure
    assert "path" in structure
    assert "files" in structure
    assert "directories" in structure


def test_dir_structure_contains_files(manager):
    manager.create_note("note1", "content1")
    manager.create_note("note2", "content2")
    structure = manager.get_dir_structure()
    assert len(structure["files"]) == 2
    assert "note1.md" in structure["files"]
    assert "note2.md" in structure["files"]


def test_dir_structure_with_subdirs(manager):
    manager.create_note("note", "content")
    subdir = manager.notes_dir / "subdir"
    subdir.mkdir()
    structure = manager.get_dir_structure()
    assert len(structure["directories"]) == 1
    assert structure["directories"][0]["name"] == "subdir"


def test_print_tree_debug_output(manager, capsys):
    manager.create_note("note1", "content")
    manager.create_note("note2", "content")
    structure = manager.get_dir_structure()
    manager.print_tree(structure)

    captured = capsys.readouterr()
    assert "📁" in captured.out
    assert "📄" in captured.out
    assert "note1.md" in captured.out
    assert "note2.md" in captured.out
