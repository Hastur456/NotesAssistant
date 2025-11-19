from pathlib import Path
import os


class NotesManager():
    def __init__(self, notes_directory):
        self.notes_dir = Path(notes_directory)
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def read_note(self, filename: str):
        filepath = self.notes_dir / filename

        if not filepath.exists():
            return {
                "status": "error",
                "message": f"Заметка {filepath} не найдена."
            }

        with open(filepath, mode="r", encoding="utf-8") as f:
            content = f.read()

        return {
            "status": "success",
            "filename": filename,
            "content": content,
            "metadata": self.metadata.get(filename, {})
        }

    def create_note(self, title, content):
        filename = title + ".md"
        filepath = self.notes_dir / filename

        with open(filepath, mode="w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(content)

        return {
            "status": "success",
            "filename": filename,
            "path": str(filepath),
            "title": title,
            "message": f"Заметка '{title}' успешно создана"
        }

    def edit_note(self, filename, content, title: str | None = None):
        filepath = self.notes_dir / filename

        if not filepath.exists():
            return {
                "status": "error",
                "message": f"Заметка {filepath} не найдена."
            }
        
        with open(filepath, "r", encoding="utf-8") as f:
            old_lines = f.readlines()
            
        if not title:
            title = old_lines[0].replace("# ", "").strip() if old_lines else "Без названия"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(content)

        return {
            "status": "success",
            "filename": filename,
            "message": f"Заметка '{title}' успешно обновлена"
        }

    def delete_note(self, filename):
        filepath = self.notes_dir / filename

        if not filepath.exists():
            return {
                "status": "error",
                "message": f"Заметка {filepath} не найдена."
            }

        filepath.unlink()

        return {
            "status": "success",
            "message": f"Заметка '{filename}' удалена"
        }

    def get_dir_structure(self, root_path: str|None = None):
        root_path = root_path if root_path else str(self.notes_dir) 
        tree = {'name': os.path.basename(root_path), 'path': os.path.abspath(root_path), 'files': [], 'directories': []}
        dir_map = {os.path.abspath(root_path): tree}

        for dirpath, dirnames, filenames in os.walk(root_path):
            current_node = dir_map[os.path.abspath(dirpath)]
            for d in dirnames:
                abs_dir = os.path.abspath(os.path.join(dirpath, d))
                new_node = {'name': d, 'path': abs_dir, 'files': [], 'directories': []}
                current_node['directories'].append(new_node)
                dir_map[abs_dir] = new_node
            current_node['files'].extend(filenames)

        return {
            "status": "success",
            "count": len(tree),
            "tree": tree
        }

    def print_tree(self, node, indent=0):
        print('  ' * indent + f"P-{node['name']}")
        for file in node['files']:
            print('  ' * (indent + 1) + f"f-{file}")
        for directory in node['directories']:
            self.print_tree(directory, indent + 1)
