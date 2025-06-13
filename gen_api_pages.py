"""Generate the code API reference pages

Refer to https://mkdocstrings.github.io/recipes/ for more details.
"""

from pathlib import Path

import mkdocs_gen_files

SRC_PATH = "./src/"
REF_PATH = "code_reference"
IGNORED_FILES = ["__init__.py", "__version__.py"]
nav = mkdocs_gen_files.Nav()

for path in sorted(Path(SRC_PATH).rglob("*.py")):
    module_path = path.relative_to(SRC_PATH).with_suffix("")
    doc_path = path.relative_to(SRC_PATH).with_suffix(".md")
    full_doc_path = Path(REF_PATH, doc_path)

    parts = list(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
    elif parts[-1] == "__main__":
        continue

    # Create root index file
    if not parts:
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
        nav["index"] = doc_path.as_posix()
        with open(f"./docs/{REF_PATH}/index.md", "r", encoding="utf-8") as index_file:
            lines = index_file.readlines()
            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                fd.writelines(lines)
        mkdocs_gen_files.set_edit_path(full_doc_path, path)

    # Add all python files that are not in the ignore list
    elif not any(ignored in path.name for ignored in IGNORED_FILES):
        nav[parts] = doc_path.as_posix()
        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            print("::: " + ".".join(parts), file=fd)
        mkdocs_gen_files.set_edit_path(full_doc_path, Path(path))

with mkdocs_gen_files.open(f"{REF_PATH}/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())