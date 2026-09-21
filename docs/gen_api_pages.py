"""Generate the code API reference pages as static Markdown files.

This is a pre-build step: the generated files under docs/code_reference/ are
picked up by literate-nav during the documentation build.

Usage: uv run python docs/gen_api_pages.py
"""

from pathlib import Path

SRC_PATH = "./src/"
REF_PATH = "code_reference"
IGNORED_FILES = ["__init__.py", "multimodal.py"]


class Nav(dict):
    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            parent = super().setdefault(item[0], Nav())
            return parent[item[1:]]
        return super().__getitem__(item)

    def __setitem__(self, item, value):
        if isinstance(item, (tuple, list)):
            if len(item) == 1:
                super().__setitem__(item[0], value)
            else:
                parent = super().setdefault(item[0], Nav())
                parent[item[1:]] = value
        else:
            super().__setitem__(item, value)

    def build_literate_nav(self, indent=4):
        def render(nav, level=0):
            items = sorted(
                nav.items(), key=lambda kv: (isinstance(kv[1], dict), str(kv[0]))
            )
            for title, target in items:
                if isinstance(target, str):
                    yield " " * (level + 1) + f"* [{title}]({target})\n"
                else:
                    yield " " * (level + 1) + f"* {title}\n"
                    yield from render(target, level + indent)

        return list(render(self))


def main():
    nav = Nav()
    out_root = Path("./docs") / REF_PATH

    for path in sorted(Path(SRC_PATH).rglob("*.py")):
        module_path = path.relative_to(SRC_PATH).with_suffix("")
        doc_path = path.relative_to(SRC_PATH).with_suffix(".md")
        full_doc_path = Path(REF_PATH, doc_path)

        parts = list(module_path.parts)

        if parts[-1] == "__init__":
            parts = parts[:-1]
        elif parts[-1] == "__main__":
            continue

        if not parts:
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
            nav["index"] = doc_path.as_posix()
            source = Path(f"./docs/{REF_PATH}/index.md").read_text(encoding="utf-8")
        elif not any(ignored in path.name for ignored in IGNORED_FILES):
            nav[parts] = doc_path.as_posix()
            source = "::: " + ".".join(parts) + "\n"
        else:
            continue

        out_path = Path("./docs") / full_doc_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(source, encoding="utf-8")

    summary = out_root / "SUMMARY.md"
    summary.write_text("".join(nav.build_literate_nav()), encoding="utf-8")


if __name__ == "__main__":
    main()
