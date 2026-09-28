import os

# Exactly matches your project structure from VS Code
PROJECT_FOLDERS = ['deribit', 'benchmarks', 'tests']
ROOT_FILES = ['README.md', 'pyproject.toml', 'Makefile', 'test.py']

def build_clean_bundle(output_file="prism_bundle.txt"):
    file_count = 0
    line_count = 0

    with open(output_file, "w", encoding="utf-8") as out:
        # 1. Add top-level project files
        for f in ROOT_FILES:
            if os.path.exists(f):
                file_count += 1
                out.write(f"\n{'='*80}\nFILE: {f}\n{'='*80}\n\n")
                with open(f, "r", encoding="utf-8", errors="ignore") as rf:
                    content = rf.read()
                    line_count += content.count('\n') + 1
                    out.write(content)

        # 2. Add code files from your specific project directories
        for folder in PROJECT_FOLDERS:
            if not os.path.exists(folder):
                continue
            for root, dirs, files in os.walk(folder):
                dirs[:] = [d for d in dirs if d not in {'__pycache__', '.pytest_cache'}]
                for file in sorted(files):
                    if file.endswith(".py") or file.endswith(".md") or file.endswith(".toml"):
                        path = os.path.join(root, file)
                        file_count += 1
                        out.write(f"\n{'='*80}\nFILE: {path}\n{'='*80}\n\n")
                        with open(path, "r", encoding="utf-8", errors="ignore") as rf:
                            content = rf.read()
                            line_count += content.count('\n') + 1
                            out.write(content)

    print(f"Bundle complete! Packaged {file_count} files ({line_count:,} total lines) into '{output_file}'.")

if __name__ == "__main__":
    build_clean_bundle()