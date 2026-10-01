import os

ignore_dirs = {".git", "__pycache__", "figs", "prism.egg-info", ".pytest_cache", "reports"}
ignore_files = {"snapshots.db"}
allowed_exts = {".py", ".md", ".toml"}

with open("prism_codebase.txt", "w", encoding="utf-8") as out:
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for f in sorted(files):
            if f in ignore_files or not any(f.endswith(ext) for ext in allowed_exts):
                continue
            path = os.path.join(root, f)
            out.write(f"\n==================== {path} ====================\n\n")
            try:
                with open(path, "r", encoding="utf-8") as infile:
                    out.write(infile.read())
            except Exception as e:
                out.write(f"Error reading file: {e}\n")

print("Successfully created prism_codebase.txt!")
