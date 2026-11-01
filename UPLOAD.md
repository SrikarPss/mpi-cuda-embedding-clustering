# Upload distributed-clustering to GitHub

Run these commands from this directory after creating an empty GitHub repository:

```bash
git init -b main
git add .
git status --short
git commit -m "Initial distributed-clustering implementation"
git remote add origin <your-repository-url>
git push -u origin main
```

Use this folder's contents as the repository root. The project includes its own `native/`, `src/`, `scripts/`, `tests/`, build configuration, and dependency files. No parent or sibling files are needed. Git ignores `.venv`, build outputs, caches, local logs, and generated results; compact published examples live under `examples/`.
