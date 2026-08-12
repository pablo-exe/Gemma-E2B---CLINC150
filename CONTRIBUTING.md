# Contributing

## Workflow

1. Create an issue or define the experiment objective.
2. Branch from `main` using `feat/`, `fix/`, `docs/` or `experiment/`.
3. Keep commits focused and use conventional commit messages.
4. Run the local quality checks.
5. Open a pull request using the repository template.
6. Merge only after CI passes and the experiment protocol remains satisfied.

```powershell
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=gemma_clinc
```

Experiment outputs should not be committed directly. Add concise, reviewed result summaries to `docs/results/` in later phases and link them to the immutable run metadata.

## Commit examples

```text
feat: add zero-shot Gemma baseline
test: cover invalid label parsing
docs: record phase 1 experiment results
fix: preserve OOS examples during evaluation
```
