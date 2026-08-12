# Architecture

The implementation separates experiment policy from model execution:

```text
YAML config
    │
    ├── official dataset downloader ──> immutable CLINC150 examples
    │
    └── prompt builder ──> Gemma inference ──> strict parser
                                                │
                                                ├── predictions.jsonl
                                                ├── run_metadata.json
                                                └── metrics.json
```

## Design decisions

- **Official source:** data is downloaded from `clinc/oos-eval`, avoiding hidden dataset transformations.
- **Windows CUDA resolution:** `torch==2.11.0` is routed to PyTorch's official `cu128` index only on Windows. The unqualified default resolved a CPU wheel during setup, which would ignore the available RTX 4070. This is a deliberate scope decision for the current Windows development environment; it does not claim that CUDA is unavailable on Linux.
- **Configuration as data:** model, prompt, split and generation settings live in committed YAML files.
- **Lazy GPU imports:** tests and analysis can run without installing the large GPU dependency group.
- **Auditable outputs:** raw generations are retained next to parsed predictions.
- **Strict parsing:** formatting failures are measured rather than silently repaired.
- **No committed weights or raw data:** large artifacts and third-party content remain outside Git.

The model backend is intentionally behind a small `predict` interface. Later phases can add a fine-tuned adapter or a mock backend without changing the evaluator.
