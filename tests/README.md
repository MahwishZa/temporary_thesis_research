# `tests/`

```bash
python -m pytest              # everything: this suite and rag2/tests
python -m pytest tests/unit   # or one level
```

`conftest.py` at the repository root puts `src/` on `sys.path`, so the suite runs
in a fresh checkout without `pip install -e .`.

| Level | What belongs here |
| --- | --- |
| `unit/` | one module, synthetic inputs, no other stage involved |
| `integration/` | two or more stages composed — chunk → embed → retrieve; acquisition end to end |
| `smoke/` | the whole architecture over a synthetic corpus, with a hash encoder |

Every test is offline: no network, no corpus files, no model weights, no GPU.

Two tests are guards rather than tests of behaviour, and both are written to fail
if they ever stop being able to fail:

- `unit/test_condition_isolation.py` — the baseline path must never read a
  publication date, and the reproduced `rag2/` tree must not drift.
- `smoke/test_architecture_smoke.py` — a rename or a moved data path must not
  pass the suite while leaving the pipeline unrunnable.

`rag2/tests/` is the reproduction's own suite and stays with it.
