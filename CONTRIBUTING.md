# Contributing to AAIRM

Thank you for contributing to AAIRM. This guide covers everything from
setting up your development environment to submitting a pull request.

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/YOUR-HANDLE/aairm.git
cd aairm

# 2. Create a branch
git checkout -b feature/your-feature-name

# 3. Install dev environment
pip install -e ".[dev]"
pre-commit install

# 4. Verify everything works
make smoke   # must pass in < 60 seconds
make test-fast
```

## Branch Naming

| Type | Pattern | Example |
|---|---|---|
| Feature | `feature/description` | `feature/tft-forecaster` |
| Bug fix | `fix/description` | `fix/rop-edge-case` |
| Experiment | `experiment/description` | `experiment/m5-ablation` |
| Docs | `docs/description` | `docs/quickstart-update` |

## Commit Messages (Conventional Commits)

```
feat: add TFT forecaster for C1 agent
fix: correct ROP boundary condition for zero lead time
docs: update quickstart notebook with 18-step walkthrough
test: add regression test for Eq. 6 supplier score
chore: bump numpy to 1.26.4
```

## Pull Request Checklist

Before opening a PR, confirm all items:

- [ ] `make lint` passes — zero ruff warnings, black-formatted
- [ ] `make typecheck` passes — zero mypy errors
- [ ] `make test-fast` passes with ≥ 80% overall coverage
- [ ] New functionality has unit tests
- [ ] Docstrings follow Google style (Args/Returns/Raises/References/Examples)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] If adding a new agent: `BaseAgent` inherited; `run()` implemented;
      `_log_start` / `_log_end` called; unit test in `tests/unit/`
- [ ] If adding a new metric: expected paper value in docstring;
      regression test in `tests/unit/test_metrics.py`
- [ ] If changing simulation behaviour: `make run-paper-experiment --fast`
      passes without assertion errors

## How to Add a New Agent

1. Create `aairm/agents/<layer>/<agent_name>.py`.
2. Inherit from `BaseAgent`; implement `run(state: AgentState) -> AgentState`.
3. Call `self._log_start(state)` at entry and `self._log_end(state, t0)` at exit.
4. Add the agent to `aairm/agents/<layer>/__init__.py`.
5. Inject the agent in `MetaOrchestrator.__init__` and call it in `run_cycle`.
6. Write a unit test in `tests/unit/test_<agent_name>.py`.

## How to Add a New Dataset Adapter

1. Create `aairm/data/adapters/<dataset>_adapter.py`.
2. Implement a `load() -> dict` method returning the unified schema.
3. Add to `aairm/data/adapters/__init__.py`.
4. Add a download entry in `scripts/download_datasets.py`.
5. Add a preprocessing call in `scripts/preprocess_all.py`.
6. Document the dataset in `docs/datasets.md`.

## Code Style

- Line length: 100 characters (black + ruff enforce this).
- Type hints on all public functions.
- Google-style docstrings on all public functions, classes, and methods.
- No `print()` in library code — use `structlog.get_logger()`.
- No `assert` in library code — raise explicit exceptions.

## Questions

Open a GitHub Discussion or email mnaqash@iu.edu.sa.
