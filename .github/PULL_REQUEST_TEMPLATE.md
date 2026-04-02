## Summary

<!-- One paragraph describing what this PR does. -->

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactoring (no behaviour change)
- [ ] Documentation update
- [ ] New experiment / ablation
- [ ] Dependency update

## Checklist

- [ ] Tests added for all new functionality
- [ ] `make lint` passes (ruff + black)
- [ ] `make typecheck` passes (mypy)
- [ ] `make test-fast` passes (unit + smoke, ≥ 80% coverage)
- [ ] Docstrings follow Google style
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] If a new agent is added: `BaseAgent` inherited; `run()` implemented; unit test exists
- [ ] If a new metric is added: expected paper value in docstring; regression test added
- [ ] If a config parameter is added: documented in `configs/default.yaml`

## Paper Consistency

<!-- If this PR changes simulation behaviour, confirm results still match paper values. -->
- [ ] `make run-paper-experiment --fast` passes
- [ ] Results within ±0.5% of paper Table 2 values (or change is intentional and documented)

## Related Issues

Closes #
