# Installation

## Requirements

- Python 3.10, 3.11, or 3.12
- pip ≥ 23.0

## Basic Install (simulation + baselines only)

```bash
git clone https://github.com/[author-handle]/aairm.git
cd aairm
pip install -e .
```

## Development Install (recommended)

```bash
pip install -e ".[dev]"
pre-commit install
```

## With LLM support (C4 negotiation agent)

```bash
pip install -e ".[llm]"
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## With RL support (C2 PPO policy)

```bash
pip install -e ".[rl]"
```

## Full install

```bash
pip install -e ".[all]"
```

## Verify installation

```bash
make smoke   # 10-SKU smoke test, < 60 seconds
```
