# Development Guide

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
# Install core dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install model dependencies (optional, for running models)
pip install -e ".[models]"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mote_error --cov=models --cov=benchmarks

# Run specific test file
pytest tests/test_metrics.py

# Run specific test
pytest tests/test_metrics.py::TestMOTe::test_mote_perfect_score
```

## Code Quality

```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .

# Type checking
mypy src/mote_error models benchmarks
```

## Project Structure

```
mote_error/
├── src/mote_error/      # Core metrics library
│   ├── layout.py       # MOTe error and components (mask mode)
│   ├── _core_bbox.py   # Analytic bounding-box fast path for MOTe
│   ├── class_metrics.py# Class-level MOTe decomposition
│   ├── visualisation.py# MOTe pixel-state overlays
│   ├── ocr.py          # Character Error Vector (SpACER, CDD)
│   └── dataset.py      # Dataset loaders
├── models/             # Model loading and inference
│   └── loader.py       # Base model class
├── benchmarks/         # Benchmarking framework
│   └── runner.py       # Evaluation orchestration
├── tests/              # Test suite
├── scripts/            # Utility scripts
├── data/               # Dataset storage (gitignored)
└── results/            # Evaluation results (gitignored)
```
