.PHONY: all install install-dev test test-cov lint format typecheck docs clean help

PYTHON   := python
PYTEST   := pytest
SRC      := src
TESTS    := tests
DOCS_DIR := docs

# Default target
all: help

## install     Install package in editable mode (runtime deps only)
install:
	$(PYTHON) -m pip install -e .

## install-dev Install package + all dev/notebook/docs extras
install-dev:
	$(PYTHON) -m pip install -e ".[all]"

## install-nb  Install notebook extras only
install-nb:
	$(PYTHON) -m pip install -e ".[notebook]"

## test         Run the test suite
test:
	$(PYTEST) $(TESTS)

## test-cov     Run tests with coverage report
test-cov:
	$(PYTEST) $(TESTS) --cov=$(SRC) --cov-report=term-missing --cov-report=html

## test-fast    Run tests, skip slow/gpu markers
test-fast:
	$(PYTEST) $(TESTS) -m "not slow and not gpu and not multi_device"

## lint         Check code style with ruff
lint:
	$(PYTHON) -m ruff check $(SRC) $(TESTS)

## format       Auto-format with black + isort
format:
	$(PYTHON) -m black $(SRC) $(TESTS)
	$(PYTHON) -m isort $(SRC) $(TESTS)

## format-check Check formatting without making changes (CI use)
format-check:
	$(PYTHON) -m black --check $(SRC) $(TESTS)
	$(PYTHON) -m isort --check-only $(SRC) $(TESTS)

## typecheck    Run mypy static type checking
typecheck:
	$(PYTHON) -m mypy $(SRC)

## docs         Build HTML documentation with Sphinx
docs:
	sphinx-build -b html $(DOCS_DIR)/source $(DOCS_DIR)/_build/html

## docs-serve   Build and open docs in browser
docs-serve: docs
	open $(DOCS_DIR)/_build/html/index.html

## clean        Remove build artefacts and caches
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage
	rm -rf $(DOCS_DIR)/_build

## help         Show this help message
help:
	@echo ""
	@echo "JAX-NSL – available make targets:"
	@echo ""
	@grep -E '^## [a-z]' Makefile | sed 's/## /  make /g'
	@echo ""
