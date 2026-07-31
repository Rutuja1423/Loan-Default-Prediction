.PHONY: help install clean train test lint format predict

help:
	@echo "Available commands:"
	@echo "  make install    : Install required dependencies"
	@echo "  make train      : Run full machine learning pipeline"
	@echo "  make test       : Run test suite with pytest"
	@echo "  make lint       : Run ruff linting check"
	@echo "  make format     : Format code with black and ruff"
	@echo "  make clean      : Clean temporary python & model cache"

install:
	pip install -r requirements.txt

train:
	python main.py

test:
	pytest tests/ -v

lint:
	ruff check .

format:
	black .
	isort .
	ruff check --fix .

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
