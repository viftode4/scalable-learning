.PHONY: sync test lint mnist supplement data clean

sync:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .

mnist:
	uv run python notebooks/mnist_fig2.py

supplement:
	bash scripts/extract_supplement.sh

data:
	uv run python scripts/prep_glue.py --task mnli

clean:
	rm -rf results/ slurm-*.out .pytest_cache .ruff_cache
