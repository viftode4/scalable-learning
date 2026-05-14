.PHONY: sync test lint check mnist mnist-smoke supplement install-supplement supplement-smoke supplement-smoke-all data clean

MODE ?= rolora
SUPPLEMENT_ZIP ?=

sync:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .

check: test lint

mnist:
	uv run python notebooks/mnist_fig2.py

mnist-smoke:
	uv run python notebooks/mnist_fig2.py --rounds 15 --clients 5 --rank 1 --local-steps 10 --subset 5000 --out results/mnist_fig2_smoke.png

supplement:
	@if [ -n "$(SUPPLEMENT_ZIP)" ]; then \
		bash scripts/extract_supplement.sh "$(SUPPLEMENT_ZIP)"; \
	else \
		bash scripts/extract_supplement.sh; \
	fi

install-supplement:
	bash scripts/install_supplement.sh

supplement-smoke:
	bash scripts/smoke_supplement.sh $(MODE)

supplement-smoke-all:
	bash scripts/smoke_supplement.sh rolora lora ffa_lora

data:
	uv run python scripts/prep_glue.py --task mnli

clean:
	rm -rf slurm-*.out .pytest_cache .ruff_cache exp *.ckpt
	find notebooks scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf results/*
	touch results/.gitkeep
