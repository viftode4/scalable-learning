.PHONY: sync test lint check mnist mnist-paper mnist-smoke local-smoke full-local table1-pilot table1-pilot-all supplement install-supplement supplement-smoke supplement-smoke-all data clean

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

mnist-paper:
	uv run python notebooks/mnist_fig2.py --rounds 200 --clients 5 --rank 1 --local-steps 20 --out results/mnist_fig2.png

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

table1-pilot:
	CONFIG=experiments/configs/table1_local_pilot.yaml LOG_PREFIX=table1_pilot bash scripts/smoke_supplement.sh $(MODE)

table1-pilot-all:
	CONFIG=experiments/configs/table1_local_pilot.yaml LOG_PREFIX=table1_pilot bash scripts/smoke_supplement.sh rolora lora ffa_lora

local-smoke:
	uv run python scripts/local_suite.py smoke

full-local:
	uv run python scripts/local_suite.py full-local

data:
	uv run python scripts/prep_glue.py --task mnli

clean:
	rm -rf slurm-*.out .pytest_cache .ruff_cache exp *.ckpt
	find notebooks scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf results/*
	touch results/.gitkeep
