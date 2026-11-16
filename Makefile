.PHONY: install install-torch data distill train train-distilled eval fuse serve smoke cpu-demo lint test

install:
	pip install -e ".[mlx,dev]"

install-torch:
	pip install -e ".[torch,dev]"

data:
	python -m peft_engine.dataset --sample-size 5000

distill:
	python -m peft_engine.distill --num-prompts 1000

train:
	caffeinate -i python -m peft_engine.trainer

train-distilled:
	caffeinate -i python -m peft_engine.trainer --data-dir data/distilled --adapter-dir models/adapters/distilled

eval:
	python -m peft_engine.evaluator

fuse:
	python -m peft_engine.fuser

serve:
	python -m peft_engine.server

smoke:
	python -m peft_engine.dataset --sample-size 200
	python -m peft_engine.trainer --iters 20

cpu-demo:
	bash scripts/run_cpu_demo.sh

lint:
	ruff check . && ruff format --check . && mypy src

test:
	pytest
