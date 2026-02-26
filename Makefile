SHELL := /bin/bash

all: train

train:
	@echo "Running training pipeline..."
	python -m src.train_pipeline --dataset data/raw/dataset.csv --out models

lint:
	@echo "Linting not implemented in this skeleton."

help:
 	@echo "Targets: train, lint, help"
