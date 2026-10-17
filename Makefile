.PHONY: all test descriptive experiments figures annotation
all: test descriptive experiments figures
test:
	python -m pytest -q tests
descriptive:
	python scripts/run_descriptive.py
experiments:
	python scripts/run_experiments.py --repeats 5
figures:
	python scripts/make_figures.py
annotation:
	python scripts/make_annotation_sheet.py
