PYTHON ?= python3
.PHONY: build test run sweep
build:
	$(PYTHON) scripts/build.py
test:
	$(PYTHON) -m unittest discover -s tests -v
run:
	$(PYTHON) run.py
sweep:
	$(PYTHON) scripts/sweep.py
