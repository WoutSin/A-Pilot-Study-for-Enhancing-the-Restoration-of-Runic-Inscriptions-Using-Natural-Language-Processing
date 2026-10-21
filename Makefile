.PHONY: help install run test clean check

PYTHON ?= python3

help:
	@echo "Targets: install, run, test, check, clean"

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) run.py

test:
	$(PYTHON) run.py --test-mode

check:
	$(PYTHON) -m compileall -q run.py runic_restoration

clean:
	rm -f outputs/*.csv outputs/*.txt outputs/*.xlsx
