.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys, mkdocs

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python3 -c "$$BROWSER_PYSCRIPT"

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/	
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts	
	rm -fr htmlcov

lint: ## check style with flake8
	flake8 yieldcurveml tests

coverage: ## check code coverage quickly with the default Python	
	coverage report --omit="venv/*,yieldcurveml/tests/*" --show-missing

docs: install ## generate docs		
	@echo "Installing documentation dependencies..."
	./venv/bin/pip install black pdoc 2>/dev/null || echo "Warning: Could not install black/pdoc"
	
	@echo "Formatting code with black..."
	@if command -v ./venv/bin/black >/dev/null 2>&1; then \
		find yieldcurveml -name "*.py" -exec ./venv/bin/black --line-length=80 {} + 2>/dev/null || true; \
	else \
		echo "Warning: black not found, skipping formatting"; \
	fi
	
	@echo "Cleaning code with autopep8..."
	@if command -v ./venv/bin/autopep8 >/dev/null 2>&1; then \
		find yieldcurveml -name "*.py" -exec ./venv/bin/autopep8 --max-line-length=80 --in-place {} + 2>/dev/null || true; \
	else \
		echo "Warning: autopep8 not found, skipping cleaning"; \
	fi
	
	@echo "Generating documentation with pdoc..."
	@if command -v ./venv/bin/pdoc >/dev/null 2>&1; then \
		./venv/bin/pdoc -t docs yieldcurveml --output-dir yieldcurveml-docs 2>/dev/null || echo "Error: pdoc failed"; \
	else \
		echo "Error: pdoc not found"; \
		exit 1; \
	fi
	
	@echo "Cleaning up..."
	find . -name '__pycache__' -exec rm -fr {} + 2>/dev/null || true
	
	@echo "Copying documentation..."
	@if [ -d "yieldcurveml-docs" ]; then \
		cp -rf yieldcurveml-docs/* ../../Pro_Website/Techtonique.github.io/yieldcurveml 2>/dev/null || echo "Warning: Could not copy docs"; \
	else \
		echo "Error: Documentation directory not found"; \
		exit 1; \
	fi
	
	@echo "Documentation generation complete!"

	
servedocs: install ## compile the docs watching for change	 	
	#pip install black pdoc 
	#black yieldcurveml/* --line-length=80	
	#find yieldcurveml/ -name "*.py" -exec autopep8 --max-line-length=80 --in-place {} +
	pdoc -t docs yieldcurveml/* 
	find . -name '__pycache__' -exec rm -fr {} +

release: dist ## package and upload a release
	pip install twine --ignore-installed
	python3 -m twine upload --repository pypi dist/* --verbose

dist: clean ## builds source and wheel package
	python3 setup.py sdist
	python3 setup.py bdist_wheel	
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	uv pip install -e .

build-site: docs ## export mkdocs website to a folder		
	cp -rf yieldcurveml-docs/* ../../Pro_Website/Techtonique.github.io/yieldcurveml
	find . -name '__pycache__' -exec rm -fr {} +

run-custom: ## run all custom examples with one command
	find examples -maxdepth 2 -name "*custom*.py" -exec  python3 {} \;

run-examples: ## run all examples with one command
	find examples -maxdepth 2 -name "*.py" -exec  python3 {} \;

run-lazy: ## run all lazy examples with one command
	find examples -maxdepth 2 -name "lazy*.py" -exec  python3 {} \;

run-conformal: ## run all lazy examples with one command
	find examples -maxdepth 2 -name "*conformal*.py" -exec  python3 {} \;

run-tests: install ## run all the tests with one command
	pip3 install coverage nose2
	python3 -m coverage run -m unittest discover -s yieldcurveml/tests -p "*.py"	

