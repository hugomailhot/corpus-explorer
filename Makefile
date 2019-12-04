.PHONY: clean-test test

clean-test: ## remove test artifacts
	rm -fr .pytest_cache

fix_imports: ## automatically standardize import sections in the project
	poetry run isort --jobs 4 --recursive .

lint: ## run flake8 linter
	poetry run flake8

test: clean-test ## Run tests using pytest
	poetry run pytest

