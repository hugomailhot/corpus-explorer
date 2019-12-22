.PHONY: clean-test download-dataset fix_imports lint test

clean-test: ## remove test artifacts
	rm -fr .pytest_cache

download-dataset: ## download 20-newsgroup dataset in data folder
	poetry run python scripts/download_newsgroup_dataset.py

fix_imports: ## automatically standardize import sections in the project
	poetry run isort --jobs 4 --recursive .

lint: ## run flake8 linter
	poetry run flake8

test: clean-test ## Run tests using pytest
	poetry run pytest

