.PHONY: clean-test dataset-newsgroup dataset-ap fix_imports lint test

clean-test: ## remove test artifacts
	rm -fr .pytest_cache

dataset-newsgroup: ## download 20-newsgroup dataset in data folder
	poetry run python scripts/download_newsgroup_dataset.py

dataset-ap: ## download Associated Press dataset in data folder
	poetry run python scripts/download_associated_press_dataset.py

fix_imports: ## automatically standardize import sections in the project
	poetry run isort --jobs 4 --recursive .

lint: ## run flake8 linter
	poetry run flake8

test: clean-test ## Run tests using pytest
	poetry run pytest

