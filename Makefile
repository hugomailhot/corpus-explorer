.PHONY: clean-test test

clean-test: ## remove test artifacts
	rm -fr .pytest_cache

test: clean-test ## Run tests using pytest
	poetry run pytest
