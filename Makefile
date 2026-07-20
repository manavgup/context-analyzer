.DEFAULT_GOAL := help

# ── Variables ────────────────────────────────────────────────────
PROJECT_NAME := context_tracker
VENV_DIR     := .venv
BIN          := $(VENV_DIR)/bin
PYTHON       := $(BIN)/python
UV           := uv

# ── Help ─────────────────────────────────────────────────────────
.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Environment ──────────────────────────────────────────────────
.PHONY: venv install install-dev update clean

venv: ## Create virtual environment
	$(UV) venv $(VENV_DIR)

install: venv ## Install dependencies
	VIRTUAL_ENV=$(VENV_DIR) $(UV) pip install -e .

install-dev: venv ## Install with dev dependencies
	VIRTUAL_ENV=$(VENV_DIR) $(UV) pip install -e ".[dev]"

update: ## Update dependencies
	VIRTUAL_ENV=$(VENV_DIR) $(UV) pip install --upgrade -e ".[dev]"

clean: ## Remove build artifacts and caches
	rm -rf $(VENV_DIR) dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ .mypy_cache/ .ruff_cache/ .pytest_cache/

# ── Quality ──────────────────────────────────────────────────────
.PHONY: lint lint-fix format format-check typecheck

lint: ## Run linter
	$(BIN)/ruff check src/ tests/

lint-fix: ## Run linter with auto-fix
	$(BIN)/ruff check --fix src/ tests/

format: ## Format code
	$(BIN)/ruff format src/ tests/

format-check: ## Check formatting without changes
	$(BIN)/ruff format --check src/ tests/

typecheck: ## Run type checker
	$(BIN)/mypy src/

# ── Testing ──────────────────────────────────────────────────────
.PHONY: test test-unit test-integration coverage

test: ## Run all tests
	$(BIN)/pytest tests/

test-unit: ## Run unit tests only
	$(BIN)/pytest tests/ -m "not integration"

test-integration: ## Run integration tests only
	$(BIN)/pytest tests/ -m integration

coverage: ## Run tests with coverage report
	$(BIN)/pytest tests/ --cov=$(PROJECT_NAME) --cov-report=term-missing --cov-report=html

# ── Server ───────────────────────────────────────────────────────
.PHONY: dev serve stop

dev: ## Start dashboard dev server with reload
	$(BIN)/uvicorn $(PROJECT_NAME).dashboard:create_default_app --factory --reload --host 0.0.0.0 --port 8080

serve: ## Start dashboard server
	$(BIN)/uvicorn $(PROJECT_NAME).dashboard:create_default_app --factory --host 0.0.0.0 --port 8080

stop: ## Stop running server
	@pkill -f "uvicorn $(PROJECT_NAME)" 2>/dev/null || echo "No server running"

# ── Hooks ────────────────────────────────────────────────────────
.PHONY: hook-install hook-uninstall

hook-install: ## Install Claude Code hooks
	$(PYTHON) -m context_tracker.installer install

hook-uninstall: ## Uninstall Claude Code hooks
	$(PYTHON) -m context_tracker.installer uninstall

# ── Database ─────────────────────────────────────────────────────
.PHONY: db-init db-migrate db-upgrade db-downgrade db-reset

db-init: ## Initialize Alembic migrations
	$(BIN)/alembic init alembic

db-migrate: ## Create a new migration (usage: make db-migrate MSG="add users table")
	$(BIN)/alembic revision --autogenerate -m "$(MSG)"

db-upgrade: ## Apply migrations to latest
	$(BIN)/alembic upgrade head

db-downgrade: ## Revert last migration
	$(BIN)/alembic downgrade -1

db-reset: ## Drop and recreate database (DESTRUCTIVE)
	$(BIN)/alembic downgrade base
	$(BIN)/alembic upgrade head

# ── CI/CD ────────────────────────────────────────────────────────
.PHONY: pre-commit verify check-env

pre-commit: ## Run pre-commit hooks on all files
	$(BIN)/pre-commit run --all-files

verify: lint format-check typecheck test ## Run full verification suite
	@echo "\033[32mAll checks passed.\033[0m"

check-env: ## Verify development environment
	@echo "Python: $$($(PYTHON) --version 2>/dev/null || echo 'NOT FOUND')"
	@echo "Venv:   $(VENV_DIR) ($$(test -d $(VENV_DIR) && echo 'exists' || echo 'missing'))"
	@echo "UV:     $$($(UV) --version 2>/dev/null || echo 'NOT FOUND')"
