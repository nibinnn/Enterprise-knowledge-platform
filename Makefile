# =============================================================================
# Enterprise Knowledge Intelligence Platform — Makefile
# =============================================================================

.PHONY: help up down build logs shell test lint format migrate

help:   ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Docker ────────────────────────────────────────────────────────────────────

up:     ## Start all services (detached)
	docker compose up -d

down:   ## Stop all services
	docker compose down

build:  ## Rebuild images
	docker compose build --no-cache

logs:   ## Tail logs (all services)
	docker compose logs -f

logs-api:   ## Tail API logs only
	docker compose logs -f api

logs-worker: ## Tail worker logs only
	docker compose logs -f worker

shell:  ## Open a bash shell in the api container
	docker compose exec api bash

worker-shell: ## Open a bash shell in the worker container
	docker compose exec worker bash

ps:     ## List running containers
	docker compose ps

# ── Dev setup ─────────────────────────────────────────────────────────────────

install: ## Install Python dependencies locally
	pip install -r requirements.txt

setup-env: ## Copy .env.example to .env
	@if [ ! -f .env ]; then cp .env.example .env && echo ".env created — fill in your API keys"; \
	else echo ".env already exists"; fi

# ── Database ──────────────────────────────────────────────────────────────────

db-init: ## Run init.sql directly (first-time setup)
	docker compose exec postgres psql -U ekip_user -d ekip -f /docker-entrypoint-initdb.d/init.sql

migrate: ## Run Alembic migrations
	docker compose exec api alembic upgrade head

migrate-gen: ## Generate a new Alembic migration (MSG="description")
	docker compose exec api alembic revision --autogenerate -m "$(MSG)"

migrate-down: ## Roll back one migration step
	docker compose exec api alembic downgrade -1

# ── Testing ───────────────────────────────────────────────────────────────────

test:   ## Run all tests
	pytest tests/ -v --cov=app --cov-report=term-missing

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

# ── Code quality ──────────────────────────────────────────────────────────────

lint:   ## Run ruff + mypy
	ruff check app/ tests/
	mypy app/

format: ## Auto-format with black + isort
	black app/ tests/
	isort app/ tests/

# ── Celery ────────────────────────────────────────────────────────────────────

flower: ## Open Celery Flower monitoring (port 5555)
	docker compose exec worker celery -A app.workers.celery_app flower --port=5555

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean: ## Remove all containers, volumes, and built images
	docker compose down -v --rmi local
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
