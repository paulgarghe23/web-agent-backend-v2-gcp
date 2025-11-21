# ==============================================================================
# 🛠️ INSTALLATION & SETUP
# ==============================================================================

install:
	@echo "📦 Installing dependencies..."
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync --dev --extra streamlit

# ==============================================================================
# 🚀 LOCAL DEPLOYMENT (PLAYGROUND)
# ==============================================================================

# Option 1: Start everything together (API + UI)
playground:
	@echo "🚀 Launching the system..."
	@$(MAKE) -j2 api ui

# Option 2: Start only the brain (API Backend)
api:
	@echo "🧠 Starting the brain (Backend)..."
	GOOGLE_CLOUD_PROJECT=web-agent-gcp-project uv run uvicorn app.server:app --host localhost --port 8000 --reload

# Option 3: Start only the face (Frontend UI)
ui:
	@echo "💅 Starting the interface (Streamlit)..."
	uv run streamlit run frontend/streamlit_app.py --browser.serverAddress=localhost --server.enableCORS=false --server.enableXsrfProtection=false

# ==============================================================================
# ☁️ CLOUD DEPLOYMENT (GOOGLE CLOUD)
# ==============================================================================

deploy:
	@echo "☁️ Uploading to cloud..."
	PROJECT_ID=$$(gcloud config get-value project) && \
	gcloud beta run deploy web-agent \
		--source . \
		--memory "4Gi" \
		--project $$PROJECT_ID \
		--region "europe-west1" \
		--allow-unauthenticated \
		--update-build-env-vars "AGENT_VERSION=0.1.0" \
		--set-env-vars "GCS_BUCKET_NAME=web-agent-data-web-agent-gcp-project"

# ==============================================================================
# 🧪 TESTS & QUALITY
# ==============================================================================

test:
	uv run pytest tests/unit

lint:
	uv run ruff check . --diff
