import os

DB_DSN = os.getenv("DB_DSN", "postgresql://nba:nba@localhost:5432/nba")
# SQLAlchemy 2.1 prefers psycopg 3 for the bare PostgreSQL scheme, while this
# project installs psycopg2-binary. Pin the declared driver when no driver was
# explicitly selected in the DSN.
if DB_DSN.startswith("postgresql://"):
    DB_DSN = DB_DSN.replace("postgresql://", "postgresql+psycopg2://", 1)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:2b")
