import os
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from pathlib import Path
from backend.config import DB_DSN

# game_recaps and injury_notes are text corpora written from the box scores, ingested for RAG retrieval
TABLES = ["game_details", "player_box_scores", "players", "teams", "game_recaps", "injury_notes"]
DATA_DIR = Path(__file__).resolve().parent / "data"

def main():
    print('Starting Database Ingestion')
    eng = sa.create_engine(DB_DSN)
    with eng.begin() as cx:
        # Ensure pgvector extension is available for the `vector` type used to store embeddings
        cx.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        for t in TABLES:
            path = os.path.join(DATA_DIR, f"{t}.csv")
            df = pd.read_csv(path)
            df.to_sql(t, cx, if_exists="replace", index=False, method="multi", chunksize=5000)
    print('Finished Database Ingestion')


if __name__ == "__main__":
    main()
