"""
Загрузка чанков из chunks.parquet в neo4j
"""
import pandas as pd
from config import get_driver, DATA_DIR
import sys

PARQUET = DATA_DIR / "chunks.parquet"
BATCH = 5000


def main():
    if not PARQUET.exists():
        print(f"Файл {PARQUET} не найден!")
        return
    
    df = None
    engines = ['pyarrow', 'fastparquet']
    for engine in engines:
        try:
            df = pd.read_parquet(PARQUET, engine=engine)
            break
        except Exception as e:
            print(f"{engine} не работает: {e}")
    
    if df is None:
        csv_file = DATA_DIR / "chunks.csv"
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Ошибка загрузки CSV: {e}")
                return
        else:
            return

    required_cols = ["text_id", "chunk_id", "text", "chunk_index", "n_tokens"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return
    
    if "page" not in df.columns:
        df["page"] = None
    
    df = df.sort_values(["text_id", "chunk_index"]).reset_index(drop=True)
    df["prev_chunk_id"] = df.groupby("text_id")["chunk_id"].shift(1)
    
    print(f"Чанков к заливке: {len(df):,} из {df.text_id.nunique()} документов")
    
    def page_val(p):
        return None if pd.isna(p) else int(p)
    
    node_rows = [
        {"text_id": r.text_id, "chunk_id": r.chunk_id, "text": r.text,
         "page": page_val(r.page), "index": int(r.chunk_index), "n_tokens": int(r.n_tokens)}
        for r in df.itertuples(index=False)
    ]
    next_rows = [
        {"prev": r.prev_chunk_id, "chunk_id": r.chunk_id}
        for r in df.itertuples(index=False) if not pd.isna(r.prev_chunk_id)
    ]
    
    print(f"Связей NEXT_CHUNK: {len(next_rows):,}")
    
    try:
        with get_driver() as driver, driver.session() as session:
            try:
                session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS "
                            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
                session.run("CREATE CONSTRAINT document_id IF NOT EXISTS "
                            "FOR (d:Document) REQUIRE d.document_id IS UNIQUE")
            except Exception as e:
                print(f"Констрейнты уже существуют: {e}")
            
            total_nodes = len(node_rows)
            for i in range(0, total_nodes, BATCH):
                batch = node_rows[i:i + BATCH]
                try:
                    session.run(
                        """UNWIND $rows AS row
                           MATCH (d:Document {document_id: row.text_id})
                           MERGE (c:Chunk {chunk_id: row.chunk_id})
                           SET c.text = row.text, c.page = row.page,
                               c.index = row.index, c.n_tokens = row.n_tokens
                           MERGE (d)-[:HAS_CHUNK]->(c)
                           MERGE (c)-[:FROM_DOCUMENT]->(d)""",
                        rows=batch)
                    print(f"Ноды {min(i + BATCH, total_nodes):,}/{total_nodes:,}")
                except Exception as e:
                    continue
            
            if next_rows:
                total_next = len(next_rows)
                for i in range(0, total_next, BATCH):
                    batch = next_rows[i:i + BATCH]
                    try:
                        session.run(
                            """UNWIND $rows AS row
                               MATCH (p:Chunk {chunk_id: row.prev})
                               MATCH (c:Chunk {chunk_id: row.chunk_id})
                               MERGE (p)-[:NEXT_CHUNK]->(c)""",
                            rows=batch)
                    except Exception as e:
                        continue
    
            print(f"\nЗагрузка завершена!")

            
    except Exception as e:
        print(f"Ошибка при загрузке в Neo4j: {e}")


if __name__ == "__main__":
    main()