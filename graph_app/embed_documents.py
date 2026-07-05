"""
векторный индекс в neo4j
"""
import time
import pandas as pd
from config import get_driver, DATA_DIR, EMBED_MODEL, EMBED_DIM

BATCH = 256

def main():
    idx = pd.read_csv(DATA_DIR / "clean_text_corpus_index.csv", dtype=str, keep_default_na=False)
    docs = pd.read_csv(DATA_DIR / "nodes_documents.csv", dtype=str, keep_default_na=False)
    df = docs[["document_id", "text_id"]].merge(
        idx[["text_id", "preview_clean", "file_name"]], on="text_id", how="left")
    df["text"] = (df["file_name"].fillna("") + ". " + df["preview_clean"].fillna("")).str.strip()
    df = df[df["text"].str.len() > 3].reset_index(drop=True)
    print(f"документов на эмбеддинг: {len(df)}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode(
        ["passage: " + t for t in df["text"].tolist()],
        batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    
    assert vecs.shape[1] == EMBED_DIM, f"размерность {vecs.shape[1]} != EMBED_DIM {EMBED_DIM}"

    with get_driver() as driver, driver.session() as session:
        session.run(f"""
            CREATE VECTOR INDEX document_embedding IF NOT EXISTS
            FOR (d:Document) ON (d.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {EMBED_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}""")

        rows = [{"id": did, "emb": v.tolist(), "snip": snip}
                for did, v, snip in zip(df["document_id"], vecs, df["preview_clean"].fillna(""))]
        for i in range(0, len(rows), BATCH):
            session.run(
                """UNWIND $rows AS row
                   MATCH (d:Document {document_id: row.id})
                   CALL db.create.setNodeVectorProperty(d, 'embedding', row.emb)
                   SET d.snippet = row.snip""",
                rows=rows[i:i + BATCH])

        for _ in range(60):
            rec = session.run(
                "SHOW VECTOR INDEXES YIELD name, state, populationPercent "
                "WHERE name = 'document_embedding' RETURN state, populationPercent"
            ).single()
            if rec and rec["state"] == "ONLINE":
                print(f"  online ({rec['populationPercent']}%)")
                break


if __name__ == "__main__":
    main()
