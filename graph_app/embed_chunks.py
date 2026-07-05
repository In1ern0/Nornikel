"""
эмбеддинг чанков -> векторный индекс
"""
import time
import pandas as pd
from config import get_driver, DATA_DIR, EMBED_MODEL, EMBED_DIM

PARQUET = DATA_DIR / "chunks.parquet"
MIN_TOKENS = 20
ENCODE_BATCH = 64
STREAM = 1024

def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"

    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    df = pd.read_parquet(PARQUET, columns=["chunk_id", "text", "n_tokens"])
    before = len(df)
    df = df[df["n_tokens"] >= MIN_TOKENS].reset_index(drop=True)

    with get_driver() as d, d.session() as s:
        done = set(r["id"] for r in s.run(
            "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN c.chunk_id AS id"))
    if done:
        df = df[~df["chunk_id"].isin(done)].reset_index(drop=True)

    if df.empty:
        return

    from sentence_transformers import SentenceTransformer
    dev = pick_device()
    model = SentenceTransformer(EMBED_MODEL, device=dev)

    ids = df["chunk_id"].tolist()
    texts = df["text"].tolist()

    with get_driver() as driver, driver.session() as session:
        session.run(f"""
            CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {EMBED_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}""")

        t0, done = time.time(), 0
        for i in range(0, len(ids), STREAM):
            blk_ids = ids[i:i + STREAM]
            blk_txt = ["passage: " + t for t in texts[i:i + STREAM]]
            vecs = model.encode(blk_txt, batch_size=ENCODE_BATCH,
                                normalize_embeddings=True, show_progress_bar=False)
            assert vecs.shape[1] == EMBED_DIM, f"размерность {vecs.shape[1]} != {EMBED_DIM}"
            rows = [{"id": c, "emb": v.tolist()} for c, v in zip(blk_ids, vecs)]
            session.run(
                """UNWIND $rows AS row
                   MATCH (c:Chunk {chunk_id: row.id})
                   CALL db.create.setNodeVectorProperty(c, 'embedding', row.emb)""",
                rows=rows)
            done += len(rows)
            rate = done / (time.time() - t0)
            eta = (len(ids) - done) / rate / 60 if rate else 0
            print(f"  {Готово:,}/{len(ids):,}  ({rate:.0f} sent/s, ~{eta:.0f} мин осталось)", flush=True)

        for _ in range(120):
            rec = session.run(
                "SHOW VECTOR INDEXES YIELD name, state, populationPercent "
                "WHERE name = 'chunk_embedding' RETURN state, populationPercent").single()
            if rec and rec["state"] == "ONLINE":
                break


if __name__ == "__main__":
    main()
