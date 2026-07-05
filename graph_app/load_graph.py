"""
Создание графа
"""
import pandas as pd
from config import get_driver, DATA_DIR

BATCH = 1000

def run_write(session, query, rows):
    for i in range(0, len(rows), BATCH):
        session.run(query, rows=rows[i:i + BATCH])

def load(session):
    for label, key in [("Document", "document_id"), ("Theme", "theme_id"), ("Term", "term_id")]:
        session.run(
            f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
        )

    docs = pd.read_csv(DATA_DIR / "nodes_documents.csv", dtype=str, keep_default_na=False)
    run_write(session,
        "UNWIND $rows AS row MERGE (d:Document {document_id: row.document_id}) SET d += row",
        docs.to_dict("records"))

    themes = pd.read_csv(DATA_DIR / "nodes_themes.csv", dtype=str, keep_default_na=False)
    run_write(session,
        "UNWIND $rows AS row MERGE (t:Theme {theme_id: row.theme_id}) SET t += row",
        themes.to_dict("records"))

    terms = pd.read_csv(DATA_DIR / "nodes_terms.csv", dtype=str, keep_default_na=False)
    run_write(session,
        "UNWIND $rows AS row MERGE (t:Term {term_id: row.term_id}) SET t += row",
        terms.to_dict("records"))

    e_theme = pd.read_csv(DATA_DIR / "edges_document_theme.csv", dtype=str, keep_default_na=False)
    run_write(session,
        """UNWIND $rows AS row
           MATCH (d:Document {document_id: row.document_id})
           MATCH (t:Theme {theme_id: row.theme_id})
           MERGE (d)-[r:HAS_THEME]->(t)
           SET r.hit_count = toInteger(row.hit_count)""",
        e_theme.to_dict("records"))

    e_term = pd.read_csv(DATA_DIR / "edges_document_term.csv", dtype=str, keep_default_na=False)
    run_write(session,
        """UNWIND $rows AS row
           MATCH (d:Document {document_id: row.document_id})
           MATCH (t:Term {term_id: row.term_id})
           MERGE (d)-[r:HAS_TERM]->(t)
           SET r.term_rank = toInteger(row.term_rank)""",
        e_term.to_dict("records"))


if __name__ == "__main__":
    with get_driver() as driver:
        with driver.session() as session:
            print("гружу граф в aura...")
            load(session)
    print("Готово")
