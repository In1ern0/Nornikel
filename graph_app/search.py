"""
семантический поиск по документам.
"""
import sys
from config import get_driver, EMBED_MODEL

TOP_K = 5
CYPHER = """
CALL db.index.vector.queryNodes('document_embedding', $k, $q)
YIELD node AS d, score
OPTIONAL MATCH (d)-[:HAS_THEME]->(t:Theme)
WITH d, score, collect(t.theme_name) AS themes
RETURN d.file_name AS file, d.language_guess AS lang, d.snippet AS snippet,
       score, themes
ORDER BY score DESC
"""


def search(query: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    q = model.encode("query: " + query, normalize_embeddings=True).tolist()

    with get_driver() as driver, driver.session() as session:
        rows = session.run(CYPHER, k=TOP_K, q=q).data()

    print(f"\nзапрос: {query}\n" + "=" * 70)
    for i, r in enumerate(rows, 1):
        themes = ", ".join(t for t in r["themes"] if t)
        snippet = (r.get("snippet") or "").replace("\n", " ")[:160]
        print(f"{i}. [{r['score']:.3f}] ({r['lang']}) {r['file']}")
        print(f"   темы: {themes}")
        print(f" {snippet}…\n")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "распределение платиновых металлов между штейном и шлаком"
    search(q)
