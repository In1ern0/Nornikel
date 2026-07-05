"""
векторный вход -> обход к документу -> ответ.
"""
import sys
import argparse
from pathlib import Path
from config import get_driver, YC_API_KEY, YC_BASE_URL, YC_MODEL
from sentence_transformers import SentenceTransformer

from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem
from neo4j_graphrag.llm import OpenAILLM

INDEX = "chunk_embedding"
OVERSAMPLE = 8
RETRIEVAL_QUERY = """
WITH node, score
MATCH (node)-[:FROM_DOCUMENT]->(d:Document)
WHERE ($year IS NULL OR d.relative_path CONTAINS $year OR d.file_name CONTAINS $year)
  AND ($lang IS NULL OR d.language_guess = $lang)
  AND ($category IS NULL OR d.relative_path CONTAINS $category)
  AND ($theme IS NULL OR EXISTS { MATCH (d)-[:HAS_THEME]->(:Theme {theme_key: $theme}) })
RETURN node.text AS text, node.page AS page,
       d.file_name AS source,
       d.document_id AS doc_id,
       d.relative_path AS file_path,
       score
ORDER BY score DESC
LIMIT $limit
"""

def search_params(top_k, year=None, lang=None, category=None, theme=None):
    """query_params для retrieval_query — все ключи обязаны быть, null = без фильтра"""
    return {"year": year, "lang": lang, "category": category,
            "theme": theme, "limit": top_k}


class E5QueryEmbedder:
    def __init__(self, model_name="intfloat/multilingual-e5-base", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)
    
    def embed_query(self, text: str):
        v = self.model.encode(["query: " + text], normalize_embeddings=True)

        return v[0].tolist()

def _fmt(record) -> RetrieverResultItem:
    src = record.get("source") or record.get("doc_id")
    page = record.get("page")
    file_path = record.get("file_path", "")
    cite = f"{src}, стр. {page}" if page is not None else f"{src}"
    
    return RetrieverResultItem(
        content=f"[{cite}]\n{record.get('text')}",
        metadata={
            "source": src, 
            "page": page,
            "doc_id": record.get("doc_id"), 
            "score": record.get("score"),
            "file_path": file_path
        },
    )


def build_retriever(driver):
    embedder = E5QueryEmbedder()
    
    return VectorCypherRetriever(
        driver=driver, 
        index_name=INDEX,
        retrieval_query=RETRIEVAL_QUERY, 
        embedder=embedder,
        result_formatter=_fmt,
    )


def build_llm():
    if not YC_API_KEY or not YC_MODEL:
        raise SystemExit()

    return OpenAILLM(
        model_name=YC_MODEL, 
        api_key=YC_API_KEY, 
        base_url=YC_BASE_URL,
        model_params={"temperature": 0.2, "max_tokens": 800},
    )

PROMPT = """Ты — ассистент по научно-технической базе (металлургия, рынок металлов).
Отвечай на РУССКОМ, опираясь ТОЛЬКО на контекст ниже. Если данных нет — так и скажи.
После каждого утверждения указывай источник в скобках в формате [файл, стр. N] — они есть в контексте.

Контекст:
{context}

Вопрос: {query_text}

Ответ с цитатами:"""


def retrieval_only(query: str, k: int = 5, year=None, lang=None):
    with get_driver() as driver:
        r = build_retriever(driver)
        res = r.search(query_text=query, top_k=k * OVERSAMPLE,
                       query_params=search_params(k, year=year, lang=lang))
        print(f"\n=== топ-{k} чанков по запросу: {query!r} ===")
        for it in res.items:
            m = it.metadata or {}
            print(f"\n[score {m.get('score'):.3f}] {m.get('source')} стр.{m.get('page')}")
            print("  " + it.content.split(chr(10), 1)[-1][:280].replace("\n", " "))
            if m.get('file_path'):
                print(f"{m.get('file_path')}")

def ask(query: str, k: int = 6):
    from neo4j_graphrag.generation import GraphRAG
    from neo4j_graphrag.generation.prompts import RagTemplate
    with get_driver() as driver:
        rag = GraphRAG(
            retriever=build_retriever(driver), 
            llm=build_llm(),
            prompt_template=RagTemplate(
                template=PROMPT,
                expected_inputs=["context", "query_text"]
            )
        )
        resp = rag.search(
            query_text=query,
            retriever_config={"top_k": k * OVERSAMPLE,
                              "query_params": search_params(k)},
            return_context=True
        )
        print(f"\n=== вопрос: {query} ===\n")
        print(resp.answer)
        seen = set()
        for it in resp.retriever_result.items:
            m = it.metadata or {}
            key = (m.get("source"), m.get("page"))
            if key not in seen:
                seen.add(key)
                print(f"  {m.get('source')}, стр. {m.get('page')}  (score {m.get('score'):.3f})")
                if m.get('file_path'):
                    print(f"{m.get('file_path')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--year", default=None, help="фильтр по году (подстрока в пути/имени)")
    ap.add_argument("--lang", default=None, help="фильтр по языку: ru / en / mixed")
    a = ap.parse_args()
    q = " ".join(a.query)
    (retrieval_only(q, a.k, a.year, a.lang) if a.retrieval_only else ask(q, a.k))
