"""
поиск по научному корпусу с цитатами
"""
import streamlit as st
import pandas as pd
import time
from urllib.parse import quote
from collections import defaultdict
import re

from config import get_driver, YC_API_KEY, YANDEX_FOLDER_URL
from graphrag_search import build_retriever, build_llm, PROMPT, search_params, OVERSAMPLE
from document_processor import DocumentProcessor, get_upload_status

st.set_page_config(page_title="Научный клубок — GraphRAG", page_icon="🧶", layout="wide")

def extract_yandex_path(file_path: str) -> str:
    """путь внутри Я.диска оргов из полного пути файла"""
    if not file_path:
        return ""
    path = file_path.replace("Задача 2. Научный клубок\\", "")
    path = path.replace("\\", "/")
    return quote(path)


def yandex_url_for(file_path: str) -> str | None:
    if not (file_path and YANDEX_FOLDER_URL):
        return None
    p = extract_yandex_path(file_path)
    return f"{YANDEX_FOLDER_URL}/{p}" if p else None


def clean_chunk_text(text: str) -> str:
    """Очищает текст чанка от излишних пробелов и форматирует его"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    return text.strip()


@st.cache_resource(show_spinner="поднимаю драйвер")
def get_retriever():
    driver = get_driver()
    return build_retriever(driver)


@st.cache_resource(show_spinner="подключаю yandexgpt...")
def get_rag():
    from neo4j_graphrag.generation import GraphRAG
    from neo4j_graphrag.generation.prompts import RagTemplate
    return GraphRAG(
        retriever=get_retriever(), llm=build_llm(),
        prompt_template=RagTemplate(template=PROMPT,
                                    expected_inputs=["context", "query_text"]))


@st.cache_data(ttl=120, show_spinner=False)
def corpus_stats():
    with get_driver() as d, d.session() as s:
        docs = s.run("MATCH (d:Document) RETURN count(d) AS c").single()["c"]
        chunks = s.run("MATCH (c:Chunk) RETURN count(c) AS c").single()["c"]
        embedded = s.run("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL "
                         "RETURN count(c) AS c").single()["c"]
    return docs, chunks, embedded

LANG_OPTIONS = {"Русский": "ru", "Английский": "en", "Смешанный": "mixed"}
TYPE_OPTIONS = {"Журналы": "Журналы", "Материалы конференций": "конференци",
                "Обзоры": "Обзоры", "Статьи": "Статьи", "Доклады": "Доклады"}
THEME_RU = {"Mining": "Горное дело", "Market and economics": "Рынок и экономика",
            "Metallurgy": "Металлургия", "Copper": "Медь", "Beneficiation": "Обогащение",
            "Nickel": "Никель", "Ecology": "Экология",
            "Science and education": "Наука и образование",
            "Platinum group metals": "Платиноиды"}


@st.cache_data(ttl=3600, show_spinner=False)
def theme_options():
    with get_driver() as d, d.session() as s:
        recs = s.run("MATCH (t:Theme) RETURN t.theme_key AS k, t.theme_name AS name "
                     "ORDER BY toInteger(t.documents_count) DESC")

        return {THEME_RU.get(r["name"], r["name"]): r["k"] for r in recs}


st.title("Научный клубок")

with st.sidebar:
    try:
        docs, chunks, embedded = corpus_stats()
        st.metric("Всего документов", f"{docs:,}", border=True)
        st.metric("Всего чанков", f"{chunks:,}", border=True)
    except Exception as e:
        st.error(f"нет связи с neo4j: {e}", icon=":material/database_off:")

    st.subheader("Поиск")
    top_k = st.slider("Чанков в контекст", 3, 12, 6)
    llm_on = st.toggle("Генерировать ответ (YandexGPT)",
                       value=bool(YC_API_KEY) and st.query_params.get("llm") != "0",
                       help="выключи — покажу только найденные фрагменты, без llm")
    st.divider()

    with st.expander("Добавить документ", icon=":material/upload_file:"):
        with st.form("upload_document", clear_on_submit=True, border=False):
            uploaded_file = st.file_uploader(
                "Выберите файл",
                type=['pdf', 'docx', 'txt', 'xlsx', 'xls'],
                help="Поддерживаются: PDF, DOCX, TXT, XLSX, XLS"
            )
            doc_name = st.text_input(
                "Название документа (опционально)",
                placeholder="По умолчанию — имя файла"
            )
            submit_btn = st.form_submit_button("Добавить", type="primary",
                                               icon=":material/upload:", width="stretch")

            if submit_btn and uploaded_file is not None:
                with st.spinner("Обработка документа..."):
                    processor = DocumentProcessor()
                    result = processor.process_document(uploaded_file, doc_name)

                    if result.get("is_duplicate", False):
                        st.warning(result['message'], icon=":material/content_copy:")
                    elif result["success"]:
                        st.success(f"{result['message']} · чанков: {result['chunks_count']}",
                                   icon=":material/check_circle:")
                        st.cache_data.clear()
                        time.sleep(1)
                    else:
                        st.error(result['message'], icon=":material/error:")
            elif submit_btn:
                st.warning("Сначала выберите файл", icon=":material/warning:")

        try:
            docs_list = get_upload_status()
            if docs_list:
                st.caption("Загруженные документы")
                df = pd.DataFrame(docs_list)
                st.dataframe(
                    df[["name", "chunks_count", "uploaded_at"]],
                    width="stretch",
                    column_config={
                        "name": "Название",
                        "chunks_count": "Чанков",
                        "uploaded_at": "Дата загрузки"
                    }
                )
        except Exception as e:
            st.error(f"Ошибка загрузки списка: {e}")

try:
    themes = theme_options()
except Exception:
    themes = {}

with st.form("search", border=False):
    with st.container(horizontal=True, vertical_alignment="bottom"):
        query = st.text_input(
            label="Введите запрос",
            placeholder="При каких температурах и давлении ведут автоклавное выщелачивание сульфидных концентратов?",
            width="stretch")
        go = st.form_submit_button("Искать", type="primary", icon=":material/search:")
    with st.container(horizontal=True):
        year_filter = st.selectbox("Год издания",
                                   ["Все"] + [str(y) for y in range(2026, 1999, -1)],
                                   width="stretch")
        theme_filter = st.selectbox("Тема", ["Все"] + list(themes), width="stretch")
        lang_filter = st.selectbox("Язык", ["Все"] + list(LANG_OPTIONS), width="stretch")
        type_filter = st.selectbox("Тип документа", ["Все"] + list(TYPE_OPTIONS),
                                   width="stretch")

if not go and "q" in st.query_params and "last" not in st.session_state:
    query, go = st.query_params["q"], True

if go and query.strip():
    qp = search_params(
        top_k,
        year=None if year_filter == "Все" else year_filter,
        lang=LANG_OPTIONS.get(lang_filter),
        category=TYPE_OPTIONS.get(type_filter),
        theme=themes.get(theme_filter),
    )
    active = [f"{lbl}: {val}" for lbl, val in
              [("год", year_filter), ("тема", theme_filter),
               ("язык", lang_filter), ("тип", type_filter)] if val != "Все"]
    filters_label = " · ".join(active)

    if llm_on:
        with st.spinner("ищу и формулирую ответ..."):
            rag = get_rag()
            try:
                resp = rag.search(query_text=query,
                                  retriever_config={"top_k": top_k * OVERSAMPLE,
                                                    "query_params": qp},
                                  return_context=True)
                st.session_state["last"] = {"query": query, "answer": resp.answer,
                                            "items": resp.retriever_result.items,
                                            "filters": filters_label}
            except Exception as e:
                res = get_retriever().search(query_text=query, top_k=top_k * OVERSAMPLE,
                                             query_params=qp)
                st.session_state["last"] = {"query": query, "answer": None,
                                            "items": res.items,
                                            "llm_error": str(e)[:200],
                                            "filters": filters_label}
    else:
        with st.spinner("ищу фрагменты..."):
            res = get_retriever().search(query_text=query, top_k=top_k * OVERSAMPLE,
                                         query_params=qp)
        st.session_state["last"] = {"query": query, "answer": None,
                                    "items": res.items, "filters": filters_label}

if "last" in st.session_state:
    last = st.session_state["last"]
    items = last["items"]

    if last.get("llm_error"):
        st.warning(f"llm недоступен, показаны только найденные фрагменты · {last['llm_error']}",
                   icon=":material/cloud_off:")

    if last["answer"] is not None:
        st.subheader("Ответ")
        with st.container(border=True):
            st.caption(f"вопрос: {last['query']}")
            if last.get("filters"):
                st.caption(f"фильтры: {last['filters']}")
            st.markdown(last["answer"])
        st.divider()

    st.subheader("Источники")
    if last.get("filters") and last["answer"] is None:
        st.caption(f"фильтры: {last['filters']}")
    
    if not items:
        st.info("Ничего не найдено. Попробуйте изменить запрос или фильтр.")
        st.stop()
    
    docs_dict = defaultdict(list)
    for it in items:
        m = it.metadata or {}
        doc_key = (m.get("source"), m.get("file_path"))
        docs_dict[doc_key].append({
            "content": it.content,
            "score": m.get("score", 0),
            "page": m.get("page"),
            "chunk_text": it.content.split("\n", 1)[-1] if "\n" in it.content else it.content
        })
    
    sorted_docs = sorted(
        docs_dict.items(),
        key=lambda x: max(c["score"] for c in x[1]),
        reverse=True
    )
    
    for doc_idx, ((doc_name, file_path), chunks_list) in enumerate(sorted_docs):
        max_score = max(c["score"] for c in chunks_list)
        pages = sorted(set(c["page"] for c in chunks_list if c["page"]))
        pages_str = ", ".join(str(p) for p in pages[:3])
        if len(pages) > 3:
            pages_str += f" и еще {len(pages)-3}"
        
        url = yandex_url_for(file_path)
        pg_info = f" · стр. {pages_str}" if pages_str else ""
        
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f":material/description: **{doc_name}**{pg_info}", width="stretch")
                st.badge(f"{max_score:.3f}", color="orange")
                if url:
                    st.link_button("Скачать документ", url, icon=":material/download:")
            if len(chunks_list)%10 == 1:
                label_fragment = f"Показать {len(chunks_list)} фрагмент"
            elif len(chunks_list)%10 > 1 and len(chunks_list)%10 <= 4:
                label_fragment = f"Показать {len(chunks_list)} фрагмента"
            else:
                label_fragment = f"Показать {len(chunks_list)} фрагментов"

            with st.expander(label_fragment, icon=":material/article:"):
                for i, chunk in enumerate(chunks_list, 1):
                    page_info = f"стр. {chunk['page']}" if chunk['page'] else "без страницы"
                    st.caption(f"**Фрагмент {i}** · `{chunk['score']:.3f}`")
                    clean_text = clean_chunk_text(chunk["chunk_text"])
                    st.markdown(
                        f'<div style="background-color: transparent; padding: 16px; border-radius: 8px; '
                        f'border-left: 4px solid #ff8c00; '
                        f'font-family: -apple-system, BlinkMacSystemFont, sans-serif; '
                        f'font-size: 15px; line-height: 1.8; white-space: pre-wrap; '
                        f'word-wrap: break-word; width: 100%;">{clean_text}</div>',
                        unsafe_allow_html=True
                    )
                    if i < len(chunks_list):
                        st.divider()
        
        st.caption("")
