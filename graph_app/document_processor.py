"""
Пайплайн обработки новых документов
"""
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm.auto import tqdm
from config import get_driver, DATA_DIR, EMBED_MODEL, EMBED_DIM
import re

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    import openpyxl
except ImportError:
    openpyxl = None

class DocumentProcessor:
    """Обработчик новых документов"""
    def __init__(self):
        self.upload_dir = DATA_DIR / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_tokens = 400
        self.overlap_tokens = 64
        
    def generate_document_id(self, file_name: str) -> str:
        hash_obj = hashlib.md5(file_name.encode('utf-8'))
        return f"doc_{hash_obj.hexdigest()[:16]}"
    
    def check_document_exists(self, doc_id: str) -> bool:
        with get_driver() as driver, driver.session() as session:
            result = session.run(
                "MATCH (d:Document {document_id: $id}) RETURN d",
                id=doc_id
            ).single()
            return result is not None
    
    def extract_text(self, file_path: Path) -> tuple[str, str]:
        ext = file_path.suffix.lower()
        
        if ext == '.pdf':
            return self._extract_pdf(file_path), 'pdf'
        elif ext in ['.docx', '.docm']:
            return self._extract_docx(file_path), 'docx'
        elif ext in ['.txt', '.md']:
            return file_path.read_text(encoding='utf-8', errors='replace'), 'txt'
        elif ext in ['.xlsx', '.xls']:
            return self._extract_excel(file_path), 'xlsx'
        else:
            raise ValueError(f"Неподдерживаемый формат: {ext}")
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Извлечение текста из PDF"""
        if PdfReader is None:
            raise ImportError()
        
        reader = PdfReader(str(file_path))
        text_parts = []
        for i, page in enumerate(reader.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(f"{i} -\n{page_text}")
            except Exception as e:
                print(f"Ошибка на странице {i}: {e}")
        
        return "\n".join(text_parts)
    
    def _extract_docx(self, file_path: Path) -> str:
        """Извлечение текста из DOCX"""
        if DocxDocument is None:
            raise ImportError()
        
        doc = DocxDocument(str(file_path))
        text_parts = []
        
        for para in doc.paragraphs:
            if para.text and para.text.strip():
                text_parts.append(para.text)
        
        return "\n".join(text_parts)
    
    def _extract_excel(self, file_path: Path) -> str:
        """Извлечение текста из Excel"""
        try:
            xls = pd.ExcelFile(file_path)
            text_parts = []
            
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, dtype=str)
                df = df.dropna(how='all').dropna(axis=1, how='all')
                
                if not df.empty:
                    text_parts.append(f"Стр.: {sheet_name}")
                    for row in df.fillna('').astype(str).itertuples(index=False):
                        values = [v.strip() for v in row]
                        if any(values):
                            text_parts.append("\t".join(values))
            
            return "\n".join(text_parts)
        except Exception as e:
            return f"Ошибка чтения Excel: {e}"
    
    def create_chunks(self, text: str, doc_id: str) -> pd.DataFrame:
        """Создает чанки из текста"""
        from transformers import AutoTokenizer
        
        tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
        PAGE_RE = re.compile(r"^\s*---\s*PAGE\s+(\d+)\s*---\s*$", re.MULTILINE)
        
        def split_pages(text):
            marks = list(PAGE_RE.finditer(text))
            if not marks:
                return [(None, text)]
            
            segments = []
            head = text[:marks[0].start()].strip()
            if head:
                segments.append((None, head))

            for i, m in enumerate(marks):
                page = int(m.group(1))
                start = m.end()
                end = marks[i+1].start() if i+1 < len(marks) else len(text)
                body = text[start:end].strip()
                if body:
                    segments.append((page, body))
            
            return segments
        
        def window_page(page_text, page, chunk_tokens=400, overlap=64):
            enc = tok(page_text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            
            if not ids:
                return
            
            step = chunk_tokens - overlap
            for start in range(0, len(ids), step):
                win = offs[start:start + chunk_tokens]
                if not win:
                    break
                c0, c1 = win[0][0], win[-1][1]
                piece = page_text[c0:c1].strip()
                if piece:
                    yield {"page": page, "n_tokens": len(win), "text": piece}
                if start + chunk_tokens >= len(ids):
                    break
        
        rows, ci = [], 0
        for page, page_text in split_pages(text):
            for ch in window_page(page_text, page):
                rows.append({
                    "text_id": doc_id,
                    "chunk_id": f"{doc_id}_c{ci}",
                    "chunk_index": ci,
                    "page": ch["page"],
                    "n_tokens": ch["n_tokens"],
                    "text": ch["text"],
                })
                ci += 1
        
        return pd.DataFrame(rows)
    
    def create_embeddings(self, df: pd.DataFrame) -> list:
        """Создает эмбеддинги для чанков"""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(EMBED_MODEL)
        texts = ["passage: " + t for t in df["text"].tolist()]
        
        embeddings = model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings
    
    def process_document(self, uploaded_file, doc_name: str = None) -> dict:
        """Пайплайн обработки нового документа"""
        result = {
            "success": False,
            "doc_id": None,
            "message": "",
            "chunks_count": 0,
            "is_duplicate": False
        }
        
        try:
            file_name = uploaded_file.name
            file_path = self.upload_dir / file_name
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            doc_id = self.generate_document_id(file_name)
            doc_name = doc_name or file_name
            
            if self.check_document_exists(doc_id):
                result["is_duplicate"] = True
                result["doc_id"] = doc_id
                result["message"] = f"Документ уже существует: {doc_name}"
                
                with get_driver() as driver, driver.session() as session:
                    info = session.run("""
                        MATCH (d:Document {document_id: $id})
                        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                        RETURN d.file_name as name, count(c) as chunks_count,
                               d.uploaded_at as uploaded_at
                    """, id=doc_id).single()
                    
                    if info:
                        result["message"] += f" (чанков: {info['chunks_count']}, загружен: {info['uploaded_at']})"
                
                return result
            
            text, ext = self.extract_text(file_path)
            if not text or len(text.strip()) < 100:
                return {
                    "success": False,
                    "doc_id": doc_id,
                    "message": "Не удалось извлечь достаточное количество текста",
                    "chunks_count": 0,
                    "is_duplicate": False
                }
            
            chunks_df = self.create_chunks(text, doc_id)
            if chunks_df.empty:
                return {
                    "success": False,
                    "doc_id": doc_id,
                    "message": "Не создано ни одного чанка",
                    "chunks_count": 0,
                    "is_duplicate": False
                }
            embeddings = self.create_embeddings(chunks_df)
            
            self._save_to_neo4j(doc_id, doc_name, file_path, ext, chunks_df, embeddings)
            result["success"] = True
            result["doc_id"] = doc_id
            result["message"] = f"Документ успешно добавлен: {doc_name}"
            result["chunks_count"] = len(chunks_df)
            
        except Exception as e:
            result["message"] = f"Ошибка: {str(e)}"
        
        return result
    
    def _save_to_neo4j(self, doc_id: str, doc_name: str, file_path: Path, 
                       ext: str, chunks_df: pd.DataFrame, embeddings):
        file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
        
        with get_driver() as driver, driver.session() as session:
            session.run("""
                MERGE (d:Document {document_id: $id})
                ON CREATE SET 
                    d.file_name = $name,
                    d.relative_path = $path,
                    d.file_hash = $hash,
                    d.uploaded_at = $uploaded_at,
                    d.extension = $ext,
                    d.quality_flag = 'ok',
                    d.corpus_role = 'text_document',
                    d.language_guess = 'mixed'
                ON MATCH SET
                    d.file_name = $name,
                    d.relative_path = $path,
                    d.uploaded_at = $uploaded_at
            """, {
                "id": doc_id,
                "name": doc_name,
                "path": str(file_path),
                "hash": file_hash,
                "uploaded_at": datetime.now().isoformat(),
                "ext": ext
            })
            
            for i in range(0, len(chunks_df), 100):
                batch = chunks_df.iloc[i:i+100]
                
                for idx, row in batch.iterrows():
                    chunk_id = row["chunk_id"]
                    
                    existing = session.run(
                        "MATCH (c:Chunk {chunk_id: $chunk_id}) RETURN c",
                        chunk_id=chunk_id
                    ).single()
                    
                    if existing:
                        continue
                    
                    session.run("""
                        MATCH (d:Document {document_id: $doc_id})
                        CREATE (c:Chunk {
                            chunk_id: $chunk_id,
                            text: $text,
                            page: $page,
                            index: $index,
                            n_tokens: $n_tokens,
                            embedding: $embedding
                        })
                        CREATE (d)-[:HAS_CHUNK]->(c)
                        CREATE (c)-[:FROM_DOCUMENT]->(d)
                    """, {
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "text": row["text"],
                        "page": int(row["page"]) if pd.notna(row["page"]) else None,
                        "index": int(row["chunk_index"]),
                        "n_tokens": int(row["n_tokens"]),
                        "embedding": embeddings[idx].tolist()
                    })
            chunks_sorted = chunks_df.sort_values("chunk_index")
            
            for i in range(len(chunks_sorted) - 1):
                current = chunks_sorted.iloc[i]
                next_chunk = chunks_sorted.iloc[i + 1]
                
                existing = session.run("""
                    MATCH (c1:Chunk {chunk_id: $current})
                    MATCH (c2:Chunk {chunk_id: $next})
                    OPTIONAL MATCH (c1)-[r:NEXT_CHUNK]->(c2)
                    RETURN r
                """, {
                    "current": current["chunk_id"],
                    "next": next_chunk["chunk_id"]
                }).single()
                
                if not existing:
                    session.run("""
                        MATCH (c1:Chunk {chunk_id: $current})
                        MATCH (c2:Chunk {chunk_id: $next})
                        CREATE (c1)-[:NEXT_CHUNK]->(c2)
                    """, {
                        "current": current["chunk_id"],
                        "next": next_chunk["chunk_id"]
                    })


def get_upload_status(doc_id: str = None):
    """Проверка статуса загруженных документов"""
    with get_driver() as driver, driver.session() as session:
        if doc_id:
            result = session.run("""
                MATCH (d:Document {document_id: $id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                RETURN d.document_id as id, d.file_name as name,
                       count(c) as chunks_count,
                       d.uploaded_at as uploaded_at
            """, id=doc_id)
            return dict(result.single()) if result else None
        else:
            result = session.run("""
                MATCH (d:Document)
                WHERE d.uploaded_at IS NOT NULL
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                RETURN d.document_id as id, d.file_name as name,
                       count(c) as chunks_count,
                       d.uploaded_at as uploaded_at
                ORDER BY d.uploaded_at DESC
            """)
            return [dict(record) for record in result]
