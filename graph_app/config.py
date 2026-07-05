"""
Настройки подключения к neo4j и пути к данным
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATA_DIR = Path(__file__).parent / "data"

EMBED_MODEL = "intfloat/multilingual-e5-base"
EMBED_DIM = 768

YC_API_KEY = os.getenv("YC_API_KEY")
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID")
YC_BASE_URL = "https://llm.api.cloud.yandex.net/v1"
YC_MODEL = f"gpt://{YC_FOLDER_ID}/yandexgpt/latest" if YC_FOLDER_ID else None
YANDEX_FOLDER_URL=os.getenv("YANDEX_FOLDER_URL")


def get_driver():
    from neo4j import GraphDatabase
    if not NEO4J_URI or not NEO4J_PASSWORD:
        raise SystemExit()
    
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
