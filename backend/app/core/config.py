from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    CHROMA_DB_PATH: str = "/app/data/chroma"
    SQLITE_DB_PATH: str = "/app/data/metadata.db"
    
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    
    DENSE_WEIGHT: float = 0.6
    SPARSE_WEIGHT: float = 0.4
    
    TOP_K: int = 20
    RERANK_TOP_K: int = 5
    
    MIN_RELEVANCE_THRESHOLD: float = 0.3

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
