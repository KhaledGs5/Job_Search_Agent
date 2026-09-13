from pathlib import Path
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # NVIDIA NIM / OpenAI-compatible API
    nvidia_api_key: str = Field(..., description="NVIDIA API key")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="NVIDIA OpenAI-compatible API base URL",
    )
    nvidia_model: str = Field(
        default="deepseek-ai/deepseek-v4-pro-0813",
        description="NVIDIA model name",
    )

    # Optional external APIs
    rapidapi_key: str = Field(default="", description="RapidAPI key for JSearch")
    linkedin_email: str = Field(default="")
    linkedin_password: str = Field(default="")

    # Paths (relative to project root)
    data_dir: Path = Field(default=BASE_DIR / "data")
    db_path: Path = Field(default=BASE_DIR / "data" / "job_search.db")
    chroma_dir: Path = Field(default=BASE_DIR / "data" / "chroma")
    resumes_dir: Path = Field(default=BASE_DIR / "data" / "resumes")
    applications_dir: Path = Field(default=BASE_DIR / "data" / "applications")
    profile_path: Path = Field(default=BASE_DIR / "data" / "candidate_profile.json")

    # Search settings
    max_jobs_per_source: int = Field(default=25)
    relevance_threshold: float = Field(default=0.6)

    # Embedding model (local, no API cost)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    def ensure_dirs(self) -> None:
        for d in [
            self.data_dir,
            self.chroma_dir,
            self.resumes_dir,
            self.applications_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
