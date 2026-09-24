"""
Application settings, loaded from environment variables (see .env.example).

Uses pydantic-settings so values are validated and typed once, at startup,
instead of scattering `os.getenv(...)` calls through the codebase.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_env: str = "development"  # development | production | test
    log_level: str = "INFO"
    port: int = 8000

    # --- Model ---
    model_path: str = "../models/v1/croplens_v1.keras"
    labels_path: str = "../models/v1/labels.json"
    crop_info_path: str = "./data/crop_info.json"
    model_version: str = "v1"

    # Predictions below this confidence are returned as "not sure".
    # TODO (Day 3): replace with the value chosen from validation data.
    confidence_threshold: float = 0.70

    # --- Upload limits ---
    max_upload_mb: int = 5
    allowed_content_types: str = "image/jpeg,image/png,image/webp"

    # --- CORS ---
    allowed_origins: str = "http://localhost:5500,http://127.0.0.1:5500"

    # --- Optional ---
    sentry_dsn: str | None = None
    rate_limit_per_minute: int = 20

    @property
    def content_types_list(self) -> list[str]:
        return [c.strip() for c in self.allowed_content_types.split(",") if c.strip()]

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached so `.env` is parsed once per process, not once per request."""
    return Settings()
