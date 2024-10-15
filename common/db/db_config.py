from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional
import os
from dotenv import load_dotenv

# Определение пути к файлу .env в зависимости от среды выполнения
if os.path.exists("/.dockerenv"):
    # Код запущен в Docker-контейнере
    env_path = "/app/.env"  # Путь к .env внутри контейнера
else:
    # Код запущен локально
    env_path = os.path.join(
        os.path.dirname(__file__), "../.env"
    )  # Путь к .env локально

print(env_path)

load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):
    DB_HOST: Optional[str] = None
    DB_PORT: Optional[int] = None
    DB_USER: Optional[str] = None
    DB_PASS: Optional[str] = None
    DB_NAME: Optional[str] = None

    SECRET_KEY: Optional[str] = None

    @property
    def DATABASE_URL_asyncpg(self):
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def DATABASE_URL_psycopg(self):
        return f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    model_config = SettingsConfigDict(env_file=env_path, extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
