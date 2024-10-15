# common/db/db_api.py

import logging
from typing import AsyncGenerator, Dict
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from contextlib import asynccontextmanager
import sys
import os

from common.models import BaseDbModel

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from common.db.db_config import get_settings

# Получаем настройки из конфигурации
dbSettings = get_settings()

# Создаём асинхронный движок
async_engine = create_async_engine(
    dbSettings.DATABASE_URL_asyncpg,  # Асинхронный URL
    echo=True,
    pool_size=5,
    max_overflow=10,
)

# Создаём асинхронный sessionmaker
async_session = async_sessionmaker(
    async_engine, expire_on_commit=False, class_=AsyncSession
)

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Session rollback due to exception: %s", e)
            raise


# Асинхронная функция инициализации базы данных
async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(BaseDbModel.metadata.create_all)
