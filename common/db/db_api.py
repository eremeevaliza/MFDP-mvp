# common/db/db_api.py

import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from contextlib import asynccontextmanager
import sys
import os

from common.models import BaseDbModel
from common.db.db_config import get_settings

# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Get database settings from configuration
dbSettings = get_settings()

# Create an asynchronous engine
async_engine = create_async_engine(
    dbSettings.DATABASE_URL_asyncpg,  # Asynchronous URL
    echo=True,
    pool_size=5,
    max_overflow=10,
)

# Create an asynchronous sessionmaker
async_session = async_sessionmaker(
    async_engine, expire_on_commit=False, class_=AsyncSession
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def get_db_session():
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Session rollback due to exception: %s", e)
            raise


# Asynchronous function to initialize the database
async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(BaseDbModel.metadata.create_all)
