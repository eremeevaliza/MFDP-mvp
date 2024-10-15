# backend/crud.py

import datetime
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from common.models import TableUserDbModel  # type: ignore
from common.schemas import UserCreateSchema


async def get_user_by_steam_id(session: AsyncSession, steam_id: str):
    result = await session.execute(
        select(TableUserDbModel).filter(TableUserDbModel.steam_id == steam_id)
    )
    return result.scalars().first()


def extract_steam_id_from_openid(claimed_id: str) -> str:
    """
    Извлекает Steam ID из URL claimed_id, возвращаемого Steam OpenID.
    URL имеет формат: 'https://steamcommunity.com/openid/id/<steam_id>'
    """
    # Разделяем строку по символу '/' и получаем последний элемент, который является Steam ID
    return claimed_id.split("/")[-1]


async def get_user_by_email(session: AsyncSession, email: str):
    result = await session.execute(
        select(TableUserDbModel).filter(TableUserDbModel.email == email)
    )
    return result.scalars().first()


async def create_user(session: AsyncSession, user: UserCreateSchema):
    db_user = TableUserDbModel(
        id=user.id,
        steam_id=user.steam_id,
        created_at=user.created_at,
        role=user.role,
        password_hash=user.password_hash,
        balance=user.balance,
        username=user.username,
        email=user.email,
        user_games=[],
    )

    session.add(db_user)
    try:
        await session.commit()
        await session.refresh(db_user)
    except IntegrityError:
        await session.rollback()
        raise ValueError("User with given email or steam_id already exists")
    return db_user  # SQLAlchemy-модель
