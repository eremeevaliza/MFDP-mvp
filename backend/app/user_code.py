# user_code.py
import httpx
import datetime
import bcrypt

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from common.schemas import UserCreateSchema
from common.models import *

import crud  # type: ignore


class PasswordUser:
    def __init__(self, password: str):
        self._password_hash = self._hash_password(password)

    def _hash_password(self, password: str) -> str:
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed_password.decode("utf-8")

    def check_password(self, input_password: str, stored_password_hash: str) -> bool:
        return bcrypt.checkpw(
            input_password.encode("utf-8"), stored_password_hash.encode("utf-8")
        )


async def get_steam_user_info(steam_id64: str, api_key: str):
    url = "http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/"
    params = {"key": api_key, "steamids": steam_id64}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            players = data.get("response", {}).get("players", [])
            if players:
                return players[0]
            return None
        except httpx.RequestError as e:
            print(f"Error fetching Steam user info: {e}")
            return None


class AsUserController:
    def __init__(self):
        self._role = "user"

    async def register_with_email(
        self, session: AsyncSession, email: str, password: str
    ) -> str:
        if not email:
            raise ValueError("Email is required")
        if not password:
            raise ValueError("Password is required")

        result = await session.execute(select(TableUserDbModel).filter_by(email=email))
        existing_user = result.scalars().first()
        if existing_user:
            raise ValueError("Email is already registered")

        password_user = PasswordUser(password)
        new_user = TableUserDbModel(
            email=email, password_hash=password_user._password_hash, role=self._role
        )
        session.add(new_user)

        try:
            await session.commit()
            await session.refresh(new_user)
            return "User registered"
        except IntegrityError:
            await session.rollback()
            raise ValueError("Email is already registered")

    async def get_or_create_user(
        self, session: AsyncSession, steam_id: str, username: str
    ) -> TableUserDbModel:
        user = await self.get_user_by_steam_id(session, steam_id=steam_id)
        if not user:
            user = TableUserDbModel(
                steam_id=steam_id,
                username=username,
                created_at=datetime.datetime.utcnow(),
                role="user",
            )
            session.add(user)
            try:
                await session.commit()
                await session.refresh(user)
            except IntegrityError:
                await session.rollback()
                raise ValueError("Steam ID is already registered")
        return user

    async def get_user_by_steam_id(
        self, session: AsyncSession, steam_id: str
    ) -> TableUserDbModel:
        result = await session.execute(
            select(TableUserDbModel).filter_by(steam_id=steam_id)
        )
        user = result.scalars().first()
        if user is None:
            raise ValueError("User not found")
        return user

    async def register_with_steam(
        self, session: AsyncSession, user: UserCreateSchema
    ) -> TableUserDbModel:
        return await crud.create_user(session, user)
