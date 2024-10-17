import time
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException, requests
from authlib.integrations.starlette_client import OAuth  # type: ignore
from starlette.config import Config
from starlette.requests import Request
from sqlalchemy.orm import Session

from common import models
from common import schemas
from common.auth.jwt_handler import create_access_token
from common.db.db_api import get_db_session

import crud  # type: ignore

config = Config(".env")
oauth = OAuth(config)

oauth.register(
    name="steam",
    client_id="YOUR_STEAM_API_KEY",
    access_token_url="https://api.steampowered.com/",
    authorize_url="https://steamcommunity.com/openid/login",
    client_kwargs={
        "scope": "openid",
    },
)

router = APIRouter()


@router.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for("auth")
    return await oauth.steam.authorize_redirect(request, redirect_uri)


@router.get("/auth")
async def auth(request: Request):
    async with get_db_session() as session:
        token = await oauth.steam.authorize_access_token(request)
        user_info = await oauth.steam.parse_id_token(request, token)
        steam_id = user_info.get("sub")
        if not steam_id:
            raise HTTPException(status_code=400, detail="Steam authentication failed")

        user = crud.get_user_by_steam_id(session, steam_id=steam_id)
        if not user:
            # Создайте нового пользователя или обработайте как требуется
            pass

        # Вернуть токен или установить сессию
        return {"message": "Authenticated", "user": user.username}
