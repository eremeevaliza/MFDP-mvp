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


"""
@router.post("/oauth_validate")
async def validate_steam_openid(openid_params: dict):
    print("auth.py: validate_steam_openid()")
    # Собираем параметры для валидации через Steam
    payload = {
        "openid.ns": openid_params.get("openid.ns"),
        "openid.mode": "check_authentication",
        "openid.signed": openid_params.get("openid.signed"),
        "openid.sig": openid_params.get("openid.sig"),
        "openid.op_endpoint": openid_params.get("openid.op_endpoint"),
        "openid.claimed_id": openid_params.get("openid.claimed_id"),
        # Добавляем остальные параметры, как требуется для Steam
    }

    # Делаем запрос на Steam для проверки подлинности
    response = requests.post("https://steamcommunity.com/openid/login", data=payload)

    if "is_valid:true" in response.text:
        steam_id = openid_params.get("openid.claimed_id")
        # Создаем JWT (пример ниже)
        token = create_access_token(data={"sub": steam_id})
        return {"access_token": token}
    else:
        raise HTTPException(status_code=400, detail="Invalid OpenID response")
"""
