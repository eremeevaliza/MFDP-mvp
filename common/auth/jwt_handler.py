import time
from datetime import datetime
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError  # type: ignore
from common.db.db_api import get_settings

settings = get_settings()
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def create_access_token(data: any) -> str:
    # payload = {"user": user, "expires": time.time() + 3600}
    data["expires"] = time.time() + 3600
    token = jwt.encode(data, SECRET_KEY, algorithm="HS256")
    return token


def verify_access_token(token: str) -> dict:
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        expire = data.get("expires")
        if expire is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No access token supplied",
            )
        if datetime.utcnow() > datetime.utcfromtimestamp(expire):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Token expired!"
            )
        return data
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token"
        )


async def get_steam_id_from_token(token: str) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Не удалось проверить токен",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        steam_id: str = payload.get("sub")
        if steam_id is None:
            raise credentials_exception
        return steam_id
    except JWTError:
        raise credentials_exception


async def get_steam_id_from_token(token: str = Depends(oauth2_scheme)) -> str:
    payload = verify_access_token(token)
    steam_id: str = payload.get("sub")
    if steam_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Не удалось извлечь Steam ID из токена",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return steam_id
