from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError  # type: ignore
from common.db.db_api import get_db_session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from common.models import TableUserDbModel

SECRET_KEY = "11111111"  # Убедитесь, что этот ключ совпадает с ключом, используемым для создания токена
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/login")


async def get_user_id_by_email(email: str, session) -> int:
    async with get_db_session() as session:
        try:
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>EMAIL: {email}")
            result = await session.execute(
                select(TableUserDbModel).filter(TableUserDbModel.email == email)
            )
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>RESULT: {result}")
            user = result.scalars().first()
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>USER: {user}")
            if user:
                return user.user_id
                print(
                    f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>USER ID: {user._user_id}"
                )
            else:
                return -1
        except Exception as e:
            raise HTTPException(status_code=500, detail="Internal Server Error")


async def get_current_user(
    token: str = Depends(oauth2_scheme), session: AsyncSession = Depends(get_db_session)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email: str = payload.get("user")  # изменено на 'user'
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>EMAIL: {email}")
        if email is None:
            raise credentials_exception
        user_id = await get_user_id_by_email(email, session)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>USER ID: {user_id}")
    except JWTError as e:
        print(f"JWTError: {e}")
        raise credentials_exception
    except Exception as e:
        print(f"Exception: {e}")
        raise credentials_exception
    return {"email": email, "user_id": user_id}
