# common/schemas.py

from pydantic import BaseModel, EmailStr
from typing import Optional, List
import datetime


class UserBaseSchema(BaseModel):
    steam_id: Optional[str] = None
    username: Optional[str] = None
    email: Optional[EmailStr] = None  # Для регистрации по email


class UserCreateSchema(BaseModel):
    id: Optional[int] = None
    steam_id: str
    username: Optional[str] = None
    email: Optional[str] = None
    created_at: Optional[datetime.datetime] = None
    role: Optional[str] = None
    password_hash: Optional[str] = None
    balance: Optional[float] = None
    user_games: Optional[List] = None
    predictions: Optional[List] = None


class UserSchema(UserBaseSchema):
    id: int
    role: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True  # Заменили orm_mode на from_attributes


class GameSchema(BaseModel):
    id: int
    name: str
    # Добавьте другие поля, если необходимо

    class Config:
        from_attributes = True


class PredictionBaseSchema(BaseModel):
    user_id: int
    game_id: int
    prediction: str


class PredictionSchema(PredictionBaseSchema):
    steam_id: str
    games: List[str]
    bundle_price: float

    class Config:
        from_attributes = True
