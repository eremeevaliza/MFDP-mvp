# models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import DeclarativeBase, relationship, mapped_column
import datetime


class BaseDbModel(DeclarativeBase):
    pass


class TableUserDbModel(BaseDbModel):
    __tablename__ = "users"
    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    steam_id = mapped_column(String, unique=True, nullable=True)
    username = mapped_column(String, nullable=True)
    email = mapped_column(String, unique=True, nullable=True)
    created_at = mapped_column(DateTime, default=datetime.datetime.utcnow)
    role = mapped_column(String, nullable=False)
    password_hash = mapped_column(String, nullable=True)
    balance = mapped_column(Float, default=0.0)

    user_games = relationship("TableUserGamesDbModel", back_populates="user")


class TableUserGamesDbModel(BaseDbModel):
    __tablename__ = "user_games"
    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    steam_id = mapped_column(String, ForeignKey("users.steam_id"))
    appid = mapped_column(String, nullable=True)
    play_hours = mapped_column(Float, default=0.0)

    user = relationship("TableUserDbModel", back_populates="user_games")
