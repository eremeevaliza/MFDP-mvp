import asyncio
from collections import defaultdict
import os
import time
import httpx
import datetime
import bcrypt
from typing import Any, Optional, Dict, List
import requests
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from catboost import CatBoostRegressor, Pool
import joblib  # Для загрузки модели KNN

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


class BundleRecommender:
    def __init__(
        self,
    ):
        self.models_loaded = False
        self.api_key = os.getenv("STEAM_API_KEY")

    def load_models(self):
        # Загрузка модели KNN
        self.knn_model = joblib.load(
            "./ml_models/model_knn.pkl"
        )  # Укажите путь к сохраненной модели KNN

        # Загрузка модели CatBoost
        self.catboost_model = CatBoostRegressor()
        self.catboost_model.load_model(
            "./ml_models/prices_model.cbm"
        )  # Укажите путь к сохраненной модели CatBoost

        # Загрузка данных, необходимых для работы моделей
        self.df_train = pd.read_csv(
            "./ml_models/steam_user_aggregated.csv"
        )  # Замените на путь к вашим данным
        self.user_game_matrix = pd.read_csv(
            "./ml_models/user_game_matrix.csv", index_col=0
        )  # Матрица пользователь-игра
        self.game_features_df = pd.read_csv(
            "./ml_models/df_for_catboost.csv"
        )  # Данные для модели CatBoost

        def load_steam_ids(self):
            """Загружает 17-значные числа из файла в переменную steam_ids."""
            file_path = "./ml_models/generated_numbers.txt"
            with open(file_path, "r") as file:
                steam_ids = [line.strip() for line in file.readlines()]
            return steam_ids

        self.steam_ids = load_steam_ids(self)

        self.models_loaded = True

    async def get_bundle_for_user(self, steam_id: str, session: AsyncSession):
        """Основная функция для получения бандла игр и его цены для пользователя."""
        # Получаем игры пользователя с Steam
        # user_games = await self.get_steam_games(steam_id)
        user_games = [
            {
                "appid": 890470,
                "name": "Fake game",
                "playtime_forever": 3600,
                "img_icon_url": "fcfb366051782b8ebf2aa297f3b746395858cb62",
                "has_community_visible_stats": True,
                "content_descriptorids": [2, 5],
            }
        ]

        async def filter_free_games(self, recommended_games, df_train):
            # Приводим appid в df_train и recommended_games к строковому типу
            df_train["appid"] = df_train["appid"].astype(str)
            recommended_games = [str(appid) for appid in recommended_games]

            # Получаем цены для рекомендованных игр
            prices_for_recommended_games = df_train[
                df_train["appid"].isin(recommended_games)
            ]
            prices_for_recommended_games = prices_for_recommended_games.drop_duplicates(
                subset="appid", keep="first"
            )
            # Оставляем только те игры, у которых цена больше 0
            paid_games = prices_for_recommended_games[
                prices_for_recommended_games["price"] > 0
            ]

            # Возвращаем appid игр, которые не бесплатные
            return paid_games["appid"].tolist()

        # Если профиль закрыт или игр нет, возвращаем популярные игры
        async def get_popular_games():
            popular_games = (
                self.df_train.groupby("appid")["play_hours"]
                .sum()
                .nlargest(3)
                .index.tolist()
            )
            bundle_price = self.estimate_bundle_price(popular_games)
            popular_games_info = await asyncio.gather(
                *[self.get_game_info_from_steam(appid) for appid in popular_games]
            )
            return {"bundle_price": bundle_price, "games": popular_games_info}

        if not user_games:
            return await get_popular_games()

        # Обновляем таблицу user_games
        await self.update_user_games(steam_id, user_games, session)

        # Фильтруем игры, которые есть в обучающем наборе данных
        known_appids = set(self.df_train["appid"].unique())
        filtered_user_games = self.filter_known_games(user_games, known_appids)

        # Если у пользователя нет игр из обучающего набора, возвращаем игры по жанру
        if not filtered_user_games:
            print(
                "!!!!!!!!!!!!!!!!!!!User has no games from the training set!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            top_game = max(user_games, key=lambda x: x["playtime_forever"])
            genre = await self.get_game_genre_from_steam(top_game["appid"])
            print(f"!!!!!!!!!!!!!!!!!!!Genre: {genre}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if genre:
                genre_games = self.df_train[
                    self.df_train["genres"].str.contains(genre, na=False)
                ]
                recommended_games = (
                    genre_games.groupby("appid")["play_hours"].sum().index.tolist()
                )
                prices_for_genre_games = self.df_train[
                    self.df_train["appid"].isin(recommended_games)
                ]
                prices_for_genre_games = prices_for_genre_games.drop_duplicates(
                    subset="appid", keep="first"
                )
                # Логирование результатов
                print(
                    f"Filtered prices for recommended games: {prices_for_genre_games[['appid', 'price']]}"
                )

                # Фильтруем бесплатные игры
                recommended_games = await filter_free_games(
                    self, recommended_games, self.df_train
                )

                # Проверяем, чтобы всегда было ровно три игры
                if len(recommended_games) < 3:
                    # Если недостаточно игр, добавляем платные популярные игры
                    additional_games_needed = 3 - len(recommended_games)
                    popular_paid_games = (
                        self.df_train[self.df_train["price"] > 0]
                        .groupby("appid")["play_hours"]
                        .sum()
                        .nlargest(additional_games_needed)
                        .index.tolist()
                    )
                    recommended_games.extend(popular_paid_games)

                    print(f"Final recommended games: {recommended_games}")
            else:
                # Если жанр не найден, возвращаем популярные игры
                recommended_games = (
                    self.df_train[self.df_train["price"] > 0]
                    .groupby("appid")["play_hours"]
                    .sum()
                    .nlargest(additional_games_needed)
                    .index.tolist()
                )

            bundle_price = self.estimate_bundle_price(recommended_games)
            recommended_games_info = await asyncio.gather(
                *[self.get_game_info_from_steam(appid) for appid in recommended_games]
            )
            return {"bundle_price": bundle_price, "games": recommended_games_info}

        # Строим матрицу и находим рекомендации с помощью KNN
        recommended_games = await self.recomend_knn(
            steam_id, session, self.user_game_matrix, self.df_train, self.knn_model
        )
        print(f"Recommended games: {recommended_games}")

        # Приведение appid к строковому типу данных
        recommended_games = [str(appid) for appid in recommended_games]
        self.df_train["appid"] = self.df_train["appid"].astype(str)

        # Проверка на отсутствие appid в df_train
        missing_appids = [
            appid
            for appid in recommended_games
            if appid not in self.df_train["appid"].values
        ]
        if missing_appids:
            print(f"Missing appids in df_train: {missing_appids}")

        # Фильтрация рекомендованных игр
        prices_for_recommended_games = self.df_train[
            self.df_train["appid"].isin(recommended_games)
        ]

        # Фильтрация рекомендованных игр и их цен
        # Отбираем только первую запись для каждого appid
        prices_for_recommended_games = prices_for_recommended_games.drop_duplicates(
            subset="appid", keep="first"
        )

        # Логирование результатов
        print(
            f"Filtered prices for recommended games: {prices_for_recommended_games[['appid', 'price']]}"
        )

        # Фильтруем бесплатные игры
        recommended_games = await filter_free_games(
            self, recommended_games, self.df_train
        )

        # Проверяем, чтобы всегда было ровно три игры
        if len(recommended_games) < 3:
            # Если недостаточно игр, добавляем платные популярные игры
            additional_games_needed = 3 - len(recommended_games)
            popular_paid_games = (
                self.df_train[self.df_train["price"] > 0]
                .groupby("appid")["play_hours"]
                .sum()
                .nlargest(additional_games_needed)
                .index.tolist()
            )
            recommended_games.extend(popular_paid_games)

        print(f"Final recommended games: {recommended_games}")

        # Оцениваем цену бандла
        bundle_price = self.estimate_bundle_price(recommended_games[:3])

        # Получаем информацию об играх
        recommended_games_info = await asyncio.gather(
            *[self.get_game_info_from_steam(appid) for appid in recommended_games[:3]]
        )

        return {"bundle_price": bundle_price, "games": recommended_games_info}

    async def get_steam_games(self, steam_id: str):
        """Асинхронно запрашивает игры пользователя у Steam."""
        url = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
        params = {
            "key": self.api_key,
            "steamid": steam_id,
            "format": "json",
            "include_appinfo": True,
            "include_played_free_games": True,
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
        if response.status_code == 200:
            games = response.json().get("response", {}).get("games", [])
            return games
        return []

    async def update_user_games(
        self, steam_id: str, games: list, session: AsyncSession
    ):
        """Обновляет таблицу user_games в базе данных с новыми данными о играх пользователя."""

        # Приводим steam_id в строковое значение на всякий случай
        steam_id = str(steam_id)

        for game in games:
            appid = game["appid"]
            playtime_hours = game["playtime_forever"] / 60  # Приводим минуты в часы
            appid = str(appid)  # Приводим appid в строку
            playtime_hours = float(playtime_hours)  # Приводим playtime_hours в число

            # Проверяем, есть ли запись в базе данных
            result = await session.execute(
                text(
                    """
                    SELECT * FROM user_games WHERE steam_id = :steam_id AND appid = :appid
                """
                ),
                {"steam_id": steam_id, "appid": appid},  # Передаём steam_id как строку
            )
            existing_entry = result.fetchone()

            if existing_entry:
                # Обновляем запись
                await session.execute(
                    text(
                        """
                        UPDATE user_games SET play_hours = :play_hours
                        WHERE steam_id = :steam_id AND appid = :appid
                    """
                    ),
                    {
                        "play_hours": playtime_hours,
                        "steam_id": steam_id,  # Передаём steam_id как строку
                        "appid": appid,
                    },
                )
            else:
                # Создаем новую запись
                new_entry = TableUserGamesDbModel(
                    steam_id=steam_id,  # Убедитесь, что это строка
                    appid=appid,
                    play_hours=playtime_hours,
                )
                session.add(new_entry)
        await session.commit()

    async def update_all_users_games(self, session: AsyncSession):
        """Асинхронно обновляет игры для всех пользователей."""

        self.load_models()  # Загружаем модели
        batch_size = 10  # Размер батча для обработки пользователей
        steam_ids = self.steam_ids  # Получаем список steam_id

        for i in range(0, len(steam_ids), batch_size):
            batch = steam_ids[
                i : i + batch_size
            ]  # Формируем текущий батч из пользователей

            # Асинхронно получаем игры для всех пользователей в текущем батче
            # Параллельно отправляем запросы для каждого steam_id в батче
            games_batch = await asyncio.gather(
                *[self.get_steam_games(steam_id) for steam_id in batch]
            )

            # Обновляем игры для каждого пользователя из текущего батча
            for steam_id, games in zip(batch, games_batch):
                if games:  # Если у пользователя есть игры
                    await self.update_user_games(steam_id, games, session)

            # Асинхронная пауза для предотвращения излишней нагрузки на Steam API
            await asyncio.sleep(1)

    def estimate_bundle_price(self, recommended_games: list):
        print(f"recommended_games in estimate_bundle_price: {recommended_games}")

        # Приведение appid к строковому типу для корректной фильтрации
        recommended_games = [str(appid) for appid in recommended_games]
        self.game_features_df["appid"] = self.game_features_df["appid"].astype(str)

        # Фильтрация игр на основе appid
        selected_games = self.game_features_df[
            self.game_features_df["appid"].isin(recommended_games)
        ]
        print(f"selected_games before drop_duplicates: {selected_games}")

        # Проверка на наличие отсутствующих игр
        missing_games = [
            appid
            for appid in recommended_games
            if appid not in self.game_features_df["appid"].values
        ]
        if missing_games:
            print(f"Missing games in game_features_df: {missing_games}")

        # Убираем дубликаты
        selected_games = selected_games.drop_duplicates(subset="appid", keep="first")
        print(f"selected_games after drop_duplicates: {selected_games}")

        if selected_games.empty:
            raise ValueError("No games found in the dataset")

        # Указываем нужные колонки
        feature_columns = [
            "achievements",
            "average_playtime",
            "developer",
            "english",
            "median_playtime",
            "negative_ratings",
            "positive_ratings",
            "price",
            "publisher",
            "required_age",
            "genres_Indie",
            "genres_Action",
            "genres_Adventure",
            "genres_Casual",
            "genres_Strategy",
            "genres_RPG",
            "genres_Simulation",
            "genres_Racing",
            "genres_Sports",
            "genres_Early Access",
            "categories_Single-player",
            "categories_Steam Trading Cards",
            "categories_Steam Achievements",
            "categories_Steam Cloud",
            "categories_Full controller support",
            "categories_Multi-player",
            "categories_Steam Leaderboards",
            "categories_Partial Controller Support",
            "categories_Co-op",
            "categories_Stats",
            "release_year",
            "platform_windows",
            "platform_mac",
            "platform_linux",
            "estimated_owners",
        ]

        features = selected_games[feature_columns]

        # Список категориальных признаков
        cat_features = [
            "developer",
            "publisher",
            # Добавьте сюда другие категориальные признаки, если они есть
        ]

        # Создаем Pool для предсказания
        predict_pool = Pool(data=features, cat_features=cat_features)

        predicted_prices = self.catboost_model.predict(predict_pool)
        bundle_price = sum(predicted_prices)
        print(f"Predicted games: {recommended_games}")
        return bundle_price

    async def get_game_info_from_steam(self, appid: int):
        """Асинхронно получает информацию об игре, включая название и иконку, с Steam."""
        url = "https://store.steampowered.com/api/appdetails"  # Используем HTTPS
        params = {"appids": appid}
        print(f"!!!!!!!!!!!!!!!!!!!!!Getting game info for appid: {appid}")

        async with httpx.AsyncClient(
            verify=False
        ) as client:  # Отключаем проверку SSL для отладки
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()  # Проверяем на наличие HTTP ошибок
            except Exception as e:
                print(f"Error fetching game info for appid {appid}: {e}")
                return {"name": "Unknown", "icon": "Unknown"}

        # Печатаем полный ответ для диагностики
        print(f"Response content for appid {appid}: {response.content}")

        try:
            response_data = response.json()
            app_data = response_data.get(str(appid), {})
            if app_data.get("success", False):
                data = app_data.get("data", {})
                print(f"Game info: {data}")
                return {
                    "name": data.get("name", "Unknown"),
                    "icon": data.get("header_image", "Unknown"),
                }
            else:
                print(f"Failed to fetch game data for appid {appid}: {response_data}")
        except (KeyError, ValueError) as e:
            print(f"Error processing response for appid {appid}: {e}")

        return {"name": "Unknown", "icon": "Unknown"}

    async def recomend_knn(
        self,
        user_id: str,
        session: AsyncSession,
        user_game_matrix,
        df_train,
        knn_model,
        top_n=3,
    ):
        # Await the user games from the database
        user_games = await self.get_user_games_from_db(user_id, session)

        # Check if the user_id exists in the user_game_matrix DataFrame
        if user_id not in user_game_matrix.index:

            playtime_by_appid = defaultdict(list)

            for game in user_games:
                appid = game[2]  # Third element is appid
                play_hours = game[3]  # Fourth element is play_hours
                playtime_by_appid[appid].append(play_hours)

            # Create a new entry for the user using the play hours for their games
            mean_play_hours_by_appid = {
                appid: np.mean(play_hours)
                for appid, play_hours in playtime_by_appid.items()
            }

            # Create an array of play hours based on user_game_matrix columns
            mean_play_hours = np.zeros(user_game_matrix.shape[1])
            for appid, playtime in mean_play_hours_by_appid.items():
                if appid in user_game_matrix.columns:
                    mean_play_hours[user_game_matrix.columns.get_loc(appid)] = playtime

            # Create a new DataFrame for the user and assign user_id as index
            new_user_df = pd.DataFrame(
                [mean_play_hours], columns=user_game_matrix.columns
            )
            new_user_df.index = [user_id]
            print(f"new_user_df in recomend_knn: {new_user_df}")

            # Concatenate to user_game_matrix
            user_game_matrix = pd.concat([user_game_matrix, new_user_df])

        # Get the index of the user from user_game_matrix DataFrame
        user_idx = user_game_matrix.index.get_loc(user_id)
        print(f"user_idx in recomend_knn: {user_idx}")
        user_vector = user_game_matrix.iloc[user_idx].values.reshape(1, -1)
        print(f"user_vector in recomend_knn: {user_vector}")

        # Get the k-nearest neighbors (including the user)
        distances, indices = knn_model.kneighbors(
            user_vector, n_neighbors=top_n + 1
        )  # +1 to exclude the user themselves

        # Get similar user indices, excluding the first one (which is the user themselves)
        similar_user_indices = indices.flatten()[1:]  # Exclude self
        similar_users = user_game_matrix.index[similar_user_indices]
        print(f"similar_users in recomend_knn: {similar_users}")

        # Convert user_games to DataFrame if necessary
        if not isinstance(user_games, pd.DataFrame):
            user_games = pd.DataFrame(
                user_games, columns=["id", "steam_id", "appid", "play_hours"]
            )

        # Get the games that the user has already played
        user_games_set = set(user_games["appid"].values)
        print(f"user_games_set in recomend_knn: {user_games_set}")

        # Filter games played by similar users
        similar_users_games = df_train[df_train["user_id"].isin(similar_users)]
        print(f"similar_users_games in recomend_knn: {similar_users_games}")

        if similar_users_games.empty:
            print("No games found for similar users.")
            return []

        # Remove games that the user has already played
        recommendations = similar_users_games[
            ~similar_users_games["appid"].isin(user_games_set)
        ]

        similar_users_games["appid"] = similar_users_games["appid"].astype(str)

        user_games_set = set(map(str, user_games_set))

        # Фильтруем игры, которые пользователь уже играл, используя метод .loc для явного фильтра
        recommendations = similar_users_games.loc[
            ~similar_users_games["appid"].isin(user_games_set)
        ]

        # Логирование результатов для проверки
        print(f"Filtered recommendations: {recommendations}")

        # Group by appid and sum the play hours
        game_recommendations = recommendations.groupby("appid")["play_hours"].sum()
        print(f"game_recommendations in recomend_knn: {game_recommendations}")

        # Get the top N recommended games
        recommended_games = game_recommendations.index.tolist()
        print(f"recommended_games in recomend_knn: {recommended_games}")
        return recommended_games

    async def get_user_games_from_db(self, steam_id: str, session: AsyncSession):
        print(f"steam_id in get_user_games_from_db: {steam_id}")
        """Fetch the user's games from the database."""
        query = """
            SELECT * FROM user_games WHERE steam_id = :steam_id
        """
        result = await session.execute(text(query), {"steam_id": steam_id})
        user_games = result.fetchall()  # Or `result.all()` depending on what you need
        return user_games

    async def get_game_genre_from_steam(self, appid: int):
        """Асинхронно получает жанр игры по appid."""
        # Используем HTTPS вместо HTTP
        url = "https://store.steampowered.com/api/appdetails"
        params = {"appids": appid}

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)

        if response.status_code == 200:
            response_data = response.json()  # Сохраняем результат запроса
            app_data = response_data.get(str(appid), {})  # Получаем данные по appid

            if app_data.get("success", False):  # Проверка успешности запроса
                data = app_data.get("data", {})
                print(f"Game data for appid {appid}: {data}")

                # Извлекаем жанры
                genres = data.get("genres", [])
                if genres:
                    return genres[0].get("description", None)  # Возвращаем первый жанр
            else:
                print(f"Failed to fetch game data for appid {appid}: {app_data}")
        else:
            print(f"HTTP error fetching data for appid {appid}: {response.status_code}")

        return None

    async def filter_known_games(self, user_games, known_appids):
        """Фильтрует игры пользователя, оставляя только те, которые есть в обучающем наборе данных."""
        filtered_user_games = [
            game for game in user_games if game["appid"] in known_appids
        ]
        return filtered_user_games
