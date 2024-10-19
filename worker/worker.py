# worker.py
import asyncio
from collections import defaultdict
import json
import os
import random
import aio_pika
import httpx
import numpy as np
import pandas as pd
from common.db.db_api import get_db_session, init_db
import joblib  # For loading KNN model
from catboost import CatBoostRegressor, Pool
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from common.models import *


class BundleRecommender:
    def __init__(self):
        self.models_loaded = False
        self.api_key = os.getenv("STEAM_API_KEY")

    def load_models(self):
        # Load KNN model
        self.knn_model = joblib.load("./ml_models/model_knn.pkl")

        # Load CatBoost model
        self.catboost_model = CatBoostRegressor()
        self.catboost_model.load_model("./ml_models/prices_model_2.cbm")

        # Load necessary data
        self.df_train = pd.read_csv("./ml_models/big_df.csv")
        self.user_game_matrix = pd.read_csv(
            "./ml_models/user_game_matrix.csv", index_col=0
        )
        self.game_features_df = pd.read_csv("./ml_models/cb_df.csv")

        # Load Steam IDs
        self.steam_ids = self.load_steam_ids()

        self.feature_columns = pd.read_csv("./ml_models/feature_columns.csv")

        self.models_loaded = True

    def load_steam_ids(self):
        """Loads 17-digit Steam IDs from a file."""
        file_path = "./ml_models/generated_numbers.txt"
        with open(file_path, "r") as file:
            steam_ids = [line.strip() for line in file.readlines()]
        return steam_ids

    async def get_bundle_for_user(self, steam_id: str, session: AsyncSession):
        """Main function to get a bundle of games and its price for the user."""
        # Simulated user games (replace with actual Steam API call)
        user_games = await self.get_steam_games(steam_id)
        print(f"get_bundle_for_user() > user_games: {user_games}")
        print(
            f"get_bundle_for_user() step 1 > steam_id: {steam_id}, user_games: {user_games}"
        )

        async def filter_free_games(recommended_games, df_train):
            # Приведение appid к строковому типу
            df_train["appid"] = df_train["appid"].astype(str)
            recommended_games = [str(appid) for appid in recommended_games]

            # Получение данных о рекомендованных играх и фильтрация дубликатов
            prices_for_recommended_games = df_train[
                df_train["appid"].isin(recommended_games)
            ]
            prices_for_recommended_games = prices_for_recommended_games.drop_duplicates(
                subset="appid", keep="first"
            )
            print("prices_for_recommended_games: ", prices_for_recommended_games)

            # Оставляем только платные игры
            paid_games = prices_for_recommended_games[
                prices_for_recommended_games["price"] > 0
            ]

            # Возвращаем список appid платных игр
            return paid_games["appid"].tolist()

        async def get_popular_games():
            # Получение 3 самых популярных игр по времени игры
            popular_games = (
                self.df_train.groupby("appid")["play_hours"]
                .sum()
                .nlargest(50)
                .index.tolist()
            )
            print(
                f"get_bundle_for_user().get_popular_games() step 1> popular_games: {popular_games}"
            )

            # Фильтрация бесплатных игр
            paid_games = await filter_free_games(popular_games, self.df_train)
            if not paid_games:
                print("No paid games available.")
                return {"bundle_price": 0, "games": []}

            random.shuffle(paid_games)
            # Берем первые 3 игры
            games_for_bundle = paid_games[:3]

            # Оценка стоимости бандла для платных игр
            bundle_price = self.estimate_bundle_price(games_for_bundle)
            print(
                f"get_bundle_for_user().get_popular_games() step 2> bundle_price: {games_for_bundle}"
            )

            # Получение информации о платных играх
            popular_games_info_list = [
                self.get_game_info_from_steam(appid) for appid in games_for_bundle
            ]
            print(
                f"get_bundle_for_user().get_popular_games() step 2.1 > popular_games_info_list: {popular_games_info_list}"
            )

            # Асинхронная загрузка информации об играх
            popular_games_info = await asyncio.gather(
                *popular_games_info_list, return_exceptions=True
            )
            print(
                f"get_bundle_for_user().get_popular_games() step 3> popular_games_info: {popular_games_info}"
            )

            return {"bundle_price": bundle_price, "games": popular_games_info}

        if not user_games:
            return await get_popular_games()

        print(
            f"get_bundle_for_user() step 3 > steam_id: {steam_id}, user_games: {user_games}"
        )
        # Update user_games in the database
        await self.update_user_games(steam_id, user_games, session)
        print(
            f"get_bundle_for_user() step 4 > steam_id: {steam_id}, user_games: {user_games}"
        )
        # Filter games present in the training dataset
        known_appids = set(self.df_train["appid"].unique())
        filtered_user_games = await self.filter_known_games(user_games, known_appids)
        print(
            f"get_bundle_for_user() step 5 > steam_id: {steam_id}, user_games: {user_games}"
        )
        if not filtered_user_games:
            print("get_bundle_for_user() > User has no games from the training set.")
            top_game = max(user_games, key=lambda x: x["playtime_forever"])
            genre = await self.get_game_genre_from_steam(top_game["appid"])
            print(f"Genre: {genre}")
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
                print(
                    f"Filtered prices for recommended games: {prices_for_genre_games[['appid', 'price']]}"
                )

                recommended_games = await filter_free_games(
                    recommended_games, self.df_train
                )

                if len(recommended_games) < 3:
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
                recommended_games = (
                    self.df_train[self.df_train["price"] > 0]
                    .groupby("appid")["play_hours"]
                    .sum()
                    .nlargest(3)
                    .index.tolist()
                )

            bundle_price = self.estimate_bundle_price(recommended_games)
            recommended_games_info = await asyncio.gather(
                *[self.get_game_info_from_steam(appid) for appid in recommended_games]
            )
            return {"bundle_price": bundle_price, "games": recommended_games_info}

        print(
            f"get_bundle_for_user() step 6 > steam_id: {steam_id}, user_games: {user_games}"
        )
        # Recommend using KNN
        recommended_games = await self.recomend_knn(
            steam_id, session, self.user_game_matrix, self.df_train, self.knn_model
        )
        print(f"Recommended games: {recommended_games}")

        recommended_games = [str(appid) for appid in recommended_games]
        self.df_train["appid"] = self.df_train["appid"].astype(str)
        print(
            f"get_bundle_for_user() step 7 > steam_id: {steam_id}, user_games: {user_games}"
        )
        missing_appids = [
            appid
            for appid in recommended_games
            if appid not in self.df_train["appid"].values
        ]
        if missing_appids:
            print(f"Missing appids in df_train: {missing_appids}")

        prices_for_recommended_games = self.df_train[
            self.df_train["appid"].isin(recommended_games)
        ]
        prices_for_recommended_games = prices_for_recommended_games.drop_duplicates(
            subset="appid", keep="first"
        )
        print(
            f"Filtered prices for recommended games: {prices_for_recommended_games[['appid', 'price']]}"
        )

        recommended_games = await filter_free_games(recommended_games, self.df_train)

        if len(recommended_games) < 3:
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

        bundle_price = self.estimate_bundle_price(recommended_games[:3])
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
        """Updates the user_games table in the database with new game data."""
        steam_id = str(steam_id)
        for game in games:
            appid = str(game["appid"])
            playtime_hours = float(game["playtime_forever"] / 60)

            result = await session.execute(
                text(
                    """
                    SELECT * FROM user_games WHERE steam_id = :steam_id AND appid = :appid
                """
                ),
                {"steam_id": steam_id, "appid": appid},
            )
            existing_entry = result.fetchone()

            if existing_entry:
                await session.execute(
                    text(
                        """
                        UPDATE user_games SET play_hours = :play_hours
                        WHERE steam_id = :steam_id AND appid = :appid
                    """
                    ),
                    {
                        "play_hours": playtime_hours,
                        "steam_id": steam_id,
                        "appid": appid,
                    },
                )
            else:
                new_entry = TableUserGamesDbModel(
                    steam_id=steam_id,
                    appid=appid,
                    play_hours=playtime_hours,
                )
                session.add(new_entry)
        await session.commit()

    async def update_all_users_games(self, session: AsyncSession):
        """Asynchronously updates games for all users."""
        batch_size = 10
        steam_ids = self.steam_ids

        for i in range(0, len(steam_ids), batch_size):
            batch = steam_ids[i : i + batch_size]
            games_batch = await asyncio.gather(
                *[self.get_steam_games(steam_id) for steam_id in batch]
            )

            for steam_id, games in zip(batch, games_batch):
                if games:
                    await self.update_user_games(steam_id, games, session)

            await asyncio.sleep(1)

    def estimate_bundle_price(self, recommended_games: list):
        print(f"recommended_games in estimate_bundle_price: {recommended_games}")

        recommended_games = [str(appid) for appid in recommended_games]
        self.game_features_df["appid"] = self.game_features_df["appid"].astype(str)

        selected_games = self.game_features_df[
            self.game_features_df["appid"].isin(recommended_games)
        ]
        print(f"selected_games before drop_duplicates: {selected_games}")

        missing_games = [
            appid
            for appid in recommended_games
            if appid not in self.game_features_df["appid"].values
        ]
        if missing_games:
            print(f"Missing games in game_features_df: {missing_games}")

        selected_games = selected_games.drop_duplicates(subset="appid", keep="first")
        print(f"selected_games after drop_duplicates: {selected_games}")

        if selected_games.empty:
            raise ValueError("No games found in the dataset")

        feature_columns = self.feature_columns["Feature Columns"].tolist()

        # Выбираем только те столбцы, которые остались после исключения
        features = selected_games[feature_columns]

        cat_features: list[int] = []

        predict_pool = Pool(data=features, cat_features=cat_features)
        predicted_prices = self.catboost_model.predict(predict_pool)
        bundle_price = sum(predicted_prices)

        print(f"Predicted games: {recommended_games}")
        return bundle_price  # Возвращаем цену бандла

    async def get_game_info_from_steam(self, appid: int):
        """Asynchronously fetches game information from Steam."""
        url = "https://store.steampowered.com/api/appdetails"
        params = {"appids": appid, "l": "english"}

        async with httpx.AsyncClient(verify=False) as client:
            try:
                print(
                    f"get_game_info_from_steam() > Fetching game info for appid: {appid}"
                )
                response = await client.get(url, params=params)
                print(
                    f"get_game_info_from_steam() > Response status code: {response.status_code}"
                )
                response.raise_for_status()
            except Exception as e:
                print(f"Error fetching game info for appid {appid}: {e}")
                return {"name": "Unknown", "icon": "Unknown"}

        print(f"get_game_info_from_steam() > Got Response content for appid {appid}")

        try:
            response_data = response.json()
            app_data = response_data.get(str(appid), {})
            if app_data.get("success", False):
                data = app_data.get("data", {})

                resp = {
                    "name": data.get("name", "Unknown"),
                    "icon": data.get("header_image", "Unknown"),
                }
                print(f"get_game_info_from_steam() > Game data: {resp}")
                return resp
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
        user_games = await self.get_user_games_from_db(user_id, session)

        if user_id not in user_game_matrix.index:
            playtime_by_appid = defaultdict(list)

            for game in user_games:
                appid = game[2]  # Third element is appid
                play_hours = game[3]  # Fourth element is play_hours
                playtime_by_appid[appid].append(play_hours)

            mean_play_hours_by_appid = {
                appid: np.mean(play_hours)
                for appid, play_hours in playtime_by_appid.items()
            }

            mean_play_hours = np.zeros(user_game_matrix.shape[1])
            for appid, playtime in mean_play_hours_by_appid.items():
                if appid in user_game_matrix.columns:
                    mean_play_hours[user_game_matrix.columns.get_loc(appid)] = playtime

            new_user_df = pd.DataFrame(
                [mean_play_hours], columns=user_game_matrix.columns
            )
            new_user_df.index = [user_id]
            print(f"new_user_df in recomend_knn: {new_user_df}")

            user_game_matrix = pd.concat([user_game_matrix, new_user_df])

        user_idx = user_game_matrix.index.get_loc(user_id)
        print(f"user_idx in recomend_knn: {user_idx}")
        user_vector = user_game_matrix.iloc[user_idx].values.reshape(1, -1)
        print(f"user_vector in recomend_knn: {user_vector}")

        distances, indices = knn_model.kneighbors(user_vector, n_neighbors=top_n + 1)
        similar_user_indices = indices.flatten()[1:]
        similar_users = user_game_matrix.index[similar_user_indices]
        print(f"similar_users in recomend_knn: {similar_users}")

        if not isinstance(user_games, pd.DataFrame):
            user_games = pd.DataFrame(
                user_games, columns=["id", "steam_id", "appid", "play_hours"]
            )

        user_games_set = set(user_games["appid"].values)
        print(f"user_games_set in recomend_knn: {user_games_set}")

        similar_users_games = df_train[df_train["user_id"].isin(similar_users)]
        print(f"similar_users_games in recomend_knn: {similar_users_games}")

        if similar_users_games.empty:
            print("No games found for similar users.")
            return []

        recommendations = similar_users_games[
            ~similar_users_games["appid"].isin(user_games_set)
        ]
        similar_users_games["appid"] = similar_users_games["appid"].astype(str)
        user_games_set = set(map(str, user_games_set))
        recommendations = similar_users_games.loc[
            ~similar_users_games["appid"].isin(user_games_set)
        ]
        print(f"Filtered recommendations: {recommendations}")

        game_recommendations = recommendations.groupby("appid")["play_hours"].sum()
        print(f"game_recommendations in recomend_knn: {game_recommendations}")

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
        user_games = result.fetchall()
        return user_games

    async def get_game_genre_from_steam(self, appid: int):
        """Asynchronously fetches the game's genre from Steam."""
        url = "https://store.steampowered.com/api/appdetails"
        params = {"appids": appid, "l": "english"}

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)

        if response.status_code == 200:
            response_data = response.json()
            app_data = response_data.get(str(appid), {})

            if app_data.get("success", False):
                data = app_data.get("data", {})
                print(f"Game data for appid {appid}: {data}")

                genres = data.get("genres", [])
                if genres:
                    return genres[0].get("description", None)
            else:
                print(f"Failed to fetch game data for appid {appid}: {response_data}")
        else:
            print(f"HTTP error fetching data for appid {appid}: {response.status_code}")

        return None

    async def filter_known_games(self, user_games, known_appids):
        """Filters user games, keeping only those present in the training dataset."""
        filtered_user_games = [
            game for game in user_games if game["appid"] in known_appids
        ]
        return filtered_user_games


# Initialize BundleRecommender once
br = BundleRecommender()
if not br.models_loaded:
    br.load_models()

RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "password")
RABBITMQ_HOST = "rabbitmq"
RABBITMQ_URL = f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:5672/"


async def process_message(message: aio_pika.IncomingMessage):
    async with get_db_session() as session:
        async with message.process():
            task = json.loads(message.body)
            task_id = task.get("id")
            task_type = task.get("type")
            payload = task.get("payload")

            result = {}
            if task_type == "recommend_bundle":
                steam_id = payload.get("steam_id")
                try:
                    print(f"process_message() > try 1 | steam_id: {steam_id}")
                    result = await br.get_bundle_for_user(steam_id, session)
                    print(f"process_message() > try 2 | result: {result}")
                except Exception as e:
                    print(f"process_message() > Error: {str(e)}")
                    result = {"error": f"Error in: process_message() >> {str(e)}"}

            # Publish the result back to the response queue
            response_queue = message.reply_to
            correlation_id = message.correlation_id
            print(
                f"process_message() > result: {result} > response_queue: {response_queue} > correlation_id: {correlation_id}"
            )
            if response_queue:
                async with connection.channel() as channel:
                    # Создаём новый кастомный обменник
                    await channel.default_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps(result).encode(),
                            correlation_id=correlation_id,
                        ),
                        routing_key=response_queue,
                    )

                    print(
                        "process_message() > Message published: ",
                        result,
                        correlation_id,
                        response_queue,
                    )


async def main():
    # Initialize Database
    await init_db()

    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    channel = await connection.channel()
    queue = await channel.declare_queue("task_queue", durable=True)

    await queue.consume(process_message, no_ack=False)

    print("Worker started. Waiting for messages...")
    return connection


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    connection = loop.run_until_complete(main())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(connection.close())
