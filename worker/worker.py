import datetime
import aio_pika
import asyncio
import json

from fastapi import HTTPException
import httpx

from backend.app import crud
from common.auth.jwt_handler import create_access_token
from common.schemas import UserCreateSchema

async def worker():
    # Подключение к RabbitMQ
    connection = await aio_pika.connect_robust("amqp://guest:guest@rabbitmq/")
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue("common_queue", durable=True)

        async def process_message(message: aio_pika.IncomingMessage):
            async with message.process():
                task_data = message.body.decode()
                task = json.loads(task_data)  # Распаковываем сообщение
                task_type = task.get("task_type")

                if task_type == "validate_openid":
                    print(f"Выполняем валидацию OpenID: {task}")

                    # Формируем payload для запроса к Steam
                    payload = {
                        "openid.ns": task["openid.ns"],
                        "openid.mode": task["openid.mode"],
                        "openid.op_endpoint": task["openid.op_endpoint"],
                        "openid.claimed_id": task["openid.claimed_id"],
                        "openid.identity": task["openid.identity"],
                        "openid.return_to": task["openid.return_to"],
                        "openid.response_nonce": task["openid.response_nonce"],
                        "openid.assoc_handle": task["openid.assoc_handle"],
                        "openid.signed": task["openid.signed"],
                        "openid.sig": task["openid.sig"],
                    }

                    # Отправляем запрос на валидацию в Steam
                    async with httpx.AsyncClient() as client:
                        try:
                            response = await client.post(
                                "https://steamcommunity.com/openid/login",
                                data=payload
                            )
                            # Логируем ответ для отладки
                            print(f"Ответ Steam: {response.text}")

                            # Check the result
                            if "is_valid:true" in response.text.strip():
                                claimed_id = openid_params.get("openid.claimed_id")
                                if isinstance(claimed_id, list):
                                    claimed_id = claimed_id[0]

                                if not claimed_id:
                                    raise HTTPException(
                                        status_code=400, detail="Missing claimed_id in OpenID response"
                                    )

                                steam_id = crud.extract_steam_id_from_openid(claimed_id)

                                # Check if the user exists
                                db_user = await crud.get_user_by_steam_id(session, steam_id=steam_id)
                                if not db_user:
                                    user_schema = UserCreateSchema(
                                        id=None,
                                        steam_id=steam_id,
                                        username=None,  # Set a default or extract from Steam
                                        email=None,
                                        created_at=datetime.datetime.now(),
                                        role="user",
                                        password_hash=None,
                                        balance=0.0,
                                        user_games=[],
                                        predictions=[],
                                    )
                                    db_user = await crud.create_user(session, user=user_schema)
                                    # Create JWT token
                                    token = create_access_token(data={"sub": steam_id})
                                    return {"access_token": token}
                            else:
                                raise HTTPException(status_code=400, detail="Invalid OpenID response")
                            
                else:
                    return print(f"Неизвестный тип задачи: {task}")

    await queue.consume(process_message)
    print("Ожидание сообщений...")
    await asyncio.Future()  # Не завершать воркер

if __name__ == "__main__":
    asyncio.run(worker())
