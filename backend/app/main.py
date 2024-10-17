# main.py
import asyncio
import json
import os
from typing import List
import uuid
import aio_pika
import requests
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import mapped_column

from common.db.db_api import get_db_session, init_db
from common.events import event_manager
from common.auth.jwt_handler import (
    create_access_token,
    get_steam_id_from_token,
)  # JWT функции
from common.schemas import *

import crud  # type: ignore
from fastapi import Body  # Import Body for request data
import httpx  # Use httpx for async HTTP requests
from aio_pika import Message, DeliveryMode, ExchangeType

# Импортируем синхронные библиотеки внутри функции, чтобы избежать проблем с асинхронностью

app = FastAPI(root_path="/backend")

RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "password")
RABBITMQ_HOST = "rabbitmq"
RABBITMQ_URL = f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:5672/"


async def send_task(task_type: str, payload: dict) -> dict:
    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    # Declare a unique response queue for this request
    result_queue = await channel.declare_queue(exclusive=True)

    correlation_id = str(uuid.uuid4())

    message = {"id": correlation_id, "type": task_type, "payload": payload}

    await channel.default_exchange.publish(
        Message(
            body=json.dumps(message).encode(),
            reply_to=result_queue.name,
            correlation_id=correlation_id,
            delivery_mode=DeliveryMode.PERSISTENT,
        ),
        routing_key="task_queue",
    )

    # Wait for the response
    future = asyncio.get_event_loop().create_future()

    async def on_response(message: aio_pika.abc.AbstractIncomingMessage):
        if message.correlation_id == correlation_id:
            future.set_result(json.loads(message.body))
            await message.channel.close()

    await result_queue.consume(on_response, no_ack=True)

    response = await future
    await connection.close()
    return response


@app.on_event("startup")
async def startup_event():
    await init_db()


@app.post("/oauth_validate")
async def validate_steam_openid(openid_params: dict = Body(...)):
    async with get_db_session() as session:
        # Build the payload for Steam validation
        payload = {
            "openid.ns": openid_params.get("openid.ns", [""])[0],
            "openid.mode": "check_authentication",
            "openid.op_endpoint": openid_params.get("openid.op_endpoint", [""])[0],
            "openid.claimed_id": openid_params.get("openid.claimed_id", [""])[0],
            "openid.identity": openid_params.get("openid.identity", [""])[0],
            "openid.return_to": openid_params.get("openid.return_to", [""])[0],
            "openid.response_nonce": openid_params.get("openid.response_nonce", [""])[
                0
            ],
            "openid.assoc_handle": openid_params.get("openid.assoc_handle", [""])[0],
            "openid.signed": openid_params.get("openid.signed", [""])[0],
            "openid.sig": openid_params.get("openid.sig", [""])[0],
        }

        # Log payload for debugging
        print(f"Payload for Steam validation: {payload}")

        # Asynchronous HTTP request to Steam
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://steamcommunity.com/openid/login", data=payload
            )

        # Log the response
        print(f"Steam validation response: {response.text}")

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


@app.post("/recommend_bundle")
async def get_prediction(steam_id: dict = Depends(get_steam_id_from_token)):
    # Get the user's Steam ID from the token
    task_payload = {"steam_id": steam_id}

    try:
        result = await send_task("recommend_bundle", task_payload)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Проверка состояния сервиса.
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
