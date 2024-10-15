import asyncio
import json
import uuid
import aiormq
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session
from common.db.db_api import get_db_session
from backend.app.user_code import (
    TableData,
    TablePredictions,
    AsUserController as UserController,
    AsPredictModel as PredictModel,
    TableUser,
)
from common.models_old.pm import (
    PredictionHistoryResponse,
    User,
    PredictionData,
    PredictionRequest,
    TokenResponse,
)
from common.events.event_manager import event_manager

from common.auth.jwt_handler import create_access_token
from common.auth.dependencies import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
import logging
import aio_pika
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "password")
DATABASE_URL = os.getenv("DATABASE_URL")

user_route = APIRouter()


async def get_user_role(email: str, session: AsyncSession) -> str:
    try:
        result = await session.execute(select(TableUser).filter_by(email=email))
        user = result.scalar()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )
        return user.role
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@user_route.get("/role")
async def read_user_role(email: str):
    async with get_db_session() as session:
        role = await get_user_role(email, session)
        return {"email": email, "role": role}


@user_route.post("/signup")
async def signup(data: User):
    async with get_db_session() as session:
        print(f"Signup request received with email: {data.email}")
        print(f"Session object: {session}")

        controller = UserController()
        try:
            result = await controller.register_with_email(
                session, data.email, data.password
            )
            print(f"User registered successfully: {data.email}")

            await event_manager.dispatch("user_registered", email=data.email)

            return {"message": result}
        except ValueError as e:
            print(f"Error during signup: {e}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@user_route.post("/login", response_model=TokenResponse)
async def login(user: OAuth2PasswordRequestForm = Depends()) -> dict:
    async with get_db_session() as session:
        controller = UserController()
        print(f"Login request received with email: {user.username}")
        try:
            result = await controller.authenticate(
                session, email=user.username, password=user.password
            )
            if result == "User authenticated":
                access_token = create_access_token(user.username)
                await event_manager.dispatch("user_logged_in", email=user.username)
                return {"access_token": access_token, "token_type": "Bearer"}
            return {"message": result}
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal Server Error",
            )


@user_route.post("/signup/superadmin")
async def signup_superadmin(data: User):
    async with get_db_session() as session:
        print(f"Signup request received with email: {data.email}")
        print(f"Session object: {session}")
        controller = UserController()
        try:
            result = await controller.admin_register(session, data.email, data.password)
            print(f"User registered successfully: {data.email}")

            await event_manager.dispatch("user_registered", email=data.email)

            return {"message": result}
        except ValueError as e:
            print(f"Error during signup: {e}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@user_route.post("/login/superadmin")
async def login_superadmin(data: User, session: Session = Depends(get_db_session)):
    controller = UserController()
    try:
        result = controller.admin_authenticate(session, data.email, data.password)

        await event_manager.dispatch("user_logged_in", email=data.email)

        return {"message": result}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@user_route.post("/data")
async def load_data(
    data: PredictionData, current_user: dict = Depends(get_current_user)
):
    try:
        task_data = {
            "email": current_user["email"],
            "user_id": current_user["user_id"],
            "data": data.data,
            "name": "load_data",
        }
        correlation_id = str(uuid.uuid4())
        result_queue = f"results_{correlation_id}"

        connection = await aio_pika.connect_robust(
            f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@rabbitmq/"
        )
        async with connection:
            channel = await connection.channel()
            result_queue_obj = await channel.declare_queue(
                result_queue, auto_delete=True
            )

            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(task_data).encode(),
                    reply_to=result_queue,
                    correlation_id=correlation_id,
                ),
                routing_key="data_tasks",
            )

            logger.debug(
                f"Data task sent for user {current_user['email']} with data {data.data}"
            )
            return {"status": "success", "message": "Data task sent successfully"}
    except Exception as e:
        logger.error(f"Error sending data task: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@user_route.get("/prediction")
async def get_prediction(current_user: dict = Depends(get_current_user)):
    async with get_db_session() as session:
        try:
            result = await session.execute(
                select(TableData.data_id)
                .filter_by(email=current_user["email"], user_id=current_user["user_id"])
                .order_by(TableData.data_id.desc())
                .limit(1)
            )
            data_id = result.scalar()

            if not data_id:
                return {"message": "No data found for the user"}

            correlation_id = str(uuid.uuid4())
            result_queue_id = f"results_{correlation_id}"

            task = {
                "email": current_user["email"],
                "user_id": current_user["user_id"],
                "data_id": data_id,
                "reply_to": result_queue_id,
                "correlation_id": correlation_id,
                "name": "get_prediction",
            }

            connection = await aio_pika.connect_robust(
                f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@rabbitmq/"
            )
            async with connection:
                channel = await connection.channel()
                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(task).encode(),
                        correlation_id=correlation_id,
                        reply_to=result_queue_id,
                    ),
                    routing_key="prediction_tasks",
                )
                logger.debug(
                    f"Task for user {current_user['email']} with data id {data_id} published"
                )
                print("Я ТУУУУУУУУУУУУУУУТ")

                response = await wait_for_result(
                    channel, result_queue_id, correlation_id
                )
                print("РЕЗУЛЬТАТ !!!!!!!! ", response)

                if not response:
                    return {"message": "No result returned from the prediction process"}
                elif "error" in response:
                    return {"message": response["error"]}
                else:
                    print(f"Так вот он же я !!! Prediction result: {response}")
                    return response

        except Exception as e:
            logger.error(f"Error processing prediction: {e}")
            return {"message": "Error processing prediction"}


async def wait_for_result(
    channel, result_queue_id, correlation_id, timeout=60, max_retries=5
):
    result_queue_obj = await channel.declare_queue(result_queue_id, auto_delete=True)

    future = asyncio.get_event_loop().create_future()

    async def callback(message: aio_pika.IncomingMessage):
        if message.correlation_id == correlation_id:
            response = json.loads(message.body)

            if "error" in response:
                print(f"Error in response: {response['error']}")
                future.set_result(response)
            else:
                print(f"Response: {response['prediction']}")
                future.set_result(response)

            await message.ack()

    consumer_id = await result_queue_obj.consume(callback)

    try:
        feature_response = await asyncio.wait_for(future, timeout=timeout)
        return feature_response
    except asyncio.TimeoutError:
        logger.error(
            f"Timeout waiting for response with correlation_id {correlation_id}"
        )
        return None
    except Exception as wait_error:
        logger.error(
            f"Error while waiting for response with correlation_id {correlation_id}: {wait_error}"
        )
        return None
    finally:
        try:
            await result_queue_obj.cancel(consumer_id)
        except Exception as cancel_error:
            logger.error(
                f"Error cancelling consumer {consumer_id} for queue {result_queue_id}: {cancel_error}"
            )
        try:
            await result_queue_obj.delete(if_unused=False, if_empty=False)
        except Exception as delete_error:
            logger.error(
                f"Error deleting queue {result_queue_id} with correlation_id {correlation_id}: {delete_error}"
            )


@user_route.get("/history", response_model=PredictionHistoryResponse)
async def get_prediction_history(current_user: dict = Depends(get_current_user)):
    async with get_db_session() as session:
        try:
            predictions = await session.execute(
                select(TablePredictions)
                .filter_by(user_id=current_user["user_id"])
                .order_by(TablePredictions.prediction_id.desc())
            )
            predictions = predictions.scalars().all()

            if not predictions:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="No predictions found"
                )

            return {"predictions": predictions}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )
