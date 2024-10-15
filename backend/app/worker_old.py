import os
import json
import logging
import aio_pika
import asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from contextlib import asynccontextmanager
from backend.app.user_code import TableData, TablePredictions
from backend.app.user_code import AsPredictModel as PredictModel, AsTransactionUser
from sqlalchemy.future import select as sql_select
from common.db.db_config import get_settings

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "password")
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_async_engine(
    get_settings().DATABASE_URL_psycopg, echo=True, pool_size=5, max_overflow=10
)
async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def get_db_session():
    session = async_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def main():
    connection = await aio_pika.connect_robust(
        f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@rabbitmq/"
    )
    async with connection:
        channel = await connection.channel()
        data_queue = await channel.declare_queue("data_tasks")
        prediction_queue = await channel.declare_queue("prediction_tasks")
        await channel.declare_queue("prediction_results")

        async def process_data(message: aio_pika.IncomingMessage):
            async with message.process():
                task = json.loads(message.body)
                task["reply_to"] = message.reply_to
                task["correlation_id"] = message.correlation_id
                if task["name"] == "load_data":
                    await handle_data(task, channel)
                elif task["name"] == "get_prediction":
                    await handle_prediction(task, channel)

        await data_queue.consume(process_data)
        await prediction_queue.consume(process_data)

        logger.info(" [*] Waiting for messages. To exit press CTRL+C")
        await asyncio.Future()


async def handle_data(task, channel):
    async with get_db_session() as session:
        try:
            email = task["email"]
            user_id = task["user_id"]
            data = task["data"]
            reply_to = task["reply_to"]
            correlation_id = task["correlation_id"]

            new_data = TableData(email=email, user_id=user_id, data=data)
            session.add(new_data)
            await session.commit()
            logger.debug(
                f"Data processed for user {email} with data id {new_data.data_id}"
            )

            task["data_id"] = new_data.data_id

            logger.debug(
                f"Data loaded for user {email}: {task['data_id']} {task['data']}"
            )
            result = {
                "email": email,
                "user_id": user_id,
                "data_id": new_data.data_id,
                "message": "Data loaded successfully",
            }
            await publish_result(channel, result, correlation_id, reply_to)
            return result

            # This is because telegram worked
            # await handle_prediction(task, channel)
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            await session.rollback()


async def handle_prediction(task, channel):
    async with get_db_session() as session:
        try:
            logger.debug(f"WORKER Task: {task}")
            data_id = task["data_id"]
            email = task["email"]
            user_id = task["user_id"]
            reply_to = task["reply_to"]
            correlation_id = task["correlation_id"]

            logger.debug(f"WORKER Data id: {data_id}")

            prediction_record = await session.execute(
                select(TablePredictions)
                .where(TablePredictions.data_id == data_id)
                .where(TablePredictions.email == email)
            )
            prediction_record = prediction_record.first()
            if prediction_record:
                logger.warning(f"Prediction for data_id {data_id} already exists")
                result = {
                    "email": email,
                    "user_id": user_id,
                    "data_id": data_id,
                    "error": "Prediction already exists",
                }
                await publish_result(channel, result, correlation_id, reply_to)
                return result

            transaction_user = AsTransactionUser(user_id, email, session)

            prediction_cost = 10.0

            current_balance = await transaction_user.get_balance_from_db()
            if current_balance < prediction_cost:
                logger.warning(
                    f"Insufficient funds for user {email}. Balance: {current_balance}, Cost: {prediction_cost}"
                )
                result = {
                    "email": email,
                    "user_id": user_id,
                    "data_id": data_id,
                    "error": "Insufficient funds",
                    "balance": current_balance,
                    "cost": prediction_cost,
                }
                await publish_result(channel, result, correlation_id, reply_to)
                return result

            predictor = PredictModel(
                model={}, session=session, email=email, user_id=user_id
            )
            predictor._data_id = data_id
            prediction = await predictor.predict()
            logger.debug(f"WORKER Prediction: {prediction}")

            prediction_record = await session.execute(
                select(TablePredictions)
                .where(TablePredictions.data_id == data_id)
                .where(TablePredictions.email == email)
            )
            new_prediction = prediction_record.scalar_one_or_none()
            prediction_id = new_prediction.prediction_id if new_prediction else None

            if not prediction_id:
                raise Exception("Prediction was not saved correctly in the database")

            transaction_result = await transaction_user.check_and_create_transaction(
                prediction_id, prediction_cost
            )

            if transaction_result["message"] == "Transaction created successfully":
                await session.commit()
                logger.debug(f"WORKER Prediction committed and payment processed")

                result = {
                    "email": email,
                    "user_id": user_id,
                    "data_id": data_id,
                    "prediction": prediction,
                    "prediction_id": prediction_id,
                    "transaction": transaction_result,
                }
                logger.debug(f"WORKER Result: {result}")
            else:
                await session.rollback()
                logger.error(
                    f"Failed to process payment: {transaction_result['message']}"
                )
                result = {
                    "email": email,
                    "user_id": user_id,
                    "data_id": data_id,
                    "error": "Payment processing failed",
                    "message": transaction_result["message"],
                }

            await publish_result(channel, result, correlation_id, reply_to)

        except Exception as e:
            logger.error(f"Error processing prediction: {e}")
            await session.rollback()

            error_result = {
                "email": email,
                "user_id": user_id,
                "data_id": data_id,
                "error": "Internal server error",
                "message": str(e),
            }
            await publish_result(channel, error_result, correlation_id, reply_to)


async def publish_result(channel, result, correlation_id, reply_to):
    try:
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(result).encode(),
                correlation_id=correlation_id,
            ),
            routing_key=reply_to,
        )
        logger.debug(f"WORKER Result published")
    except Exception as e:
        logger.error(f"Error publishing result: {e}")


if __name__ == "__main__":
    asyncio.run(main())
