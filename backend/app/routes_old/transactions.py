from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from common.db.db_api import get_db_session
from common.models_old.pm import (
    BalanceRequest,
    BalanceResponse,
    GetBalanceRequest,
    GetBalanceResponse,
    TransactionHistoryResponse,
)
from backend.app.user_code import (
    AsTransactionUser as TransactionUser,
    TableUser,
    TableTransactions,
)
from typing import Dict
from common.events.event_manager import event_manager
from common.auth.dependencies import get_current_user

transaction_route = APIRouter()


@transaction_route.post("/topup", response_model=BalanceResponse)
async def topup_balance(
    request: BalanceRequest, current_user: dict = Depends(get_current_user)
):
    async with get_db_session() as session:
        try:
            user = await session.execute(
                select(TableUser).filter_by(
                    email=current_user["email"], user_id=current_user["user_id"]
                )
            )
            user = user.scalar()

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
                )

            transaction_user = TransactionUser(
                user_id=user.user_id, email=user.email, session=session
            )
            print(
                f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>transaction_user: {request.amount}"
            )
            new_balance = await transaction_user.topup_balance(request.amount)
            await event_manager.dispatch(
                "balance_topped_up",
                email=current_user["email"],
                amount=request.amount,
                new_balance=new_balance,
            )
            return {
                "message": "Balance topped up successfully",
                "new_balance": new_balance,
            }
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )


@transaction_route.post("/withdraw", response_model=BalanceResponse)
async def withdraw_balance(
    request: BalanceRequest, current_user: dict = Depends(get_current_user)
):
    async with get_db_session() as session:
        try:
            user = await session.execute(
                select(TableUser).filter_by(
                    email=current_user["email"], user_id=current_user["user_id"]
                )
            )
            user = user.scalar()

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
                )

            transaction_user = TransactionUser(
                user_id=user.user_id, email=user.email, session=session
            )
            new_balance = await transaction_user.withdraw_balance(request.amount)
            await event_manager.dispatch(
                "balance_withdrawn",
                email=current_user["email"],
                amount=request.amount,
                new_balance=new_balance,
            )
            return {
                "message": "Balance withdrawn successfully",
                "new_balance": new_balance,
            }
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )


@transaction_route.get("/balance", response_model=GetBalanceResponse)
async def get_balance(current_user: dict = Depends(get_current_user)):
    async with get_db_session() as session:
        try:
            user = await session.execute(
                select(TableUser).filter_by(
                    email=current_user["email"], user_id=current_user["user_id"]
                )
            )
            user = user.scalar()

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
                )

            transaction_user = TransactionUser(
                user_id=user.user_id, email=user.email, session=session
            )
            balance = await transaction_user.get_balance_from_db()
            return {"balance": balance}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )


@transaction_route.post("/transaction", response_model=BalanceResponse)
async def create_transaction(
    request: BalanceRequest, current_user: dict = Depends(get_current_user)
):
    async with get_db_session() as session:
        try:
            transaction_user = TransactionUser(
                user_id=current_user["user_id"],
                email=current_user["email"],
                session=session,
            )
            result = await transaction_user.check_and_create_transaction(
                0, request.amount  # пока использую фэйковый id предсказания
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )


@transaction_route.get("/history", response_model=TransactionHistoryResponse)
async def get_transaction_history(current_user: dict = Depends(get_current_user)):
    async with get_db_session() as session:
        try:
            transactions = await session.execute(
                select(TableTransactions)
                .filter_by(user_id=current_user["user_id"])
                .order_by(TableTransactions.transaction_id.desc())
            )
            transactions = transactions.scalars().all()

            if not transactions:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No transactions found",
                )

            return {"transactions": transactions}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )
