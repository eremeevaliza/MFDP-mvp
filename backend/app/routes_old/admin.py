from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from common.db.db_api import get_db_session
from common.models_old.pm import (
    AddBalanceModel,
    AdminUserModel,
    AdminDB,
    EmailUser,
    UserDeleteModel,
)
from backend.app.user_code import TableUser, AsAdminUser
from common.events.event_manager import event_manager
from common.auth.dependencies import get_current_user

admin_route = APIRouter()


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


@admin_route.post("/new_user")
async def new_user(
    data: AdminUserModel,
    current_user: dict = Depends(get_current_user),
):
    async with get_db_session() as session:
        if await get_user_role(current_user["email"], session) != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access forbidden"
            )

        admin_user = AsAdminUser(email=current_user["email"])
        try:
            result = await admin_user.register_with_steam(
                session,
                email=data.user_email,
                password=data.password,
                role=data.user_role,
            )
            await event_manager.dispatch(
                "user_created", email=data.user_email, role=data.user_role
            )
            return {"message": result}
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@admin_route.delete("/user")
async def delete_user(
    data: UserDeleteModel,
    current_user: dict = Depends(get_current_user),
):
    async with get_db_session() as session:
        if await get_user_role(current_user["email"], session) != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access forbidden"
            )

        admin_user = AsAdminUser(email=current_user["email"])
        try:
            result = await admin_user.delete_user(session, email=data.user_email)
            await event_manager.dispatch("user_deleted", email=data.user_email)
            return {"message": result}
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@admin_route.post("/balance")
async def add_balance(
    data: AddBalanceModel,
    current_user: dict = Depends(get_current_user),
):
    async with get_db_session() as session:
        if await get_user_role(current_user["email"], session) != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access forbidden"
            )

        admin_user = AsAdminUser(email=current_user["email"])
        try:
            result = await admin_user.add_balance(
                session, amount=data.amount, email=data.user_email
            )
            await event_manager.dispatch(
                "balance_added", email=data.user_email, amount=data.amount
            )
            return {"message": result}
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@admin_route.get("/db_users")
async def view_users(
    current_user: dict = Depends(get_current_user),
):
    async with get_db_session() as session:
        if await get_user_role(current_user["email"], session) != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access forbidden"
            )

        admin_user = AsAdminUser(email=current_user["email"])
        try:
            users = await admin_user.view_users(session)
            await event_manager.dispatch(
                "users_viewed", admin_email=current_user["email"]
            )
            return users
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )


@admin_route.get("/db_predictions")
async def view_predictions(
    current_user: dict = Depends(get_current_user),
):
    async with get_db_session() as session:
        if await get_user_role(current_user["email"], session) != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access forbidden"
            )

        admin_user = AsAdminUser(email=current_user["email"])
        try:
            predictions = await admin_user.view_predictions(session)
            await event_manager.dispatch(
                "predictions_viewed", admin_email=current_user["email"]
            )
            return predictions
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )


@admin_route.get("/db_transactions")
async def view_transactions(
    current_user: dict = Depends(get_current_user),
):
    async with get_db_session() as session:
        if await get_user_role(current_user["email"], session) != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access forbidden"
            )

        admin_user = AsAdminUser(email=current_user["email"])
        try:
            transactions = await admin_user.view_transactions(session)
            await event_manager.dispatch(
                "transactions_viewed", admin_email=current_user["email"]
            )
            return transactions
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )


@admin_route.get("/db_data")
async def view_data(
    current_user: dict = Depends(get_current_user),
):
    async with get_db_session() as session:
        if await get_user_role(current_user["email"], session) != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access forbidden"
            )

        admin_user = AsAdminUser(email=current_user["email"])
        try:
            data_records = await admin_user.view_data(session)
            await event_manager.dispatch(
                "data_viewed", admin_email=current_user["email"]
            )
            return data_records
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )
