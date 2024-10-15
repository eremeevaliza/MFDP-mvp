from pydantic import BaseModel
from typing import Dict, List, Optional


class User(BaseModel):
    email: str
    password: str


class AdminUserModel(BaseModel):
    user_email: str
    password: str
    user_role: str


class UserDeleteModel(BaseModel):
    user_email: str


class EmailUser(BaseModel):
    user_email: str


class AdminDB(BaseModel):
    admin_email: str


class BalanceRequest(BaseModel):
    amount: float


class BalanceResponse(BaseModel):
    message: str
    new_balance: float


class GetBalanceRequest(BaseModel):
    email: str


class GetBalanceResponse(BaseModel):
    balance: float


class PredictionData(BaseModel):
    data: Dict[str, float]


class PredictionRequest(BaseModel):
    data_id: int
    user_id: int
    email: str


class UserCreateRequest(BaseModel):
    admin_email: str
    email: str
    password: str
    role: str


class UserDeleteRequest(BaseModel):
    admin_email: str
    email: str


class AddBalanceRequest(BaseModel):
    admin_email: str
    email: str
    amount: float


class UsersResponse(BaseModel):
    users: List[User]


class Prediction(BaseModel):
    data_id: int
    prediction: float


class PredictionsResponse(BaseModel):
    predictions: List[Prediction]


class Transaction(BaseModel):
    transaction_id: int
    amount: float


class TransactionsResponse(BaseModel):
    transactions: List[Transaction]


class DataModel(BaseModel):
    email: str
    user_id: int
    data: dict


class PredictionRequestModel(BaseModel):
    data_id: int


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


class TransactionRecord(BaseModel):
    transaction_id: int
    user_id: int
    amount: float
    source: str
    transaction_type: str
    prediction_id: Optional[int] = None


class TransactionHistoryResponse(BaseModel):
    transactions: List[TransactionRecord]


class PredictionRecord(BaseModel):
    prediction_id: int
    email: str
    prediction: float


class PredictionHistoryResponse(BaseModel):
    predictions: List[PredictionRecord]


class AddBalanceModel(BaseModel):
    user_email: str
    amount: float
