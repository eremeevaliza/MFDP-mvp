import pytest
from fastapi.testclient import TestClient
from app.main import app  # Импортируйте ваше приложение FastAPI

# Инициализация клиента для тестирования
client = TestClient(app)


# Модуль для тестирования асинхронных функций
@pytest.mark.asyncio
async def test_oauth_validate():
    # Пример данных OpenID для тестирования
    openid_params = {
        "openid.ns": "http://specs.openid.net/auth/2.0",
        "openid.op_endpoint": "https://steamcommunity.com/openid/login",
        "openid.claimed_id": "https://steamcommunity.com/openid/id/76561199776892478",  # Ваш Steam ID
        "openid.identity": "https://steamcommunity.com/openid/id/76561199776892478",
        "openid.return_to": "http://example.com",
        "openid.response_nonce": "2024-01-01T00:00:00Z123456",
        "openid.assoc_handle": "1234567890",
        "openid.signed": "op_endpoint,claimed_id,identity,return_to",
        "openid.sig": "signature_here",
    }

    # Отправляем POST-запрос с OpenID данными
    response = client.post("/oauth_validate", json=openid_params)

    # Проверяем успешный статус и наличие токена
    assert response.status_code == 200
    assert "access_token" in response.json()


@pytest.mark.asyncio
async def test_recommend_bundle():
    # Используем статический Steam ID
    steam_id = "76561199776892478"
    token = "your_test_jwt_token"

    # Добавляем токен в заголовок Authorization
    headers = {"Authorization": f"Bearer {token}"}

    # Отправляем запрос на получение рекомендованного бандла
    response = client.post("/recommend_bundle", headers=headers)

    # Проверяем успешный статус и корректность данных
    assert response.status_code == 200
    assert "bundle_price" in response.json()
    assert "games" in response.json()
