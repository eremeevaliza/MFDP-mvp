# streamlit_app.py

from urllib.parse import urlencode, urlparse
import requests
import streamlit as st

st.set_page_config(
    page_title="Log in",
    page_icon="🗝️",
)


def get_query_params():
    return st.experimental_get_query_params()


# Получаем параметры из URL
query_params = get_query_params()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "token" not in st.session_state:
    st.session_state.token = None

API_URL = "http://backend:8000"  # Replace with your backend API URL

# Steam API details
STEAM_API_KEY = "D9BA287D5D3668490B52A82727BE5DB6"  # Replace with your Steam API key
STEAM_OPENID_URL = "https://steamcommunity.com/openid/login"

claimed_id = query_params.get("openid.claimed_id", [None])[
    0
]  # Если открываем страницу в первый раз, то токена нет, если редирект со Steam, то токен есть

# Steam OpenID настройки
STEAM_OPENID_URL = "https://steamcommunity.com/openid/login"
PUBLIC_ADDR = "https://03af-2600-1700-1690-19a0-41a1-ecfc-4b70-df3b.ngrok-free.app"
PUBLIC_OAUTH = f"{PUBLIC_ADDR}"


def clear_query_params():
    # clear query params in streamlit
    st.experimental_set_query_params()


if claimed_id and not st.session_state.logged_in:
    if query_params:

        # Передаем параметры на бэкэнд для валидации
        validate_url = f"{API_URL}/oauth_validate"  # Ссылка на твой FastAPI контейнер
        response = requests.post(validate_url, json=query_params)

        if response.status_code == 200:
            st.session_state.logged_in = True
            clear_query_params()
            # Получаем JWT токен от бэкенда и сохраняем в сессии
            token = response.json().get("access_token")
            st.session_state["token"] = token
            st.write("Вы успешно вошли через Steam!")
        else:
            st.session_state.logged_in = False
            st.error("Ошибка валидации через Steam")

    # Здесь вы можете сохранить токен в сессии или использовать его для дальнейших запросов
else:
    params = {
        "openid.ns": "http://specs.openid.net/auth/2.0",
        "openid.mode": "checkid_setup",
        "openid.return_to": PUBLIC_OAUTH,
        "openid.realm": PUBLIC_ADDR,
        "openid.identity": "http://specs.openid.net/auth/2.0/identifier_select",
        "openid.claimed_id": "http://specs.openid.net/auth/2.0/identifier_select",
    }
    query_string = urlencode(params)
    redirect_url = f"{STEAM_OPENID_URL}?{query_string}"
    st.markdown(f"[Войти через Steam]({redirect_url})", unsafe_allow_html=True)

if st.session_state.logged_in:
    st.write("Вы авторизованы через Steam!")
    # Отображаем кнопку "Получить бандл"
    if st.button("Получить бандл"):
        # Вызываем эндпоинт для получения бандла
        headers = {"Authorization": f"Bearer {st.session_state['token']}"}
        st.write("Запрашиваем бандл...")
        recommend_url = f"{API_URL}/recommend_bundle"
        response = requests.post(recommend_url, headers=headers)
        st.write("Ответ от сервера:")

        if response.status_code == 200:
            bundle = response.json()
            st.subheader(f"Цена бандла: ${bundle['bundle_price']:.2f}")
            st.write("Рекомендуемые игры:")
            for game in bundle["games"]:
                st.image(game["icon"], width=100)
                st.write(f"**{game['name']}**")
        else:
            st.error("Ошибка при получении бандла")
else:
    st.write("Пожалуйста, войдите через Steam, чтобы получить рекомендации.")
