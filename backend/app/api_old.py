import uvicorn
from fastapi import FastAPI
from routes_old.admin import admin_route  # type: ignore
from routes_old.user import user_route  # type: ignore
from routes_old.transactions import transaction_route  # type: ignore
from common.db.db_api import init_db


app = FastAPI()

app.include_router(user_route, prefix="/user")
app.include_router(admin_route, prefix="/superadmin")
app.include_router(transaction_route, prefix="/transaction")


@app.on_event("startup")
def on_startup():
    print("on_startup() > before init_db()")
    init_db()
    print("on_startup() > after init_db()")


@app.get("/")
async def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
