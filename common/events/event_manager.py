class AsyncEventManager:
    def __init__(self):
        self.listeners = {}

    def register_event(self, event_name: str):
        if event_name not in self.listeners:
            self.listeners[event_name] = []

    def add_listener(self, event_name: str, listener):
        if event_name not in self.listeners:
            self.register_event(event_name)
        self.listeners[event_name].append(listener)

    def remove_listener(self, event_name: str, listener):
        if event_name in self.listeners:
            self.listeners[event_name].remove(listener)
            if not self.listeners[event_name]:
                del self.listeners[event_name]

    async def dispatch(self, event_name: str, *args, **kwargs):
        if event_name in self.listeners:
            for listener in self.listeners[event_name]:
                await listener(*args, **kwargs)

    def get_registered_events(self):
        return list(self.listeners.keys())

    def get_listeners(self, event_name: str):
        if event_name in self.listeners:
            return [listener.__name__ for listener in self.listeners[event_name]]
        return []


event_manager = AsyncEventManager()

async def on_user_created(email: str, role: str):
    print(f"User created with email: {email} and role: {role}")

async def on_user_deleted(email: str):
    print(f"User deleted with email: {email}")

async def on_balance_added(email: str, amount: float):
    print(f"Balance added for email: {email} with amount: {amount}")

async def on_users_viewed(admin_email: str):
    print(f"Users viewed by admin: {admin_email}")

async def on_predictions_viewed(admin_email: str):
    print(f"Predictions viewed by admin: {admin_email}")

async def on_transactions_viewed(admin_email: str):
    print(f"Transactions viewed by admin: {admin_email}")

async def on_balance_topped_up(email: str, amount: float, new_balance: float):
    print(f"Balance topped up for {email} with amount {amount}. New balance: {new_balance}")

async def on_balance_withdrawn(email: str, amount: float, new_balance: float):
    print(f"Balance withdrawn for {email} with amount {amount}. New balance: {new_balance}")


event_manager.register_event("user_created")
event_manager.register_event("user_deleted")
event_manager.register_event("balance_added")
event_manager.register_event("users_viewed")
event_manager.register_event("predictions_viewed")
event_manager.register_event("transactions_viewed")
event_manager.register_event("balance_topped_up")
event_manager.register_event("balance_withdrawn")

event_manager.add_listener("user_created", on_user_created)
event_manager.add_listener("user_deleted", on_user_deleted)
event_manager.add_listener("balance_added", on_balance_added)
event_manager.add_listener("users_viewed", on_users_viewed)
event_manager.add_listener("predictions_viewed", on_predictions_viewed)
event_manager.add_listener("transactions_viewed", on_transactions_viewed)
event_manager.add_listener("balance_topped_up", on_balance_topped_up)
event_manager.add_listener("balance_withdrawn", on_balance_withdrawn)


