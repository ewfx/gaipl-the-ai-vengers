import time


# ✅ Mock functions to simulate server tasks
def mock_restart_server(server_name: str):
    return f"✅ Server {server_name} has been restarted successfully!"

def mock_get_server_status(server_name: str):
    return f"ℹ️ The status of server {server_name} is: Running."