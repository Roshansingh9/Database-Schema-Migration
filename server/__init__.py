from server.app import app  # re-export so `uvicorn server:app` resolves correctly

__all__ = ["app"]
