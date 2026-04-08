"""
app.py — HuggingFace Spaces entry point.

HF Spaces looks for app.py (Gradio) or server.py (Docker).
This file just delegates to the FastAPI server via uvicorn.
The Dockerfile uses CMD ["python", "server.py"] directly,
but app.py is kept for HF Spaces SDK compatibility.
"""
import os
import uvicorn
from server import app  # noqa: F401 — imported so HF can detect the app

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
