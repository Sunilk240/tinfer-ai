"""
Tinfer — Tiny Inference Engine

Run LLMs locally with GPU acceleration.

Usage:
    # CLI
    tinfer -m model.gguf -p "Hello"

    # Server
    tinfer-server -m model.gguf --port 8080

    # Python API
    from tinfer import Server, chat
    with Server("model.gguf", port=8080) as s:
        response = chat("Hello, what is AI?")
        print(response)
"""

__version__ = "0.2.0"
__author__ = "Sunil"

from tinfer.server import Server
from tinfer.client import chat, complete, models

__all__ = ["Server", "chat", "complete", "models", "__version__"]
