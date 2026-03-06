"""
Tinfer Client — OpenAI-compatible HTTP client.

Usage:
    from tinfer import chat, complete, models

    # Chat (OpenAI-compatible)
    response = chat("What is AI?")
    print(response)

    # Chat with full control
    response = chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is AI?"}
        ],
        max_tokens=200,
        temperature=0.7
    )

    # List models
    for model in models():
        print(model)
"""

import requests

# Default server URL
DEFAULT_BASE_URL = "http://127.0.0.1:8080"

_base_url = DEFAULT_BASE_URL


def set_server(url="http://127.0.0.1:8080"):
    """
    Set the server URL for all client functions.

    Args:
        url: Server base URL (default: http://127.0.0.1:8080)
    """
    global _base_url
    _base_url = url.rstrip("/")


def chat(messages=None, prompt=None, model=None, max_tokens=512,
         temperature=0.7, top_p=0.95, stream=False, base_url=None, **kwargs):
    """
    Send a chat completion request to the Tinfer server.

    Args:
        messages: List of message dicts [{"role": "user", "content": "..."}].
                  If a string is passed, it's wrapped as a user message.
        prompt: Shorthand for a single user message (alternative to messages).
        model: Model name (optional, server uses loaded model).
        max_tokens: Maximum tokens to generate (default: 512).
        temperature: Sampling temperature (default: 0.7).
        top_p: Top-p sampling (default: 0.95).
        stream: Whether to stream the response (default: False).
        base_url: Override server URL for this request.
        **kwargs: Additional parameters passed to the API.

    Returns:
        str: The assistant's response text.

    Example:
        >>> chat("What is AI?")
        'AI stands for Artificial Intelligence...'

        >>> chat([
        ...     {"role": "system", "content": "Be concise."},
        ...     {"role": "user", "content": "What is AI?"}
        ... ])
    """
    url = (base_url or _base_url) + "/v1/chat/completions"

    # Handle convenience inputs
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    elif prompt is not None:
        messages = [{"role": "user", "content": prompt}]
    elif messages is None:
        raise ValueError("Either 'messages' or 'prompt' must be provided")

    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
        **kwargs
    }

    if model:
        payload["model"] = model

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if stream:
            return data  # Return raw for streaming

        return data["choices"][0]["message"]["content"]

    except requests.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Tinfer server. "
            "Make sure tinfer-server is running:\n"
            "  tinfer-server -m <model.gguf> --port 8080"
        )
    except requests.HTTPError as e:
        raise RuntimeError(f"Server error: {e.response.status_code} - {e.response.text}")


def complete(prompt, model=None, max_tokens=512, temperature=0.7,
             top_p=0.95, base_url=None, **kwargs):
    """
    Send a text completion request (non-chat format).

    Args:
        prompt: The text prompt to complete.
        model: Model name (optional).
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        base_url: Override server URL.
        **kwargs: Additional parameters.

    Returns:
        str: The completion text.
    """
    url = (base_url or _base_url) + "/v1/completions"

    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        **kwargs
    }

    if model:
        payload["model"] = model

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["text"]
    except requests.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Tinfer server. "
            "Make sure tinfer-server is running."
        )


def models(base_url=None):
    """
    List available models on the server.

    Returns:
        list: List of model info dicts.
    """
    url = (base_url or _base_url) + "/v1/models"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except requests.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Tinfer server. "
            "Make sure tinfer-server is running."
        )
