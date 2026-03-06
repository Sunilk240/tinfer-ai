"""
Tinfer Server — Programmatic server management.

Usage:
    from tinfer import Server

    # As context manager (auto start/stop)
    with Server("model.gguf", port=8080, n_gpu_layers=99) as s:
        print(f"Server running on {s.base_url}")
        # Use tinfer.chat(), tinfer.complete() etc.

    # Manual control
    server = Server("model.gguf", port=8080)
    server.start()
    # ... use the server ...
    server.stop()
"""

import os
import sys
import time
import subprocess
import requests


class Server:
    """Manages a tinfer-server process."""

    def __init__(self, model_path, port=8080, host="127.0.0.1",
                 n_gpu_layers=None, ctx_size=None, n_parallel=None,
                 extra_args=None):
        """
        Initialize a Tinfer server configuration.

        Args:
            model_path: Path to the GGUF model file.
            port: Port to listen on (default: 8080).
            host: Host to bind to (default: 127.0.0.1).
            n_gpu_layers: Number of layers to offload to GPU (-1 = all).
            ctx_size: Context window size.
            n_parallel: Number of parallel sequences.
            extra_args: List of additional CLI arguments.
        """
        self.model_path = os.path.abspath(model_path)
        self.port = port
        self.host = host
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.n_parallel = n_parallel
        self.extra_args = extra_args or []
        self.process = None
        self.base_url = f"http://{host}:{port}"

    def _get_server_exe(self):
        """Get path to the bundled tinfer-server executable."""
        bin_dir = os.path.join(os.path.dirname(__file__), "bin")
        return os.path.join(bin_dir, "tinfer-server.exe")

    def _build_args(self):
        """Build the command-line arguments for tinfer-server."""
        args = [self._get_server_exe()]
        args.extend(["-m", self.model_path])
        args.extend(["--port", str(self.port)])
        args.extend(["--host", self.host])

        if self.n_gpu_layers is not None:
            args.extend(["-ngl", str(self.n_gpu_layers)])
        if self.ctx_size is not None:
            args.extend(["-c", str(self.ctx_size)])
        if self.n_parallel is not None:
            args.extend(["-np", str(self.n_parallel)])

        args.extend(self.extra_args)
        return args

    def start(self, timeout=30):
        """
        Start the server and wait for it to be ready.

        Args:
            timeout: Max seconds to wait for server to become healthy.

        Returns:
            self (for chaining)

        Raises:
            RuntimeError: If server fails to start within timeout.
        """
        if self.process and self.process.poll() is None:
            raise RuntimeError("Server is already running")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        args = self._build_args()
        self.process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )

        # Wait for server to become healthy
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=2)
                if resp.status_code == 200:
                    return self
            except requests.ConnectionError:
                pass

            # Check if process died
            if self.process.poll() is not None:
                stderr = self.process.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Server exited with code {self.process.returncode}: {stderr}")

            time.sleep(0.5)

        self.stop()
        raise RuntimeError(f"Server did not become healthy within {timeout}s")

    def stop(self):
        """Stop the server process."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None

    def is_running(self):
        """Check if the server is running and healthy."""
        if not self.process or self.process.poll() is not None:
            return False
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=2)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __repr__(self):
        status = "running" if self.is_running() else "stopped"
        return f"<Server model={os.path.basename(self.model_path)} port={self.port} {status}>"
