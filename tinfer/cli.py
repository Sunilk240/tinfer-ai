"""
CLI entry points for Tinfer commands.

These are thin wrappers that forward all arguments to the correct executables.
The engine is located in either:
  1. ~/.tinfer/bin/ (downloaded via 'tinfer setup')
  2. tinfer/bin/ (bundled in the package, for backwards compatibility)
"""

import os
import sys
import json
import platform
import subprocess


def _get_engine_dir():
    """
    Find the directory containing the tinfer executables.
    Priority:
      1. TINFER_ENGINE_PATH environment variable (user override)
      2. ~/.tinfer/bin/ (downloaded via setup)
      3. tinfer/bin/ inside the package (legacy/bundled)
    """
    # 1. Environment variable override
    env_path = os.environ.get("TINFER_ENGINE_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path

    # 2. Downloaded engine at ~/.tinfer/bin/
    home_engine = os.path.join(os.path.expanduser("~"), ".tinfer", "bin")
    if os.path.isdir(home_engine) and os.listdir(home_engine):
        return home_engine

    # 3. Bundled engine inside the package
    bundled = os.path.join(os.path.dirname(__file__), "bin")
    if os.path.isdir(bundled) and os.listdir(bundled):
        return bundled

    return None


def _get_binary_name(base_name):
    """Get the platform-appropriate binary name."""
    if platform.system() == "Windows":
        return f"{base_name}.exe"
    else:
        return base_name


def _is_cuda_engine(engine_dir):
    """Check if the installed engine is a CUDA build."""
    # Check metadata file saved by 'tinfer setup'
    metadata_path = os.path.join(engine_dir, ".tinfer-metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return "cuda" in metadata.get("target", "")
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: check for CUDA DLLs/shared libs
    for f in os.listdir(engine_dir):
        if "cuda" in f.lower():
            return True

    return False


def _is_metal_engine(engine_dir):
    """Check if the installed engine is a Metal (macOS) build."""
    metadata_path = os.path.join(engine_dir, ".tinfer-metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return "metal" in metadata.get("target", "")
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: check for Metal shader files
    for f in os.listdir(engine_dir):
        if f.endswith(".metal"):
            return True

    return False


def _run_binary(base_name):
    """Run the correct binary with all CLI arguments forwarded."""
    engine_dir = _get_engine_dir()

    if engine_dir is None:
        print(
            "\n"
            "[Tinfer] Inference engine not found!\n"
            "\n"
            "  Run this command to download the engine:\n"
            "    tinfer-setup\n"
            "\n"
            "  Or set TINFER_ENGINE_PATH to a custom directory.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    binary_name = _get_binary_name(base_name)
    exe_path = os.path.join(engine_dir, binary_name)

    if not os.path.exists(exe_path):
        print(
            f"[Tinfer] Error: {binary_name} not found at {engine_dir}\n"
            f"[Tinfer] Run 'tinfer-setup' to reinstall the engine.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build arguments
    args = [exe_path] + sys.argv[1:]

    # Auto-detect GPU layers for the main CLI and server (not bench/quantize)
    if base_name in ("tinfer", "tinfer-server"):
        # Only inject -ngl if the user hasn't specified it
        user_args_str = " ".join(sys.argv[1:])
        if "-ngl" not in user_args_str and "--n-gpu-layers" not in user_args_str:
            if _is_cuda_engine(engine_dir) or _is_metal_engine(engine_dir):
                args.extend(["-ngl", "99"])

    # Set library path so shared libs can be found
    env = os.environ.copy()
    if platform.system() == "Linux":
        ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{engine_dir}:{ld_path}"
    elif platform.system() == "Darwin":
        dyld_path = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = f"{engine_dir}:{dyld_path}"

    # Run the binary
    try:
        result = subprocess.run(args, cwd=os.getcwd(), env=env)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"[Tinfer] Error running {binary_name}: {e}", file=sys.stderr)
        sys.exit(1)


# ============================================================
# CLI Entry Points
# ============================================================
def main_cli():
    """Entry point for 'tinfer' command."""
    _run_binary("tinfer")


def main_server():
    """Entry point for 'tinfer-server' command."""
    _run_binary("tinfer-server")


def main_bench():
    """Entry point for 'tinfer-bench' command."""
    _run_binary("tinfer-bench")


def main_quantize():
    """Entry point for 'tinfer-quantize' command."""
    _run_binary("tinfer-quantize")


def main_setup():
    """Entry point for 'tinfer-setup' command."""
    from tinfer.setup_engine import run_setup
    run_setup()


if __name__ == "__main__":
    main_cli()
