"""
Tinfer Setup Command — Downloads the correct inference engine for the user's system.

Detects OS, CPU architecture, and GPU availability, then downloads the matching
pre-compiled binary bundle from GitHub Releases to ~/.tinfer/bin/.
"""

import os
import sys
import platform
import subprocess
import shutil
import zipfile
import tarfile
import urllib.request
import json


# ============================================================
# Configuration
# ============================================================
TINFER_VERSION = "0.2.0"
GITHUB_REPO = "Sunilk240/tinfer-ai"
GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases/download"

# Where to store the downloaded engine
def get_engine_dir():
    """Get the engine directory: ~/.tinfer/bin/"""
    return os.path.join(os.path.expanduser("~"), ".tinfer", "bin")


# ============================================================
# System Detection
# ============================================================
def detect_os():
    """Detect the operating system."""
    system = platform.system()
    if system == "Windows":
        return "win"
    elif system == "Linux":
        return "linux"
    elif system == "Darwin":
        return "macos"
    else:
        return None


def detect_arch():
    """Detect CPU architecture."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x64"
    elif machine in ("arm64", "aarch64"):
        return "arm64"
    else:
        return None


def detect_nvidia_gpu():
    """Check if an NVIDIA GPU is available."""
    # Method 1: Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 2: Try loading CUDA driver via ctypes
    try:
        import ctypes
        if platform.system() == "Windows":
            ctypes.CDLL("nvcuda.dll")
            return "NVIDIA GPU (detected via driver)"
        else:
            ctypes.CDLL("libcuda.so")
            return "NVIDIA GPU (detected via driver)"
    except (OSError, ImportError):
        pass

    return None


def determine_bundle_name(version):
    """Determine the correct bundle filename for this system."""
    os_name = detect_os()
    arch = detect_arch()

    if os_name is None:
        print(f"[Tinfer] Error: Unsupported operating system: {platform.system()}")
        sys.exit(1)

    if arch is None:
        print(f"[Tinfer] Error: Unsupported CPU architecture: {platform.machine()}")
        sys.exit(1)

    gpu_name = detect_nvidia_gpu()

    # Determine the target
    if os_name == "macos" and arch == "arm64":
        target = "macos-arm64-metal"
        ext = "tar.gz"
        gpu_info = "Apple Metal GPU"
    elif os_name == "macos" and arch == "x64":
        target = "macos-x64-cpu"
        ext = "tar.gz"
        gpu_info = "CPU only (Intel Mac)"
    elif os_name == "linux" and arch == "arm64":
        target = "linux-arm64-cpu"
        ext = "tar.gz"
        gpu_info = "CPU only (ARM64)"
    elif gpu_name:
        target = f"{os_name}-x64-cuda"
        ext = "zip" if os_name == "win" else "tar.gz"
        gpu_info = f"NVIDIA CUDA ({gpu_name})"
    else:
        target = f"{os_name}-x64-cpu"
        ext = "zip" if os_name == "win" else "tar.gz"
        gpu_info = "CPU only (no NVIDIA GPU detected)"

    bundle_name = f"tinfer-v{version}-{target}.{ext}"
    return bundle_name, target, gpu_info


# ============================================================
# Download & Extract
# ============================================================
def download_with_progress(url, dest_path):
    """Download a file with a simple progress bar."""
    print(f"[Tinfer] Downloading from: {url}")

    try:
        response = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        print(f"[Tinfer] Error: Download failed (HTTP {e.code})")
        print(f"[Tinfer] URL: {url}")
        print(f"[Tinfer] This version may not have been released yet.")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"[Tinfer] Error: Could not connect to GitHub. Check your internet connection.")
        print(f"[Tinfer] Details: {e.reason}")
        sys.exit(1)

    total_size = int(response.headers.get("Content-Length", 0))
    downloaded = 0
    block_size = 8192

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    with open(dest_path, "wb") as f:
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            if total_size > 0:
                percent = downloaded / total_size * 100
                bar_len = 40
                filled = int(bar_len * downloaded // total_size)
                bar = "#" * filled + "-" * (bar_len - filled)
                size_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\r[{bar}] {percent:.0f}% ({size_mb:.1f}/{total_mb:.1f} MB)")
                sys.stdout.flush()

    print()  # newline after progress bar
    return dest_path


def extract_bundle(archive_path, dest_dir):
    """Extract a ZIP or TAR.GZ archive to the destination directory."""
    # Clean existing files in destination
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif archive_path.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest_dir)
    else:
        print(f"[Tinfer] Error: Unknown archive format: {archive_path}")
        sys.exit(1)

    # Make executables runnable on Linux/macOS
    if platform.system() != "Windows":
        for f in os.listdir(dest_dir):
            filepath = os.path.join(dest_dir, f)
            if os.path.isfile(filepath) and not f.endswith((".so", ".dylib", ".metal")):
                os.chmod(filepath, 0o755)


# ============================================================
# Setup Command
# ============================================================
def run_setup():
    """Main setup function — detects system and downloads the correct engine."""
    print()
    print("=" * 55)
    print("  Tinfer Setup — Inference Engine Installer")
    print("=" * 55)
    print()

    # Detect system
    os_name = detect_os()
    arch = detect_arch()
    gpu_name = detect_nvidia_gpu()

    print(f"[Tinfer] OS:   {platform.system()} ({os_name})")
    print(f"[Tinfer] Arch: {platform.machine()} ({arch})")
    if gpu_name:
        print(f"[Tinfer] GPU:  {gpu_name}")
    else:
        print(f"[Tinfer] GPU:  Not detected (will use CPU-only engine)")
    print()

    # Determine what to download
    bundle_name, target, gpu_info = determine_bundle_name(TINFER_VERSION)
    download_url = f"{GITHUB_RELEASES_URL}/v{TINFER_VERSION}/{bundle_name}"
    engine_dir = get_engine_dir()

    print(f"[Tinfer] Selected Engine: {target}")
    print(f"[Tinfer] Acceleration:    {gpu_info}")
    print(f"[Tinfer] Install Path:    {engine_dir}")
    print()

    # Check if already installed
    if os.path.exists(engine_dir) and os.listdir(engine_dir):
        print(f"[Tinfer] Engine already exists at {engine_dir}")
        response = input("[Tinfer] Reinstall? (y/N): ").strip().lower()
        if response != "y":
            print("[Tinfer] Setup cancelled.")
            return

    # Download
    temp_dir = os.path.join(os.path.expanduser("~"), ".tinfer", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    archive_path = os.path.join(temp_dir, bundle_name)

    download_with_progress(download_url, archive_path)

    # Extract
    print(f"[Tinfer] Extracting to {engine_dir}...")
    extract_bundle(archive_path, engine_dir)

    # Save metadata (which variant is installed)
    metadata = {
        "version": TINFER_VERSION,
        "target": target,
        "gpu_info": gpu_info,
        "os": platform.system(),
        "arch": platform.machine(),
    }
    metadata_path = os.path.join(engine_dir, ".tinfer-metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Cleanup temp
    shutil.rmtree(temp_dir, ignore_errors=True)

    print()
    print("[Tinfer] Done! Setup complete!")
    print()
    print("  You can now run:")
    print("    tinfer -m path/to/model.gguf -p \"Hello!\"")
    print("    tinfer-server -m path/to/model.gguf --port 8080")
    print()


if __name__ == "__main__":
    run_setup()
