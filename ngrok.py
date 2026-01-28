#!/usr/bin/env python3
"""
ngrok.py - Start FastAPI app with Uvicorn and ngrok (v3) reliably
Features:
 - Detects Python file containing 'app = FastAPI()' automatically
 - Starts uvicorn first, then ngrok on the same port
 - Optional authtoken from NGROK_AUTHTOKEN env var
 - Polls ngrok API until public_url appears
 - Clean shutdown on Ctrl+C (Windows/Unix)
Usage:
  python ngrok.py --port 8000
Environment:
  NGROK_AUTHTOKEN - optional ngrok authtoken
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import requests

DEFAULT_PORT = 8000
POLL_RETRIES = 20
POLL_INTERVAL = 0.5  # seconds

# ---------------- Helper Functions ---------------- #

def run_cmd(cmd, check=True, capture_output=True, text=True):
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=text)

def is_ngrok_available() -> bool:
    try:
        res = run_cmd(["ngrok", "version"])
        print("✓ ngrok:", res.stdout.strip().splitlines()[0])
        return True
    except FileNotFoundError:
        print("✗ ngrok not found. Install ngrok and add to PATH: https://ngrok.com/download")
        return False
    except Exception as e:
        print("✗ Error checking ngrok:", e)
        return False

def configure_ngrok_authtoken_if_provided():
    token = os.environ.get("NGROK_AUTHTOKEN") or os.environ.get("NGROK_AUTH_TOKEN")
    if not token:
        return False
    try:
        print("Configuring ngrok authtoken from NGROK_AUTHTOKEN environment variable...")
        run_cmd(["ngrok", "config", "add-authtoken", token])
        print("✓ ngrok authtoken configured")
        return True
    except Exception as e:
        print("✗ Failed to configure authtoken automatically:", e)
        return False



def start_uvicorn(port: int, host: str = "0.0.0.0") -> subprocess.Popen:
    module_name = "server"
    if not module_name:
        print("✗ Could not find a Python file with 'app = FastAPI()' in this folder")
        sys.exit(1)
    cmd = [sys.executable, "-m", "uvicorn", f"{module_name}:app", "--host", host, "--port", str(port)]
    print(f"Starting Uvicorn: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(1.0)  # give uvicorn a moment to start
    return proc

def start_ngrok(port: int) -> subprocess.Popen:
    cmd = ["ngrok", "http", str(port), "--log", "stdout"]
    print(f"Starting ngrok: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def get_ngrok_tunnels_json() -> tuple[dict | None, str | None]:
    try:
        res = requests.get("http://127.0.0.1:4040/api/tunnels")
        if res.status_code != 200:
            return None, f"HTTP {res.status_code}"
        return res.json(), None
    except Exception as e:
        return None, str(e)

def wait_for_public_url(retries: int = POLL_RETRIES, interval: float = POLL_INTERVAL) -> Optional[str]:
    for i in range(retries):
        parsed, debug = get_ngrok_tunnels_json()
        if parsed and isinstance(parsed, dict):
            tunnels = parsed.get("tunnels") or []
            for t in tunnels:
                public_url = t.get("public_url")
                if public_url:
                    return public_url
        if i == 0 or i == retries-1:
            print(f"Waiting for ngrok tunnel... (attempt {i+1}/{retries})")
            if debug:
                short = debug if len(str(debug)) < 400 else str(debug)[:400]+"..."
                print("  ngrok API response (non-json or empty):", short)
        time.sleep(interval)
    return None

def terminate_process(proc: subprocess.Popen, name: str):
    if not proc:
        return
    try:
        if proc.poll() is None:
            print(f"Stopping {name} (pid={proc.pid})...")
            if platform.system() == "Windows":
                try:
                    subprocess.run(["taskkill","/F","/T","/PID",str(proc.pid)], check=False, capture_output=True)
                except Exception:
                    proc.terminate()
            else:
                proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception as e:
        print(f"Error while terminating {name}: {e}")

# ---------------- Main ---------------- #

def main():
    parser = argparse.ArgumentParser(description="Start FastAPI app + ngrok tunnel")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT, help=f"Port (default {DEFAULT_PORT})")
    parser.add_argument("--host", "-H", type=str, default="0.0.0.0", help="Host for uvicorn")
    parser.add_argument("--no-autoconfig-token", action="store_true", help="Do not auto-config NGROK_AUTHTOKEN")
    args = parser.parse_args()

    if not is_ngrok_available():
        sys.exit(1)

    if not args.no_autoconfig_token:
        configure_ngrok_authtoken_if_provided()

    uvicorn_proc = None
    ngrok_proc = None
    try:
        uvicorn_proc = start_uvicorn(args.port, args.host)
        print(f"✓ Uvicorn started (pid={uvicorn_proc.pid})")

        ngrok_proc = start_ngrok(args.port)

        public_url = wait_for_public_url()
        if public_url:
            print("\n" + "="*60)
            print(f"Local:  http://{args.host}:{args.port}")
            print(f"Public: {public_url}")
            print("="*60)
            print("Press Ctrl+C to stop both processes.")
        else:
            print("✗ Failed to obtain ngrok public URL after retries")
            raise SystemExit(1)

        while True:
            time.sleep(0.5)
            if uvicorn_proc.poll() is not None:
                print("✗ Uvicorn exited.")
                break
            if ngrok_proc.poll() is not None:
                print("✗ ngrok exited.")
                break

    except KeyboardInterrupt:
        print("\nCtrl+C detected — shutting down")
    finally:
        terminate_process(ngrok_proc, "ngrok")
        terminate_process(uvicorn_proc, "uvicorn")
        print("All stopped. Bye.")

if __name__ == "__main__":
    main()
