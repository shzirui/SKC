import os
import requests
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

def _get_base_url(base_url: Optional[str] = None) -> str:
    """
    Get the base URL for the backend Docker service.
    Priority: function parameter > OSWORLD_BASE_URL environment variable > default localhost.
    """
    url = base_url or os.getenv("OSWORLD_BASE_URL") or "http://localhost:50003"
    return url.rstrip("/")


def _auth_headers(token: Optional[str]) -> Dict[str, str]:
    """
    Generate request headers with Bearer Token (returns empty dict if token is None).
    """
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def list_emulators_by_token(token: str, base_url: Optional[str] = None, detail: bool = False) -> Any:
    """
    Query all virtual machines (emulator_id) under a specific token.
    - detail=False: Return list of emulator_id
    - detail=True: Return list of dictionaries with detailed information (including ports, container_id, resources, etc.)

    Notes:
    - Use /emulators endpoint to get all instances, then filter by token on client side (lightweight, no resources)
    - Use /emulator_resources endpoint to get all instance resource information at once (includes container_id), then filter by token
    """
    url = _get_base_url(base_url)
    try:
        if detail:
            # Include container_id, resources and other information
            resp = requests.get(f"{url}/emulator_resources", headers=_auth_headers(token), timeout=15)
            resp.raise_for_status()
            items = resp.json()
            return [item for item in items if item.get("token") == token]
        else:
            # Only get ID list (lightweight)
            resp = requests.get(f"{url}/emulators", headers=_auth_headers(token), timeout=10)
            resp.raise_for_status()
            items = resp.json()
            return [i.get("emulator_id") for i in items if i.get("token") == token]
    except Exception as e:
        return {"error": str(e)}


def get_emulator_status(emulator_id: str, base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Query the status of a specific virtual machine.
    Current server convention:
    - If /emulator_resources/<id> returns 200, consider status='running'
    - If returns 404, consider status='not_found'

    Return example:
    {
        "emulator_id": "...",
        "status": "running" | "not_found" | "error",
        "token": "...",
        "container_id": "...",
        "ports": {...},
        "duration_minutes": 3,
        "resources": {...} | {"error": "..."} | None,
        "error": "..." (only when error occurs)
    }
    """
    url = _get_base_url(base_url)
    try:
        r = requests.get(f"{url}/emulator_resources/{emulator_id}", timeout=15)
        if r.status_code == 404:
            return {"emulator_id": emulator_id, "status": "not_found"}
        r.raise_for_status()
        data = r.json()
        return {
            "emulator_id": emulator_id,
            "status": "running",
            "token": data.get("token"),
            "container_id": data.get("container_id"),
            "ports": {
                "server_port": data.get("server_port"),
                "vnc_port": data.get("vnc_port"),
                "chromium_port": data.get("chromium_port"),
                "vlc_port": data.get("vlc_port"),
            },
            "duration_minutes": data.get("duration_minutes"),
            "resources": data.get("resources"),
        }
    except Exception as e:
        return {"emulator_id": emulator_id, "status": "error", "error": str(e)}


def stop_emulator(emulator_id: str, token: Optional[str] = None, base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Stop a specific virtual machine (given emulator_id).
    - Use /stop_emulator endpoint, pass JSON: {"emulator_id": "..."} as required.
    - If server has REQUIRE_TOKEN enabled, Authorization: Bearer <token> header can be included.

    Return example:
    {
        "emulator_id": "...",
        "stopped": True | False,
        "response": {...} | None,
        "message": "...",        # Only in certain error/404 cases
        "error": "..."           # Only when request exception occurs
    }
    """
    url = _get_base_url(base_url)
    try:
        r = requests.post(
            f"{url}/stop_emulator",
            json={"emulator_id": emulator_id},
            headers=_auth_headers(token),
            timeout=20,
        )
        if r.status_code == 404:
            return {"emulator_id": emulator_id, "stopped": False, "message": "not found"}
        # Expect backend to return {"code": 0} for success
        data = r.json()
        return {
            "emulator_id": emulator_id,
            "stopped": data.get("code") == 0,
            "response": data,
        }
    except Exception as e:
        return {"emulator_id": emulator_id, "stopped": False, "error": str(e)}

def clean_env():
    # Read environment variables
    token = os.environ.get("OSWORLD_TOKEN")
    base_url = os.environ.get("OSWORLD_BASE_URL", "http://localhost:50003")
    print(f"[env_test] base_url={base_url}, token={token}")
    
    try:
        # List all virtual machines under this token (IDs only)
        urls = os.environ.get("OSWORLD_BASE_URL").split(',')
        for url in urls:
            base_url = url.strip()
            print(f"[env_test] base_url={base_url}")
            print("\n[env_test] List emulators by token (detail=False):")
            emulators = list_emulators_by_token(token, base_url, detail=False)
            print(emulators)
            print(f"Number of emulators: {len(emulators)}")

            # Stop each emulator
            for emu_id in list_emulators_by_token(token, base_url, detail=False):
                print("\n[env_test] Stop emulator:")
                stop_res = stop_emulator(emu_id, token, base_url)
                print(stop_res)

        # List emulators again after stopping
        for url in urls:
            base_url = url.strip()
            print(f"[env_test] base_url={base_url}")
            print("\n[env_test] List emulators after stop (detail=False):")
            print(list_emulators_by_token(token, base_url, detail=False))

    finally:
        print('finish')