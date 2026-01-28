# server.py
# FastAPI watch-together — democratic control version (LAN)
# Any participant can control playback
# Запуск:
#   python -m venv venv
#   source venv/bin/activate  # або venv\Scripts\activate на Windows
#   pip install -r requirements.txt
#   uvicorn server:app --host 0.0.0.0 --port 8000

import asyncio
import json
import os
import time
import uuid
from typing import Dict, Any, Set
from urllib.parse import urlparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Статика на /static, а відео на /video
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/video", StaticFiles(directory="static/videos"), name="videos")

# Прості редіректи для зручності
@app.get("/")
def root_redirect():
    return RedirectResponse(url="/static/index.html")
# https://cloud.kodik-cdn.com/animetvseries/3c3d55bd6309ea3b2a6554248ee17fd1c362e516/f2f015a4c6ca6a31fcc2b59a825be945:2026012911/720.mp4
@app.get("/admin")
def admin_redirect():
    return RedirectResponse(url="/static/admin.html")

# API для отримання списку відео
@app.get("/api/videos")
def get_videos():
    """Повертає список доступних відео файлів з папки static/videos."""
    videos_dir = "static/videos"
    allowed_extensions = {'.mp4', '.webm', '.ogg', '.mkv', '.mov'}
    videos = []
    
    try:
        if os.path.exists(videos_dir):
            for filename in sorted(os.listdir(videos_dir)):
                ext = os.path.splitext(filename)[1].lower()
                if ext in allowed_extensions:
                    # Отримуємо розмір файлу
                    filepath = os.path.join(videos_dir, filename)
                    size = os.path.getsize(filepath)
                    size_str = format_file_size(size)
                    
                    videos.append({
                        "name": filename,
                        "url": f"/video/{filename}",
                        "size": size_str,
                        "type": ext[1:]  # без крапки
                    })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to list videos: {str(e)}"}
        )
    
    return {"videos": videos}


def format_file_size(size_bytes: int) -> str:
    """Форматує розмір файлу в читабельний вигляд."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


# Conflict resolution tolerance in seconds
CONFLICT_TOLERANCE_S = 0.05  # 50ms

# Rooms in-memory storage
# Structure: {
#   key: {
#     "participants": Set[WebSocket],
#     "user_ids": Dict[WebSocket, str],  # Map websocket to user_id
#     "nicknames": Dict[str, str],       # Map user_id to nickname
#     "state": {
#       "video": str | None,
#       "time": float,
#       "playing": bool,
#       "last_action_timestamp": float,
#       "last_action_user": str | None,
#       "last_update_time": float  # Server time when state was last updated
#     },
#     "sync_task": asyncio.Task | None  # Background sync task
#   }
# }
rooms: Dict[str, Dict[str, Any]] = {}
rooms_lock = asyncio.Lock()


def generate_user_id() -> str:
    """Generate a unique user ID."""
    return f"u_{uuid.uuid4().hex[:8]}"


def make_room_if_missing(key: str):
    """Create a new room if it doesn't exist."""
    if key not in rooms:
        rooms[key] = {
            "participants": set(),
            "user_ids": {},
            "nicknames": {},
            "state": {
                "video": None,
                "time": 0.0,
                "playing": False,
                "last_action_timestamp": 0.0,
                "last_action_user": None,
                "last_update_time": time.time()
            },
            "sync_task": None
        }


def is_valid_video_link(link: str) -> bool:
    """
    Allow:
      - local paths starting with /video/
      - http(s) links
    """
    if not isinstance(link, str) or not link:
        return False
    if link.startswith("/video/"):
        return True
    parsed = urlparse(link)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def get_current_time(room_state: dict) -> float:
    """Calculate current playback time based on state."""
    if room_state["playing"]:
        elapsed = time.time() - room_state["last_update_time"]
        return room_state["time"] + elapsed
    return room_state["time"]


async def broadcast_to_room(key: str, message: dict, exclude_ws: WebSocket = None):
    """Send JSON message to all participants in room (best-effort)."""
    if key not in rooms:
        return
    
    dead: Set[WebSocket] = set()
    text = json.dumps(message)
    
    for ws in list(rooms[key]["participants"]):
        if ws is exclude_ws:
            continue
        try:
            await ws.send_text(text)
        except Exception:
            dead.add(ws)
    
    # Cleanup dead connections
    for d in dead:
        await remove_participant(key, d)


async def send_to_ws(ws: WebSocket, message: dict):
    """Send a message to a specific websocket."""
    try:
        await ws.send_text(json.dumps(message))
    except Exception:
        pass


async def remove_participant(key: str, ws: WebSocket):
    """Remove a participant from a room."""
    if key not in rooms:
        return
    
    room = rooms[key]
    room["participants"].discard(ws)
    
    user_id = room["user_ids"].pop(ws, None)
    if user_id:
        room["nicknames"].pop(user_id, None)
        
        # Notify other participants
        await broadcast_to_room(key, {
            "type": "participant_left",
            "user_id": user_id
        })
    
    # Clean up room if empty
    if not room["participants"]:
        # Cancel sync task
        if room["sync_task"]:
            room["sync_task"].cancel()
            try:
                await room["sync_task"]
            except asyncio.CancelledError:
                pass
        del rooms[key]


async def start_sync_task(key: str):
    """Start the periodic sync broadcast task for a room."""
    if key not in rooms:
        return
    
    room = rooms[key]
    if room["sync_task"] is not None:
        return  # Already running
    
    room["sync_task"] = asyncio.create_task(sync_loop(key))


async def sync_loop(key: str):
    """Broadcast sync message every 3 seconds to all participants."""
    try:
        while True:
            await asyncio.sleep(3.0)
            
            if key not in rooms:
                break
            
            room = rooms[key]
            if not room["participants"]:
                break
            
            # Calculate current time
            current_time = get_current_time(room["state"])
            
            await broadcast_to_room(key, {
                "type": "sync",
                "time": current_time,
                "state": "playing" if room["state"]["playing"] else "paused",
                "server_time": int(time.time() * 1000)
            })
    except asyncio.CancelledError:
        raise
    except Exception:
        pass  # Task will be restarted if needed


def check_conflict_resolution(room_state: dict, event_timestamp: float, user_id: str) -> tuple[bool, str]:
    """
    Check if an event should be accepted based on conflict resolution rules.
    Returns: (accepted: bool, reason: str)
    """
    last_ts = room_state["last_action_timestamp"]
    last_user = room_state["last_action_user"]
    
    # Fast reject: event is older than last processed (outside tolerance)
    if event_timestamp < last_ts - CONFLICT_TOLERANCE_S:
        return False, "stale_event"
    
    # Within tolerance: need tiebreaker
    if abs(event_timestamp - last_ts) <= CONFLICT_TOLERANCE_S:
        # If we have a last actor, use lexicographical comparison
        if last_user is not None and user_id > last_user:
            return False, "tiebreaker_loss"
    
    return True, ""


async def handle_control_message(key: str, ws: WebSocket, msg: dict, user_id: str):
    """Handle control messages (play, pause, seek, load) from any participant."""
    if key not in rooms:
        return
    
    room = rooms[key]
    typ = msg.get("type")
    event_timestamp = float(msg.get("timestamp", 0))
    
    # Check conflict resolution
    accepted, reason = check_conflict_resolution(room["state"], event_timestamp, user_id)
    
    if not accepted:
        await send_to_ws(ws, {
            "type": "control_rejected",
            "reason": reason,
            "your_timestamp": event_timestamp,
            "server_timestamp": room["state"]["last_action_timestamp"]
        })
        return
    
    # Update state tracking
    room["state"]["last_action_timestamp"] = event_timestamp
    room["state"]["last_action_user"] = user_id
    room["state"]["last_update_time"] = time.time()
    
    if typ == "load":
        url = msg.get("url")
        if not is_valid_video_link(url):
            await send_to_ws(ws, {"type": "error", "msg": "invalid video url"})
            return
        
        room["state"]["video"] = url
        room["state"]["time"] = float(msg.get("time", 0))
        room["state"]["playing"] = False
        
        await broadcast_to_room(key, {
            "type": "state_update",
            "state": {
                "video": room["state"]["video"],
                "state": "paused",
                "time": room["state"]["time"],
                "last_update": int(event_timestamp * 1000),
                "last_actor": user_id
            }
        })
    
    elif typ == "play":
        room["state"]["time"] = float(msg.get("time", 0))
        room["state"]["playing"] = True
        
        await broadcast_to_room(key, {
            "type": "state_update",
            "state": {
                "video": room["state"]["video"],
                "state": "playing",
                "time": room["state"]["time"],
                "last_update": int(event_timestamp * 1000),
                "last_actor": user_id
            }
        })
    
    elif typ == "pause":
        room["state"]["time"] = float(msg.get("time", 0))
        room["state"]["playing"] = False
        
        await broadcast_to_room(key, {
            "type": "state_update",
            "state": {
                "video": room["state"]["video"],
                "state": "paused",
                "time": room["state"]["time"],
                "last_update": int(event_timestamp * 1000),
                "last_actor": user_id
            }
        })
    
    elif typ == "seek":
        room["state"]["time"] = float(msg.get("time", 0))
        # Keep playing state as is
        
        await broadcast_to_room(key, {
            "type": "state_update",
            "state": {
                "video": room["state"]["video"],
                "state": "playing" if room["state"]["playing"] else "paused",
                "time": room["state"]["time"],
                "last_update": int(event_timestamp * 1000),
                "last_actor": user_id
            }
        })


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Democratic watch-together WebSocket protocol.
    
    Connection:
      - Client sends: {"key": "room_name"}
      - Server responds with welcome message including user_id
    
    Control Messages (any participant can send):
      - play:  {type: "play", time: float, timestamp: float, user_id: str}
      - pause: {type: "pause", time: float, timestamp: float, user_id: str}
      - seek:  {type: "seek", time: float, timestamp: float, user_id: str}
      - load:  {type: "load", url: str, timestamp: float, user_id: str}
    
    Conflict Resolution:
      - Events with timestamp >= last_action_timestamp - 0.05s are accepted
      - Within 50ms tolerance, user_id lexicographical comparison is tiebreaker
      - Rejected events receive "control_rejected" response
    
    Periodic Sync:
      - Server broadcasts sync every 3 seconds
    """
    await ws.accept()
    key = None
    user_id = generate_user_id()
    
    try:
        # Receive init message
        try:
            init_raw = await asyncio.wait_for(ws.receive_text(), timeout=8.0)
        except asyncio.TimeoutError:
            return
        
        try:
            init = json.loads(init_raw)
            key = init.get("key")
        except Exception:
            return
        
        if not key:
            return
        
        async with rooms_lock:
            make_room_if_missing(key)
            room = rooms[key]
            room["participants"].add(ws)
            room["user_ids"][ws] = user_id
            
            # Get or assign nickname
            nickname = init.get("nickname", f"User_{user_id[-4:]}")
            room["nicknames"][user_id] = nickname
            
            # Start sync task if first participant
            if len(room["participants"]) == 1:
                await start_sync_task(key)
        
        # Send welcome message with user_id and current state
        current_state = rooms[key]["state"]
        participants_list = [
            {"user_id": uid, "nickname": room["nicknames"].get(uid, "Unknown")}
            for uid in room["nicknames"].keys()
            if uid != user_id  # Exclude self from list
        ]
        
        await send_to_ws(ws, {
            "type": "welcome",
            "user_id": user_id,
            "participants": participants_list,
            "state": {
                "video": current_state["video"],
                "state": "playing" if current_state["playing"] else "paused",
                "time": get_current_time(current_state),
                "last_update": int(current_state["last_action_timestamp"] * 1000),
                "last_actor": current_state["last_action_user"]
            }
        })
        
        # Notify other participants
        await broadcast_to_room(key, {
            "type": "participant_joined",
            "user_id": user_id,
            "nickname": nickname
        }, exclude_ws=ws)
        
        # Handle messages
        while True:
            try:
                raw = await ws.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                break
            
            # Parse message
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            
            msg_type = msg.get("type")
            
            # Handle control messages (play, pause, seek, load)
            if msg_type in ("play", "pause", "seek", "load"):
                await handle_control_message(key, ws, msg, user_id)
            
            # Handle request_sync (client asking for current state)
            elif msg_type == "request_sync":
                current_state = rooms[key]["state"]
                await send_to_ws(ws, {
                    "type": "state_update",
                    "state": {
                        "video": current_state["video"],
                        "state": "playing" if current_state["playing"] else "paused",
                        "time": get_current_time(current_state),
                        "last_update": int(current_state["last_action_timestamp"] * 1000),
                        "last_actor": current_state["last_action_user"]
                    }
                })
            
            # Handle ping/pong for connection keepalive
            elif msg_type == "ping":
                await send_to_ws(ws, {
                    "type": "pong",
                    "client_timestamp": msg.get("timestamp"),
                    "server_timestamp": int(time.time() * 1000)
                })
            
            # Ignore unknown message types
    
    finally:
        # Cleanup
        if key:
            await remove_participant(key, ws)
