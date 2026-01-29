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
from fastapi.middleware.gzip import GZipMiddleware
import gzip
import json

app = FastAPI()

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

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

# API для отримання списку відео та серій
@app.get("/api/videos")
def get_videos():
    """Повертає список доступних відео файлів та серій з папки static/videos."""
    videos_dir = "static/videos"
    allowed_extensions = {'.mp4', '.webm', '.ogg', '.mkv', '.mov'}
    result = {"videos": [], "series": []}
    
    try:
        if os.path.exists(videos_dir):
            for item in sorted(os.listdir(videos_dir)):
                item_path = os.path.join(videos_dir, item)
                if os.path.isfile(item_path):
                    ext = os.path.splitext(item)[1].lower()
                    if ext in allowed_extensions:
                        # Отримуємо розмір файлу
                        size = os.path.getsize(item_path)
                        size_str = format_file_size(size)
                        
                        result["videos"].append({
                            "name": item,
                            "url": f"/video/{item}",
                            "size": size_str,
                            "type": ext[1:]  # без крапки
                        })
                    elif ext == '.txt':
                        # Читаємо URL з текстового файлу
                        with open(item_path, 'r', encoding='utf-8') as f:
                            url = f.read().strip()
                        if url:
                            result["videos"].append({
                                "name": item.replace('.txt', ''),
                                "url": url,
                                "size": "--",
                                "type": "url"
                            })
                elif os.path.isdir(item_path):
                    # Знайдемо всі відео в папці серії
                    episodes = []
                    for filename in sorted(os.listdir(item_path)):
                        file_path = os.path.join(item_path, filename)
                        if os.path.isfile(file_path):
                            ext = os.path.splitext(filename)[1].lower()
                            if ext in allowed_extensions:
                                size = os.path.getsize(file_path)
                                size_str = format_file_size(size)
                                episodes.append({
                                    "name": filename,
                                    "url": f"/video/{item}/{filename}",
                                    "size": size_str,
                                    "type": ext[1:]
                                })
                            elif ext == '.txt':
                                # Читаємо URL з текстового файлу
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    url = f.read().strip()
                                if url:
                                    episodes.append({
                                        "name": filename.replace('.txt', ''),
                                        "url": url,
                                        "size": "--",
                                        "type": "url"
                                    })
                    if episodes:
                        result["series"].append({
                            "name": item,
                            "path": item,
                            "episodes": episodes
                        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to list videos: {str(e)}"}
        )
    
    return result


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


# API для створення серії та додавання епізодів
@app.post("/api/series")
async def create_series(series_data: dict):
    """Створює нову серію та додає епізоди."""
    series_name = series_data.get("name")
    episodes = series_data.get("episodes", [])

    if not series_name:
        return JSONResponse(status_code=400, content={"error": "Series name is required"})

    series_dir = os.path.join("static/videos", series_name)

    try:
        if not os.path.exists(series_dir):
            os.makedirs(series_dir)

        # Обробляємо епізоди
        for i, episode in enumerate(episodes):
            episode_name = episode.get("name")
            episode_url = episode.get("url")

            if episode_name and episode_url:
                # Створюємо файл з посиланням
                episode_filename = f"episode_{i+1:02d}_{episode_name.replace(' ', '_')}"
                if not episode_filename.lower().endswith(('.mp4', '.webm', '.ogg', '.mkv', '.mov')):
                    episode_filename += ".txt"

                episode_path = os.path.join(series_dir, episode_filename)

                with open(episode_path, 'w', encoding='utf-8') as f:
                    f.write(episode_url)

        return JSONResponse(status_code=201, content={"success": True, "message": "Series created successfully"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to create series: {str(e)}"})


# API для додавання одиночного відео
@app.post("/api/single-video")
async def add_single_video(video_data: dict):
    """Додає одиночне відео до папки videos."""
    video_name = video_data.get("name")
    video_url = video_data.get("url")

    if not video_name or not video_url:
        return JSONResponse(status_code=400, content={"error": "Video name and URL are required"})

    videos_dir = "static/videos"

    try:
        if not os.path.exists(videos_dir):
            os.makedirs(videos_dir)

        # Створюємо файл з посиланням
        video_filename = f"{video_name.replace(' ', '_')}"
        if not video_filename.lower().endswith(('.mp4', '.webm', '.ogg', '.mkv', '.mov')):
            video_filename += ".txt"

        video_path = os.path.join(videos_dir, video_filename)

        with open(video_path, 'w', encoding='utf-8') as f:
            f.write(video_url)

        return JSONResponse(status_code=201, content={"success": True, "message": "Video added successfully"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to add video: {str(e)}"})


# API для видалення серії
@app.delete("/api/series/{series_name}")
async def delete_series(series_name: str):
    """Видаляє серію та всі її епізоди."""
    series_dir = os.path.join("static/videos", series_name)

    try:
        if not os.path.exists(series_dir):
            return JSONResponse(status_code=404, content={"error": "Series not found"})

        # Видаляємо всю папку серії
        import shutil
        shutil.rmtree(series_dir)

        return JSONResponse(status_code=200, content={"success": True, "message": "Series deleted successfully"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to delete series: {str(e)}"})


# API для оновлення серії
@app.put("/api/series/{old_series_name}")
async def update_series(old_series_name: str, series_data: dict):
    """Оновлює серію та її епізоди."""
    new_series_name = series_data.get("name")
    episodes = series_data.get("episodes", [])

    if not new_series_name:
        return JSONResponse(status_code=400, content={"error": "Series name is required"})

    old_series_dir = os.path.join("static/videos", old_series_name)
    new_series_dir = os.path.join("static/videos", new_series_name)

    try:
        if not os.path.exists(old_series_dir):
            return JSONResponse(status_code=404, content={"error": "Series not found"})

        # Якщо назва змінилась, перейменовуємо папку
        if old_series_name != new_series_name:
            if os.path.exists(new_series_dir):
                return JSONResponse(status_code=400, content={"error": "Series with this name already exists"})
            os.rename(old_series_dir, new_series_dir)
        else:
            new_series_dir = old_series_dir

        # Видаляємо старі файли епізодів
        for filename in os.listdir(new_series_dir):
            file_path = os.path.join(new_series_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Створюємо нові файли епізодів
        for i, episode in enumerate(episodes):
            episode_name = episode.get("name")
            episode_url = episode.get("url")

            if episode_name and episode_url:
                episode_filename = f"episode_{i+1:02d}_{episode_name.replace(' ', '_')}"
                if not episode_filename.lower().endswith(('.mp4', '.webm', '.ogg', '.mkv', '.mov')):
                    episode_filename += ".txt"

                episode_path = os.path.join(new_series_dir, episode_filename)

                with open(episode_path, 'w', encoding='utf-8') as f:
                    f.write(episode_url)

        return JSONResponse(status_code=200, content={"success": True, "message": "Series updated successfully"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to update series: {str(e)}"})


# API для видалення одиночного відео
@app.delete("/api/single-video/{video_name}")
async def delete_single_video(video_name: str):
    """Видаляє одиночне відео."""
    videos_dir = "static/videos"

    try:
        # Знаходимо файл відео
        video_filename = f"{video_name.replace(' ', '_')}"
        if not video_filename.lower().endswith(('.mp4', '.webm', '.ogg', '.mkv', '.mov', '.txt')):
            video_filename += ".txt"

        video_path = os.path.join(videos_dir, video_filename)

        if not os.path.exists(video_path):
            return JSONResponse(status_code=404, content={"error": "Video not found"})

        os.remove(video_path)

        return JSONResponse(status_code=200, content={"success": True, "message": "Video deleted successfully"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to delete video: {str(e)}"})


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
