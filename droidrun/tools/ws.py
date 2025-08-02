"""
WsTools - WebSocket-based implementation of the Tools interface.

The PC runs a WebSocket **server** and waits for a single phone/client
connection.  Every UI-level primitive (tap, swipe, input_text, …) is
translated into a JSON frame and sent over the socket.  The phone is
responsible for executing the action (via accessibility service) and
sending back an acknowledgement result frame.

Message format PC → phone (action request):
{
  "type": "action",
  "id": <int>,          # correlates request/response
  "name": "tap_by_index",
  "args": { ... }
}

Message format phone → PC (action result):
{
  "type": "action_result",
  "id": <int>,          # same as request
  "status": "ok" | "error",
  "info": "human readable text",
  "data": {...}
}

Any other messages (e.g. ping/pong, state updates) are simply logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Tuple

from .tools import Tools
from .wsdevice import WsDevice

logger = logging.getLogger("droidrun-tools")


import websockets

class WsTools(Tools):
    """A *minimal* synchronous wrapper around an async `websockets` connection.

    Internally we use the running asyncio loop to send/receive messages but
    present a blocking (synchronous) API because the rest of DroidRun expects
    `Tools` methods to be synchronous.
    """

    def __init__(self, websocket: "websockets.WebSocketServerProtocol", loop: asyncio.AbstractEventLoop):
        self.ws = websocket
        self.loop = loop

        # Correlation-id bookkeeping
        self._next_id: int = 0
        self._pending: Dict[int, asyncio.Future] = {}

        # State flags expected elsewhere in the framework
        self.finished: bool = False
        self.success: bool | None = None
        self.reason: str | None = None
        self.memory: List[str] = []  # memory helpers not used in WS mode

        # Thin wrapper providing adbutils-like API (click, swipe, ...)
        self.device = WsDevice(self)

        # Start background listener that routes action results
        self._listener_task = loop.create_task(self._listener())

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    async def _listener(self) -> None:
        """Background task that dispatches incoming messages to waiting futures."""
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                except Exception as e:  # pragma: no cover
                    logger.debug(f"[WsTools] Dropping non-json message: {message!r} – {e}")
                    continue

                msg_type = data.get("type")
                if msg_type == "action_result":
                    msg_id = data.get("id")
                    fut = self._pending.pop(msg_id, None)
                    if fut and not fut.done():
                        fut.set_result(data)
                elif msg_type == "pong":
                    logger.debug("[WsTools] <pong>")
                else:
                    logger.debug(f"[WsTools] Unhandled message type: {msg_type}")
        except asyncio.CancelledError:
            # Normal shutdown
            return
        except Exception as e:  # pragma: no cover
            logger.error(f"[WsTools] Listener stopped with error: {e}")

    def _send_action_sync(self, name: str, **args: Any) -> Any:
        """Serialize + send the action and block until result is received."""
        self._next_id += 1
        msg_id = self._next_id
        frame = {
            "type": "action",
            "id": msg_id,
            "name": name,
            "args": args,
        }

        # Schedule coroutine in the *running* loop and wait for result
        fut = asyncio.run_coroutine_threadsafe(self._send_action(frame, msg_id), self.loop)
        response = fut.result()  # blocks current thread until done

        status = response.get("status", "ok")
        info = response.get("info", "")
        data = response.get("data")
        if status != "ok":
            return f"Error: {info or status}"
        return data if data is not None else info

    async def _send_action(self, frame: Dict[str, Any], msg_id: int) -> Dict[str, Any]:
        # Prepare a future to be resolved by the listener
        pending_fut: asyncio.Future = self.loop.create_future()
        self._pending[msg_id] = pending_fut

        await self.ws.send(json.dumps(frame))
        # Wait for phone to reply
        result: Dict[str, Any] = await pending_fut
        return result

    # ---------------------------------------------------------------------
    # Tools API implementations (mostly 1-liners delegating to _send_action_sync)
    # ---------------------------------------------------------------------

    # UI interaction ------------------------------------------------------
    def tap_by_index(self, index: int):
        return self._send_action_sync("tap_by_index", index=index)

    def tap_by_description(self, description: str):
        return self._send_action_sync("tap_by_description", description=description)

    def tap_by_coordinates(self, x: int, y: int):
        """
        Tap on absolute screen coordinates and save an annotated debug screenshot.

        This mirrors the behaviour of ``AdbTools.tap_by_description`` so that we
        retain a visual audit trail of every tap even when running in WebSocket
        mode.

        Steps:
        1) Grab a screenshot from the device.
        2) Draw a small red dot at the tapped location.
        3) Persist the annotated image to ``~/Desktop/DebugImage``.
        4) Append metadata to ``log.json`` inside that folder.
        5) Forward the tap to the phone via the existing WebSocket connection.
        """
        try:
            # 1) Screenshot
            img_format, img_bytes = self.take_screenshot()

            # 2) Annotate screenshot with a red dot
            import io
            import os
            import json as _json
            from datetime import datetime

            from PIL import Image, ImageDraw  # pillow dependency

            img = Image.open(io.BytesIO(img_bytes))
            draw = ImageDraw.Draw(img)
            r = 8  # radius of the red dot
            draw.ellipse((x - r, y - r, x + r, y + r), fill="red")

            # 3) Ensure debug directory exists
            desktop = os.path.expanduser("~/Desktop")
            debug_dir = os.path.join(desktop, "DebugImage")
            os.makedirs(debug_dir, exist_ok=True)

            # 4) Save annotated screenshot with unique filename
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = f"tap_{ts}.png"
            save_path = os.path.join(debug_dir, fname)
            img.save(save_path)

            # 5) Log the action
            log_path = os.path.join(debug_dir, "log.json")
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    try:
                        log = _json.load(f)
                    except _json.JSONDecodeError:
                        log = []
            else:
                log = []

            log.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "file": fname,
                    "x": x,
                    "y": y,
                }
            )
            with open(log_path, "w", encoding="utf-8") as f:
                _json.dump(log, f, indent=2)

            # 6) Perform the tap on the device
            result = self.device.click(x, y)

            return (
                f"Tapped at ({x}, {y}); annotation saved as {save_path}"
            )
        except Exception as e:  # pragma: no cover
            logger.debug(f"[WsTools] Error tapping by coordinates: {e}", exc_info=True)
            return f"Error tapping by coordinates: {e}"

    def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 300,
    ):
        return self.device.swipe(
            start_x,
            start_y,
            end_x,
            end_y,
            duration_ms,
        )

    def input_text(self, text: str):
        return self._send_action_sync("input_text", text=text)

    def back(self):
        return self._send_action_sync("back")

    def press_key(self, keycode: int):
        return self._send_action_sync("press_key", keycode=keycode)

    # App / system --------------------------------------------------------
    def start_app(self, package: str, activity: str | None = None):
        return self._send_action_sync("start_app", package=package, activity=activity)

    def take_screenshot(self) -> Tuple[str, bytes]:
        data = self._send_action_sync("take_screenshot")
        # Expecting phone to return {"format":"PNG","base64":"..."}
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected screenshot data type: {type(data)} – {data}")
        import base64

        img_bytes = base64.b64decode(data.get("base64", ""))
        img_format = data.get("format", "PNG")
        return img_format, img_bytes

    def list_packages(self, include_system_apps: bool = False):
        data = self._send_action_sync("list_packages", include_system_apps=include_system_apps)
        return data if isinstance(data, list) else []

    # State helpers -------------------------------------------------------
    def get_state(self):
        data = self._send_action_sync("get_state")
        return data if isinstance(data, dict) else {}

    # Memory helpers ------------------------------------------------------
    def remember(self, information: str):
        # No-op in WebSocket mode
        return "Remember/Memory not supported in WsTools"

    def get_memory(self):
        return []

    # Complete ------------------------------------------------------------
    def complete(self, success: bool, reason: str = ""):
        self.finished = True
        self.success = success
        self.reason = reason
        return True
