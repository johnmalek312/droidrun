"""wsdevice.py - Minimal adbutils-like wrapper around a WebSocket connection.

This helper offers a very small surface API that mirrors the most commonly
used calls from `adbutils.Device`: ``click`` and ``swipe``.

The goal is to let the rest of DroidRun interact with a *device-like* object
independent of the transport layer (ADB vs WebSocket).

It delegates all work to an existing :class:`~droidrun.tools.ws.WsTools`
instance, which already maintains the WebSocket connection, correlation IDs
and background listener.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover – avoid runtime circular import
    from droidrun.tools.ws import WsTools

logger = logging.getLogger("droidrun-tools")


class WsDevice:
    """Thin wrapper that exposes ``click`` and ``swipe`` helpers.

    Parameters
    ----------
    tools : WsTools
        *Existing* ``WsTools`` instance that owns the WebSocket connection. We
        purposefully reuse it instead of creating a second listener so that
        only **one** coroutine consumes messages from the socket.
    """

    def __init__(self, tools: "WsTools") -> None:  # type: ignore[name-defined]
        self._tools = tools

    # ------------------------------------------------------------------
    # Public helpers (adbutils-style)
    # ------------------------------------------------------------------
    def click(self, x: int, y: int) -> Any:
        """Tap on the given absolute screen coordinates (pixels).

        This maps to the ``tap_by_coordinates`` action on the phone side.
        """
        logger.debug(f"[WsDevice] click({x}, {y})")
        return self._tools._send_action_sync("tap_by_coordinates", x=x, y=y)

    def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 300,
    ) -> Any:
        """Perform a straight-line swipe gesture."""
        logger.debug(
            f"[WsDevice] swipe(({start_x}, {start_y}) → ({end_x}, {end_y}), {duration_ms}ms)"
        )
        return self._tools._send_action_sync(
            "swipe",
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            duration_ms=duration_ms,
        )
