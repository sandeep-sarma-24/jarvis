"""Frame → JPEG, for the web preview and for images attached to a model request."""
from __future__ import annotations

import cv2
import numpy as np

from app.perception.types import Frame


def frame_to_jpeg(frame: Frame, *, max_side: int | None = None, quality: int = 70) -> bytes:
    """Encode an rgb24 Frame; blocking, call via asyncio.to_thread."""
    img = np.frombuffer(frame.data, np.uint8).reshape(frame.height, frame.width, 3)
    if max_side and max(frame.width, frame.height) > max_side:
        scale = max_side / max(frame.width, frame.height)
        img = cv2.resize(img, (round(frame.width * scale), round(frame.height * scale)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()
