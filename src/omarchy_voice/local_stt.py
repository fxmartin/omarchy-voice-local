"""whisper.cpp speech recognition client for the local engine.

Posts one utterance to the server's `/inference` endpoint using only the
standard library. The audio exists only as bytes in memory: it is wrapped in a
WAV container in a buffer and sent, never written to disk.
"""

from __future__ import annotations

import io
import json
import secrets
import socket
import urllib.error
import urllib.request
import wave
from urllib.parse import urlsplit

from .config import Config

SAMPLE_RATE = 16000  # whisper.cpp's native rate; 16-bit mono PCM


class SttError(Exception):
    """Recognition failed; the message is short enough to show the user."""


def pcm_to_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def _multipart(fields: dict[str, str], wav: bytes) -> tuple[bytes, str]:
    boundary = secrets.token_hex(16)
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in fields.items()
    ]
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="utterance.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
        + wav + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def transcribe(config: Config, pcm: bytes) -> str:
    """Return the transcript of 16 kHz mono 16-bit PCM, or raise SttError."""
    base = config.local_stt_url.rstrip("/")
    if urlsplit(base).scheme not in ("http", "https"):
        raise SttError("stt_url must be an http:// or https:// address")
    body, content_type = _multipart(
        {"response_format": "json", "temperature": "0.0", "language": config.local_language},
        pcm_to_wav(pcm))
    request = urllib.request.Request(
        base + "/inference", data=body, method="POST",
        headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(request, timeout=config.local_stt_timeout_seconds) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise SttError(f"recognition server returned HTTP {exc.code}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise SttError("recognition timed out") from exc
    except (urllib.error.URLError, OSError) as exc:
        if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise SttError("recognition timed out") from exc
        raise SttError("recognition server unreachable") from exc
    try:
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError
    except ValueError as exc:
        raise SttError("recognition server sent an unreadable reply") from exc
    if data.get("error"):
        raise SttError("recognition server reported an error")
    text = data.get("text")
    if not isinstance(text, str):
        raise SttError("recognition server sent no transcript")
    return text.strip()
