"""
Taille maximale du corps des requêtes (#66).

Starlette écrit tout le fichier d'un upload multipart sur disque avant
d'appeler la route : une vérification dans la route arriverait trop tard
(disque déjà rempli). Ce middleware ASGI refuse donc la requête dès
l'en-tête `Content-Length` quand il annonce trop, et compte de toute façon
les octets reçus, pour les corps sans `Content-Length` (chunked) ou dont
l'en-tête ment.
"""

import json

from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

PAYLOAD_TOO_LARGE = 413


def too_large_detail(max_bytes: int) -> str:
    return f"Request body larger than the {max_bytes // 1024} KB limit (MAX_UPLOAD_MB)."


class MaxBodySizeMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI relance les HTTPException levées en lisant le
                    # corps : la réponse passe par l'enveloppe d'erreurs
                    raise HTTPException(PAYLOAD_TOO_LARGE, too_large_detail(self.max_bytes))
            return message

        await self.app(scope, limited_receive, send)

    async def _reject(self, send: Send) -> None:
        body = json.dumps({"error": "payload_too_large", "detail": too_large_detail(self.max_bytes)}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": PAYLOAD_TOO_LARGE,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})
