import re
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Scope, Receive, Send
from api.configs.settings import settings

# matches /{UUID}/{UUID} exactly matches any path with /{project_id}/{chatbot_id}
_PUBLIC_PATH = re.compile(
    r"^/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE
)

class DualCORSMiddleware:
    """
    DualCORSMiddleware is a custom middleware that handles CORS for two different scenarios:
    1. Public endpoints: These endpoints are accessible by anyone and do not require authentication.
    2. Private endpoints: These endpoints require authentication and are only accessible by Intellichat's frontend.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

        self.private_cors = CORSMiddleware(
            app=app,
            allow_origins=[
                settings.CORS_DEV_ORIGIN,
                settings.CORS_PROD_ORIGIN,
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.public_cors = CORSMiddleware(
            app=app,
            allow_origins=["*"],
            allow_credentials=False,   # must be False with "*"
            allow_methods=["POST", "OPTIONS"],
            allow_headers=["Content-Type", f"{settings.API_KEY_HEADER_NAME}"],
        )

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        if _PUBLIC_PATH.match(path):
            await self.public_cors(scope, receive, send)
        else:
            await self.private_cors(scope, receive, send)
        