import os
import logging

# cloud deployment logging 
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Starting app - PORT from env: {os.environ.get('PORT', 'not set')}")
logger.debug(f"Binding to 0.0.0.0:{os.environ.get('PORT', '8080')}")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from doc_worker.configs.settings import settings

app = FastAPI(
    title=settings.APP_NAME
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.CORS_DEV_ORIGIN, settings.CORS_PROD_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  # Allows all headers
)
