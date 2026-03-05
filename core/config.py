"""
config.py
=========
Centralised configuration for ShelfScan.
All environment variables are defined here.
"""

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------

ENV = os.environ.get("ENV", "development")

# ---------------------------------------------------------------------
# SECURITY
# ---------------------------------------------------------------------

JWT_SECRET = os.environ.get("JWT_SECRET")

if not JWT_SECRET:
    if ENV == "production":
        raise RuntimeError(
            "JWT_SECRET must be set in production. "
            "Add it to your .env file: JWT_SECRET=your-secret-key-here"
        )
    else:
        JWT_SECRET = "dev-secret-change-me"
        logger.warning(
            "JWT_SECRET not set — using insecure default for development. "
            "Set JWT_SECRET in your .env file before deploying."
        )

JWT_EXPIRY_DAYS = int(os.environ.get("JWT_EXPIRY_DAYS", 7))
BCRYPT_ROUNDS   = int(os.environ.get("BCRYPT_ROUNDS",   12))

# ---------------------------------------------------------------------
# AUTH RATE LIMITING
# ---------------------------------------------------------------------

MAX_LOGIN_ATTEMPTS = int(os.environ.get("MAX_LOGIN_ATTEMPTS", 5))
LOCKOUT_SECONDS    = int(os.environ.get("LOCKOUT_SECONDS",    300))

# ---------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "data/shelfscan.db"
)

# ---------------------------------------------------------------------
# ML MODEL
# ---------------------------------------------------------------------

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "data/health_model.pkl"
)

# ---------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------

MAX_COMPARE_ITEMS = 2