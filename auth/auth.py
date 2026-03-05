"""
auth/auth.py
============
Authentication layer for ShelfScan.
bcrypt password hashing + JWT session tokens.

Security:
- JWT_SECRET loaded from core.config — fails loud in production if missing,
  warns in development.
- Timing-safe login — bcrypt always runs even for unknown emails.
- Rate limiting per email — configurable via MAX_LOGIN_ATTEMPTS / LOCKOUT_SECONDS.
- Token revocation denylist — call revoke_token() on logout.
- Generic error messages — prevents account enumeration.
"""

import logging
import re
import time

import bcrypt
import jwt
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from core.config import (
    JWT_SECRET,
    JWT_EXPIRY_DAYS,
    BCRYPT_ROUNDS,
    MAX_LOGIN_ATTEMPTS,
    LOCKOUT_SECONDS,
)
from database.db import create_user, get_user_by_email

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------

ALGORITHM = "HS256"
EMAIL_RE  = re.compile(r"^[^@]+@[^@]+\.[^@]+$")

# -----------------------------------------------------------------------
# RATE LIMITING
# -----------------------------------------------------------------------

# In-process store. Replace with Redis in production:
#   redis_client.zadd(f"login_attempts:{email}", {now: now})
#   redis_client.zremrangebyscore(f"login_attempts:{email}", 0, now - LOCKOUT_SECONDS)
_login_attempts: dict[str, list[float]] = defaultdict(list)


def _is_rate_limited(email: str) -> bool:
    """
    Return True if this email has exceeded MAX_LOGIN_ATTEMPTS
    within the LOCKOUT_SECONDS window.
    Prunes stale timestamps on every call.
    """
    now          = time.monotonic()
    window_start = now - LOCKOUT_SECONDS

    _login_attempts[email] = [
        t for t in _login_attempts[email] if t > window_start
    ]

    if len(_login_attempts[email]) >= MAX_LOGIN_ATTEMPTS:
        return True

    _login_attempts[email].append(now)
    return False


# -----------------------------------------------------------------------
# TOKEN REVOCATION
# -----------------------------------------------------------------------

# In-process denylist. Replace with Redis SET + TTL in production:
#   redis_client.setex(f"revoked:{token}", JWT_EXPIRY_DAYS * 86400, "1")
_revoked_tokens: set[str] = set()


def revoke_token(token: str) -> None:
    """
    Revoke a JWT so it is rejected by verify_jwt even before expiry.
    Call on logout or when a token is suspected compromised.
    """
    _revoked_tokens.add(token)


# -----------------------------------------------------------------------
# PASSWORD HELPERS
# -----------------------------------------------------------------------

# Pre-computed dummy hash — ensures bcrypt always runs in login(),
# preventing timing attacks that reveal whether an email is registered.
_DUMMY_HASH = bcrypt.hashpw(
    b"dummy-timing-password",
    bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
).decode()


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(
        password.encode(), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    ).decode()


def _check_password(password: str, hashed: str) -> bool:
    """Verify password against bcrypt hash. Never raises."""
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


# -----------------------------------------------------------------------
# JWT HELPERS
# -----------------------------------------------------------------------

def _create_token(user_id: int, email: str) -> str:
    now     = datetime.now(timezone.utc)
    payload = {
        "user_id": user_id,
        "email":   email,
        "iat":     now,
        "exp":     now + timedelta(days=JWT_EXPIRY_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def verify_jwt(token: str) -> int | None:
    """
    Verify a JWT token.
    Returns user_id if valid and not revoked, otherwise None.
    """
    if token in _revoked_tokens:
        return None

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[ALGORITHM],
            options={"verify_aud": False},
        )
        return payload.get("user_id")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# -----------------------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------------------

def _valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(email))


def _valid_password(password: str) -> bool:
    return isinstance(password, str) and len(password) >= 8


# -----------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------

def signup(email: str, password: str) -> dict:
    """
    Register a new user.

    Returns:
        {'ok': True,  'token': str, 'user_id': int, 'email': str}
        {'ok': False, 'error': str}
    """
    email = (email or "").strip().lower()

    if not _valid_email(email):
        return {"ok": False, "error": "Invalid email address."}

    if not _valid_password(password):
        return {"ok": False, "error": "Password must be at least 8 characters."}

    password_hash = _hash_password(password)
    user_id       = create_user(email, password_hash)

    if user_id is None:
        return {"ok": False, "error": "Email already registered."}

    token = _create_token(user_id, email)
    return {"ok": True, "token": token, "user_id": user_id, "email": email}


def login(email: str, password: str) -> dict:
    """
    Authenticate an existing user.

    Returns:
        {'ok': True,  'token': str, 'user_id': int, 'email': str}
        {'ok': False, 'error': str}

    Security notes:
    - Rate limited per email to prevent brute-force.
    - Always runs bcrypt regardless of whether the email exists,
      so response time does not reveal account existence.
    - Generic error message prevents account enumeration.
    """
    email = (email or "").strip().lower()

    if _is_rate_limited(email):
        return {"ok": False, "error": "Too many attempts. Please try again later."}

    user          = get_user_by_email(email)
    hash_to_check = user["password_hash"] if user else _DUMMY_HASH
    password_ok   = _check_password(password, hash_to_check)

    if not user or not password_ok:
        return {"ok": False, "error": "Invalid email or password."}

    token = _create_token(user["id"], email)
    return {"ok": True, "token": token, "user_id": user["id"], "email": email}