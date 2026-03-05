"""
utils/auth.py
=============
Authentication layer for ShelfScan.
bcrypt password hashing + JWT session tokens.

Security improvements:
- Requires JWT_SECRET environment variable (no insecure default)
- Stronger email validation
- Generic login error to prevent account enumeration
- JWT includes issued-at (iat)
- Configurable bcrypt cost via env
- Rate limiting on login to prevent brute-force attacks
- Token revocation support for logout/invalidation
- Timing-safe login to prevent account enumeration via response time
"""

import os
import re
import time
import jwt
import bcrypt
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from utils.db import create_user, get_user_by_email

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------

JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET environment variable must be set")

JWT_EXPIRY_DAYS = int(os.environ.get("JWT_EXPIRY_DAYS", 7))
BCRYPT_ROUNDS   = int(os.environ.get("BCRYPT_ROUNDS", 12))
ALGORITHM       = "HS256"

EMAIL_RE = re.compile(r"^[^@]+@[^@]+\.[^@]+$")

# -----------------------------------------------------------------------
# RATE LIMITING
# -----------------------------------------------------------------------

MAX_LOGIN_ATTEMPTS = int(os.environ.get("MAX_LOGIN_ATTEMPTS", 5))
LOCKOUT_SECONDS    = int(os.environ.get("LOCKOUT_SECONDS", 300))  # 5 minutes

# Maps email -> list of failed attempt timestamps.
# Replace with Redis in production for multi-instance deployments:
#   redis_client.zadd(f"login_attempts:{email}", {now: now})
#   redis_client.zremrangebyscore(f"login_attempts:{email}", 0, now - LOCKOUT_SECONDS)
_login_attempts: dict[str, list[float]] = defaultdict(list)


def _is_rate_limited(email: str) -> bool:
    """
    Return True if this email has exceeded MAX_LOGIN_ATTEMPTS
    within the LOCKOUT_SECONDS window.
    Prunes stale timestamps on every call.
    """
    now = time.monotonic()
    window_start = now - LOCKOUT_SECONDS

    # Discard attempts outside the current window
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
    Call this on logout or when a token is suspected compromised.
    """
    _revoked_tokens.add(token)


# -----------------------------------------------------------------------
# PASSWORD
# -----------------------------------------------------------------------

# A pre-computed dummy hash used in login() to ensure bcrypt always runs,
# preventing timing attacks that would reveal whether an email is registered.
_DUMMY_HASH = bcrypt.hashpw(b"dummy-timing-password", bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode()


def _hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(
        password.encode(), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    ).decode()


def _check_password(password: str, hashed: str) -> bool:
    """Verify password against stored bcrypt hash. Never raises."""
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


# -----------------------------------------------------------------------
# JWT
# -----------------------------------------------------------------------

def _create_token(user_id: int, email: str) -> str:
    """Create a signed JWT token for a user session."""
    now = datetime.now(timezone.utc)
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
    Returns user_id if valid, otherwise None.
    """
    # Reject revoked tokens before any cryptographic work
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
    """Return True if email matches basic email pattern."""
    return bool(EMAIL_RE.match(email))


def _valid_password(password: str) -> bool:
    """
    Basic password policy: minimum 8 characters.
    """
    return isinstance(password, str) and len(password) >= 8


# -----------------------------------------------------------------------
# SIGNUP / LOGIN
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
    return {
        "ok":      True,
        "token":   token,
        "user_id": user_id,
        "email":   email,
    }


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

    # Check rate limit before doing any DB or bcrypt work
    if _is_rate_limited(email):
        return {"ok": False, "error": "Too many attempts. Please try again later."}

    user = get_user_by_email(email)

    # Always run bcrypt to normalize response time.
    # If the user doesn't exist, compare against a dummy hash so the
    # function takes the same amount of time either way.
    hash_to_check = user["password_hash"] if user else _DUMMY_HASH
    password_ok   = _check_password(password, hash_to_check)

    # Generic error prevents leaking whether the email is registered
    if not user or not password_ok:
        return {"ok": False, "error": "Invalid email or password."}

    token = _create_token(user["id"], email)
    return {
        "ok":      True,
        "token":   token,
        "user_id": user["id"],
        "email":   email,
    }
