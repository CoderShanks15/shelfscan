"""
utils/auth.py
=============
Authentication layer for ShelfScan.
bcrypt password hashing + JWT session tokens.
"""

import os
import jwt
import bcrypt
from datetime import datetime, timedelta, timezone

from utils.db import create_user, get_user_by_email

JWT_SECRET  = os.environ.get('JWT_SECRET', 'shelfscan-dev-secret-change-in-prod')
JWT_EXPIRY  = 7   # days
ALGORITHM   = 'HS256'


# -----------------------------------------------------------------------
# PASSWORD
# -----------------------------------------------------------------------

def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def _check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# -----------------------------------------------------------------------
# JWT
# -----------------------------------------------------------------------

def _create_token(user_id: int, email: str) -> str:
    payload = {
        'user_id': user_id,
        'email':   email,
        'exp':     datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRY),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def verify_jwt(token: str) -> int | None:
    """
    Verify a JWT token. Returns user_id if valid, None if expired/invalid.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return payload.get('user_id')
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# -----------------------------------------------------------------------
# SIGNUP / LOGIN
# -----------------------------------------------------------------------

def signup(email: str, password: str) -> dict:
    """
    Register a new user.
    Returns:
        {'ok': True,  'token': str, 'user_id': int}
        {'ok': False, 'error': str}
    """
    email = email.strip().lower()

    if not email or '@' not in email:
        return {'ok': False, 'error': 'Invalid email address.'}
    if len(password) < 8:
        return {'ok': False, 'error': 'Password must be at least 8 characters.'}

    password_hash = _hash_password(password)
    user_id       = create_user(email, password_hash)

    if user_id is None:
        return {'ok': False, 'error': 'Email already registered.'}

    token = _create_token(user_id, email)
    return {'ok': True, 'token': token, 'user_id': user_id, 'email': email}


def login(email: str, password: str) -> dict:
    """
    Authenticate an existing user.
    Returns:
        {'ok': True,  'token': str, 'user_id': int, 'email': str}
        {'ok': False, 'error': str}
    """
    email = email.strip().lower()
    user  = get_user_by_email(email)

    if not user:
        return {'ok': False, 'error': 'Email not found.'}
    if not _check_password(password, user['password_hash']):
        return {'ok': False, 'error': 'Incorrect password.'}

    token = _create_token(user['id'], email)
    return {
        'ok':      True,
        'token':   token,
        'user_id': user['id'],
        'email':   email,
    }