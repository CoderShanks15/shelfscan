"""
auth/oauth.py
=============
OAuth2 flows for Google and GitHub.

Uses Streamlit query params for the OAuth callback:
  1. User clicks "Continue with Google/GitHub"
  2. Browser redirects to provider's auth page
  3. Provider redirects back to APP_URL with ?code=xxx&provider=yyy
  4. This module exchanges the code for user info
  5. User is logged in (or created) seamlessly

Requirements:
  - Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET in .env
  - Set GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET in .env
  - In Google Cloud Console, add APP_URL as authorized redirect URI
  - In GitHub OAuth App, set callback URL to APP_URL
"""

import logging
import urllib.parse
import requests

from core.config import (
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GITHUB_CLIENT_ID,
    GITHUB_CLIENT_SECRET,
    APP_URL,
)

logger = logging.getLogger(__name__)

TIMEOUT = 15

# -----------------------------------------------------------------------
# AVAILABILITY CHECKS
# -----------------------------------------------------------------------

def google_available() -> bool:
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

def github_available() -> bool:
    return bool(GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET)


# -----------------------------------------------------------------------
# GOOGLE OAUTH2
# -----------------------------------------------------------------------

_GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_USER_URL  = "https://www.googleapis.com/oauth2/v2/userinfo"


def google_auth_url() -> str:
    """Generate the Google OAuth2 authorization URL."""
    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  APP_URL,
        "response_type": "code",
        "scope":         "openid email profile",
        "state":         "google",
        "access_type":   "offline",
        "prompt":        "select_account",
    }
    return f"{_GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"


def google_exchange(code: str) -> dict:
    """
    Exchange a Google authorization code for user info.

    Returns:
        {'ok': True,  'email': str, 'name': str, 'picture': str}
        {'ok': False, 'error': str}
    """
    try:
        # Exchange code for token
        token_resp = requests.post(_GOOGLE_TOKEN_URL, data={
            "code":          code,
            "client_id":     GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri":  APP_URL,
            "grant_type":    "authorization_code",
        }, timeout=TIMEOUT)

        if token_resp.status_code != 200:
            logger.error("Google token exchange failed: %s", token_resp.text)
            return {"ok": False, "error": "Google authentication failed."}

        access_token = token_resp.json().get("access_token")
        if not access_token:
            return {"ok": False, "error": "No access token from Google."}

        # Fetch user info
        user_resp = requests.get(_GOOGLE_USER_URL, headers={
            "Authorization": f"Bearer {access_token}",
        }, timeout=TIMEOUT)

        if user_resp.status_code != 200:
            return {"ok": False, "error": "Failed to get Google profile."}

        user = user_resp.json()
        return {
            "ok":      True,
            "email":   user.get("email", ""),
            "name":    user.get("name", ""),
            "picture": user.get("picture", ""),
        }

    except requests.exceptions.Timeout:
        return {"ok": False, "error": "Google request timed out."}
    except Exception as e:
        logger.exception("Google OAuth error")
        return {"ok": False, "error": f"Google OAuth error: {e}"}


# -----------------------------------------------------------------------
# GITHUB OAUTH2
# -----------------------------------------------------------------------

_GITHUB_AUTH_URL  = "https://github.com/login/oauth/authorize"
_GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
_GITHUB_USER_URL  = "https://api.github.com/user"
_GITHUB_EMAIL_URL = "https://api.github.com/user/emails"


def github_auth_url() -> str:
    """Generate the GitHub OAuth authorization URL."""
    params = {
        "client_id":    GITHUB_CLIENT_ID,
        "redirect_uri": APP_URL,
        "scope":        "read:user user:email",
        "state":        "github",
    }
    return f"{_GITHUB_AUTH_URL}?{urllib.parse.urlencode(params)}"


def github_exchange(code: str) -> dict:
    """
    Exchange a GitHub authorization code for user info.

    Returns:
        {'ok': True,  'email': str, 'name': str, 'avatar': str}
        {'ok': False, 'error': str}
    """
    try:
        # Exchange code for token
        token_resp = requests.post(_GITHUB_TOKEN_URL, data={
            "code":          code,
            "client_id":     GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "redirect_uri":  APP_URL,
        }, headers={
            "Accept": "application/json",
        }, timeout=TIMEOUT)

        if token_resp.status_code != 200:
            logger.error("GitHub token exchange failed: %s", token_resp.text)
            return {"ok": False, "error": "GitHub authentication failed."}

        access_token = token_resp.json().get("access_token")
        if not access_token:
            return {"ok": False, "error": "No access token from GitHub."}

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept":        "application/json",
        }

        # Fetch user profile
        user_resp = requests.get(_GITHUB_USER_URL, headers=headers, timeout=TIMEOUT)
        if user_resp.status_code != 200:
            return {"ok": False, "error": "Failed to get GitHub profile."}

        user = user_resp.json()
        email = user.get("email")

        # If email is private, fetch from /user/emails
        if not email:
            email_resp = requests.get(_GITHUB_EMAIL_URL, headers=headers, timeout=TIMEOUT)
            if email_resp.status_code == 200:
                emails = email_resp.json()
                for e in emails:
                    if e.get("primary"):
                        email = e.get("email")
                        break
                if not email and emails:
                    email = emails[0].get("email", "")

        return {
            "ok":     True,
            "email":  email or "",
            "name":   user.get("name") or user.get("login", ""),
            "avatar": user.get("avatar_url", ""),
        }

    except requests.exceptions.Timeout:
        return {"ok": False, "error": "GitHub request timed out."}
    except Exception as e:
        logger.exception("GitHub OAuth error")
        return {"ok": False, "error": f"GitHub OAuth error: {e}"}
