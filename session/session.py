"""
session/session.py
==================
Streamlit session state management for ShelfScan.

Initialises all session keys and provides typed helpers for auth,
scan results, and compare mode.

Call init_session() once at the top of app.py.
"""

import copy
import streamlit as st
from typing import Any

from core.config import MAX_COMPARE_ITEMS


# -----------------------------------------------------------------------
# DEFAULT SESSION STATE
# -----------------------------------------------------------------------

_SESSION_DEFAULTS: dict[str, Any] = {

    # ---------------- AUTH ----------------
    "user_id":   None,
    "email":     None,
    "jwt_token": None,
    "logged_in": False,

    # ---------------- SCAN ----------------
    "current_product": None,
    "current_health":  None,
    "current_image":   None,
    "current_price":   None,

    # ---------------- COMPARE ----------------
    # List of {product, health, price} dicts — cleaner than parallel keys.
    "compare_items": [],

    # ---------------- UI STATE ----------------
    "active_tab":    "scan",
    "barcode_input": "",
    "show_history":  False,
    "show_compare":  False,

    # ---------------- HISTORY ----------------
    # Shallow copy sufficient while entries are primitives/strings.
    # Switch to copy.deepcopy() if entries become nested dicts.
    "scan_history": [],
}

SCAN_KEYS = (
    "current_product",
    "current_health",
    "current_image",
    "current_price",
)


# -----------------------------------------------------------------------
# INIT
# -----------------------------------------------------------------------

def init_session() -> None:
    """
    Initialise Streamlit session state.
    Safe to call on every page load — only missing keys are created.
    """
    for key, default in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            if isinstance(default, list):
                st.session_state[key] = default.copy()
            elif isinstance(default, dict):
                st.session_state[key] = copy.deepcopy(default)
            else:
                st.session_state[key] = default


# -----------------------------------------------------------------------
# AUTH HELPERS
# -----------------------------------------------------------------------

def is_logged_in() -> bool:
    """
    Return True only if all three auth conditions hold:
      - logged_in flag is True
      - jwt_token is present
      - user_id is present
    Defensive against partial state corruption.
    """
    return (
        bool(st.session_state.get("logged_in"))
        and st.session_state.get("jwt_token") is not None
        and st.session_state.get("user_id") is not None
    )


def set_logged_in(user_id: int, email: str, token: str) -> None:
    """
    Set auth state after successful login or signup.
    Centralises the 4-key auth write — prevents drift if keys change.
    """
    st.session_state["user_id"]   = user_id
    st.session_state["email"]     = email
    st.session_state["jwt_token"] = token
    st.session_state["logged_in"] = True


def logout() -> None:
    """
    Clear authentication and all dependent state.
    Does NOT call st.rerun() — caller's responsibility.
    """
    st.session_state["user_id"]   = None
    st.session_state["email"]     = None
    st.session_state["jwt_token"] = None
    st.session_state["logged_in"] = False

    clear_scan()
    clear_compare()

    st.session_state["show_history"] = False
    st.session_state["show_compare"] = False


def get_user_id() -> int | None:
    return st.session_state.get("user_id")


def get_token() -> str | None:
    return st.session_state.get("jwt_token")


# -----------------------------------------------------------------------
# SCAN HELPERS
# -----------------------------------------------------------------------

def set_scan_result(
    product:      dict,
    health:       dict,
    price:        dict | None = None,
    image_result: dict | None = None,
) -> None:
    """
    Store current scan results.
    UI layer should use this instead of writing keys directly.
    """
    st.session_state["current_product"] = product
    st.session_state["current_health"]  = health
    st.session_state["current_price"]   = price
    st.session_state["current_image"]   = image_result


def get_scan_result() -> tuple:
    """Return (product, health, price, image_result) — any may be None."""
    return (
        st.session_state.get("current_product"),
        st.session_state.get("current_health"),
        st.session_state.get("current_price"),
        st.session_state.get("current_image"),
    )


def clear_scan() -> None:
    """
    Clear scan results and reset active tab.
    Resets active_tab to 'scan' — leaving a non-default tab
    active with no content is a confusing UI state.
    """
    for key in SCAN_KEYS:
        st.session_state[key] = None
    st.session_state["active_tab"]    = "scan"
    st.session_state["barcode_input"] = ""


# -----------------------------------------------------------------------
# COMPARE HELPERS
# -----------------------------------------------------------------------

def add_to_compare(
    product: dict,
    health:  dict,
    price:   dict | None = None,
) -> None:
    """
    Add a scanned product to the compare list.
    Skips duplicates (matched by barcode).
    Replaces oldest item if at MAX_COMPARE_ITEMS capacity.
    """
    items = st.session_state.get("compare_items", [])

    # Duplicate guard
    for existing in items:
        if existing["product"].get("barcode") == product.get("barcode"):
            return

    if len(items) >= MAX_COMPARE_ITEMS:
        items.pop(0)

    items.append({"product": product, "health": health, "price": price})
    st.session_state["compare_items"] = items


def get_compare_items() -> list:
    return st.session_state.get("compare_items", [])


def clear_compare() -> None:
    """Clear compare list and hide the panel."""
    st.session_state["compare_items"]  = []
    st.session_state["show_compare"]   = False