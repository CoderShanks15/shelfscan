"""
utils/db.py
===========
SQLite database layer for ShelfScan.
Handles users, scan history, and favourites.

Swappable to PostgreSQL by setting DATABASE_URL environment variable.
WAL mode enabled for better concurrent read performance.

Tables:
  users         — registered accounts
  scan_history  — every product scan per user
  favourites    — bookmarked products per user

  
  
GOAL:
ADD 
PostgreSQL + Redis + queues
"""

import os
import json
import sqlite3
import contextlib

from core.config import DATABASE_URL

_tables_created = False


# -----------------------------------------------------------------------
# CONNECTION
# -----------------------------------------------------------------------

@contextlib.contextmanager
def _conn():
    """
    Context manager — single connection per operation.
    Auto-commits on success, rolls back on error, always closes.
    Ensures data directory and tables exist on first call.
    """
    global _tables_created

    # Ensure directory exists (handles both 'data/shelfscan.db' and '/abs/path.db')
    db_dir = os.path.dirname(DATABASE_URL)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(DATABASE_URL, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    if not _tables_created:
        _create_schema(conn)
        _tables_created = True

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# -----------------------------------------------------------------------
# SCHEMA
# -----------------------------------------------------------------------

def _create_schema(conn):
    """Create all tables and indexes. Called once on first connection."""
    conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                email           TEXT    UNIQUE NOT NULL,
                password_hash   TEXT    NOT NULL,
                created_at      TEXT    NOT NULL DEFAULT (datetime('now','utc')),
                dietary_profile TEXT    DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS scan_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                barcode      TEXT    NOT NULL,
                name         TEXT,
                brand        TEXT,
                score        REAL,
                verdict      TEXT,
                nutriscore   TEXT,
                nova_group   INTEGER,
                scanned_at   TEXT    NOT NULL DEFAULT (datetime('now','utc')),
                product_json TEXT
            );

            CREATE TABLE IF NOT EXISTS favourites (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                barcode    TEXT    NOT NULL,
                name       TEXT,
                brand      TEXT,
                score      REAL,
                nutriscore TEXT,
                added_at   TEXT    NOT NULL DEFAULT (datetime('now','utc')),
                UNIQUE(user_id, barcode)
            );

            CREATE INDEX IF NOT EXISTS idx_scan_user
                ON scan_history(user_id, scanned_at DESC);
            CREATE INDEX IF NOT EXISTS idx_fav_user
                ON favourites(user_id);
            CREATE INDEX IF NOT EXISTS idx_fav_barcode
                ON favourites(user_id, barcode);
            CREATE INDEX IF NOT EXISTS idx_scan_barcode
                ON scan_history(barcode);
        """)
    conn.commit()


def create_tables():
    """Public function for explicit schema creation (e.g. from scripts)."""
    with _conn():
        pass  # _conn() handles schema creation on first call

# -----------------------------------------------------------------------
# INPUT VALIDATION HELPERS
# -----------------------------------------------------------------------

def _validate_score(score: float) -> float:
    """Clamp score to 0-100 range."""
    return max(0.0, min(100.0, float(score)))


def _validate_barcode(barcode: str) -> bool:
    """Barcode must be non-empty and 8-14 digits."""
    return bool(barcode) and barcode.strip().isdigit() and 8 <= len(barcode.strip()) <= 14


# -----------------------------------------------------------------------
# USER OPERATIONS
# -----------------------------------------------------------------------

def create_user(email: str, password_hash: str) -> int | None:
    """
    Insert a new user.
    Returns user_id on success, None if email already exists.
    Raises ValueError if password_hash is not a bcrypt hash.
    """
    if not password_hash.startswith('$2'):
        raise ValueError("password_hash must be a bcrypt hash")

    try:
        with _conn() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                (email.lower().strip(), password_hash)
            )
            return cur.lastrowid
    except sqlite3.IntegrityError:
        return None   # email already registered


def get_user_by_email(email: str) -> dict | None:
    """Return user row as dict or None if not found."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.lower().strip(),)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    """Return user row as dict or None if not found."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
        return dict(row) if row else None


def update_password(user_id: int, new_password_hash: str):
    """Update a user's password hash."""
    if not new_password_hash.startswith('$2'):
        raise ValueError("password_hash must be a bcrypt hash")
    with _conn() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (new_password_hash, user_id)
        )


def update_dietary_profile(user_id: int, profile: dict):
    """Update user's dietary preferences. Accepts dict, stores as JSON."""
    with _conn() as conn:
        conn.execute(
            "UPDATE users SET dietary_profile = ? WHERE id = ?",
            (json.dumps(profile), user_id)
        )


def delete_user(user_id: int):
    """
    Delete a user account and all associated data.
    CASCADE handles scan_history and favourites automatically.
    """
    with _conn() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


# -----------------------------------------------------------------------
# SCAN HISTORY
# -----------------------------------------------------------------------

def save_scan(user_id: int, product: dict, score: float, verdict: str):
    """
    Save a scan to history.
    Skips silently if barcode is missing or invalid.
    Deduplicates: if the same barcode was scanned today by the same user,
    updates the existing row instead of creating a duplicate.
    """
    barcode = (product.get('barcode') or '').strip()
    if not _validate_barcode(barcode):
        return   # don't save scanless or invalid rows

    with _conn() as conn:
        # Check if scanned today already (Bug 7 fix: dedup)
        existing = conn.execute("""
            SELECT id FROM scan_history
            WHERE user_id = ? AND barcode = ?
              AND date(scanned_at) = date('now', 'utc')
            LIMIT 1
        """, (user_id, barcode)).fetchone()

        if existing:
            # Update existing row with latest score/verdict
            conn.execute("""
                UPDATE scan_history
                SET score = ?, verdict = ?, scanned_at = datetime('now','utc'),
                    product_json = ?
                WHERE id = ?
            """, (
                _validate_score(score),
                verdict,
                json.dumps(product),
                existing['id'],
            ))
        else:
            conn.execute("""
                INSERT INTO scan_history
                  (user_id, barcode, name, brand, score, verdict,
                   nutriscore, nova_group, product_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                barcode,
                product.get('name', ''),
                product.get('brand', ''),
                _validate_score(score),
                verdict,
                product.get('nutriscore', ''),
                product.get('nova_group'),
                json.dumps(product),
            ))


def get_history(user_id: int, limit: int = 50, offset: int = 0) -> list[dict]:
    """
    Return scan history for a user, most recent first.
    Supports pagination via offset.
    """
    with _conn() as conn:
        rows = conn.execute("""
            SELECT * FROM scan_history
            WHERE user_id = ?
            ORDER BY scanned_at DESC
            LIMIT ? OFFSET ?
        """, (user_id, limit, offset)).fetchall()
        return [dict(r) for r in rows]


def get_history_count(user_id: int) -> int:
    """Return total number of scans for a user."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM scan_history WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        return row[0] if row else 0   # safe — never raises TypeError


def clear_history(user_id: int):
    """Delete all scan history for a user."""
    with _conn() as conn:
        conn.execute(
            "DELETE FROM scan_history WHERE user_id = ?",
            (user_id,)
        )


# -----------------------------------------------------------------------
# FAVOURITES
# -----------------------------------------------------------------------

def save_favourite(user_id: int, product: dict, score: float):
    """Save a product to favourites. Silently ignores duplicates."""
    barcode = (product.get('barcode') or '').strip()
    if not _validate_barcode(barcode):
        return

    try:
        with _conn() as conn:
            conn.execute("""
                INSERT INTO favourites
                  (user_id, barcode, name, brand, score, nutriscore)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                barcode,
                product.get('name', ''),
                product.get('brand', ''),
                _validate_score(score),
                product.get('nutriscore', ''),
            ))
    except sqlite3.IntegrityError:
        pass   # already in favourites


def remove_favourite(user_id: int, barcode: str):
    """Remove a product from favourites."""
    with _conn() as conn:
        conn.execute(
            "DELETE FROM favourites WHERE user_id = ? AND barcode = ?",
            (user_id, barcode)
        )


def get_favourites(user_id: int) -> list[dict]:
    """Return all favourites for a user, most recently added first."""
    with _conn() as conn:
        rows = conn.execute("""
            SELECT * FROM favourites
            WHERE user_id = ?
            ORDER BY added_at DESC
        """, (user_id,)).fetchall()
        return [dict(r) for r in rows]


def is_favourite(user_id: int, barcode: str) -> bool:
    """Return True if this barcode is in the user's favourites."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM favourites WHERE user_id = ? AND barcode = ?",
            (user_id, barcode)
        ).fetchone()
        return row is not None


# -----------------------------------------------------------------------
# STARTUP
# No longer auto-create tables at import time.
# Tables are created lazily on first _conn() call.