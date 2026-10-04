"""Optional API key check and a simple rate limiter (bonus feature)."""

import secrets
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

from app.config import get_api_key, get_rate_limit

WINDOW_SECONDS = 60

# client address -> times of its recent requests (kept in memory only)
request_times = defaultdict(deque)


def reset_rate_limiter():
    """Forget all past requests (used by tests)."""
    request_times.clear()


def check_api_key(x_api_key):
    """If API_KEY is set, the client must send the same value in X-API-Key."""
    expected_key = get_api_key()
    if not expected_key:
        return  # protection is switched off
    if not secrets.compare_digest(x_api_key or "", expected_key):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


def check_rate_limit(client_address):
    """Allow only a fixed number of requests per minute for each client."""
    now = time.time()
    times = request_times[client_address]

    # Drop requests older than one window
    while times and now - times[0] > WINDOW_SECONDS:
        times.popleft()

    if len(times) >= get_rate_limit():
        raise HTTPException(
            status_code=429, detail="Too many requests. Try again soon."
        )
    times.append(now)


def check_access(request: Request, x_api_key: str | None = Header(default=None)):
    """FastAPI dependency: run both checks before a prediction endpoint."""
    check_api_key(x_api_key)
    client_address = request.client.host if request.client else "unknown"
    check_rate_limit(client_address)
