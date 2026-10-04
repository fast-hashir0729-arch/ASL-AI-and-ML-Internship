"""Turns validation errors into clear 400 responses with an error category."""

from fastapi import Request
from fastapi.responses import JSONResponse

# Pydantic error types that mean "the value is not allowed"
INVALID_VALUE_TYPES = [
    "greater_than_equal",
    "less_than_equal",
    "literal_error",
    "too_short",
    "too_long",
]


def get_category(error_type):
    """Map a pydantic error type to one of our simple categories."""
    if error_type == "missing":
        return "missing_field"
    if error_type == "json_invalid":
        return "malformed_json"
    if error_type == "extra_forbidden":
        return "unknown_field"
    if error_type in INVALID_VALUE_TYPES:
        return "invalid_value"
    return "wrong_type"


def get_field_name(location):
    """Build a readable field name like 'records.0.age' from the error location."""
    parts = [str(part) for part in location if part != "body"]
    return ".".join(parts) or "body"


async def validation_error_handler(request: Request, error):
    """Return HTTP 400 (instead of FastAPI's default 422) with details."""
    details = []
    for item in error.errors():
        details.append(
            {
                "field": get_field_name(item["loc"]),
                "category": get_category(item["type"]),
                "problem": item["msg"],
            }
        )

    categories = {detail["category"] for detail in details}
    main_category = (
        details[0]["category"] if len(categories) == 1 else "multiple_errors"
    )

    body = {
        "error": main_category,
        "message": "The request body is not valid. See details.",
        "details": details,
    }
    return JSONResponse(status_code=400, content=body)


# Readable error names for non-validation errors
HTTP_ERROR_NAMES = {
    401: "unauthorized",
    404: "not_found",
    405: "method_not_allowed",
    429: "rate_limited",
    503: "model_unavailable",
}


async def http_error_handler(request: Request, error):
    """Give every HTTP error the same JSON shape."""
    name = HTTP_ERROR_NAMES.get(error.status_code, "http_error")
    body = {"error": name, "message": str(error.detail)}
    return JSONResponse(status_code=error.status_code, content=body)


async def unexpected_error_handler(request: Request, error):
    """Last safety net: never leak a stack trace to the client."""
    body = {"error": "internal_error", "message": "Something went wrong on the server."}
    return JSONResponse(status_code=500, content=body)
