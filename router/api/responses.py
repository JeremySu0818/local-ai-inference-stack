from fastapi.responses import JSONResponse


def error(message, status, kind="invalid_request_error"):
    return JSONResponse(
        {"error": {"message": str(message), "type": kind, "param": None, "code": None}},
        status_code=status,
    )
