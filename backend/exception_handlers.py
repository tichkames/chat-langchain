import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


async def default_error_handler(_: Request, exception: Exception) -> JSONResponse:
    """High level exception handler for all exceptions
    """
    logging.exception(exception)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={'message': 'Unhandled Internal Server Error'}
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Add exception handlers to FastAPI app
    """
    app.add_exception_handler(Exception, default_error_handler)
