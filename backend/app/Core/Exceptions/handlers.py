from fastapi import Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.Http.DTOs.error_schemas import APIErrorResponse, ErrorObject, ErrorDetail, ErrorSource
from typing import Any, Dict, List

def register_exception_handlers(app: Any):
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        error_object = ErrorObject(
            status=exc.status_code,
            code=getattr(exc, "code", "API_ERROR"),
            title=str(exc.detail),
            docs_url=f"https://docs.tito.ai/errors#{getattr(exc, 'code', 'API_ERROR')}"
        )
        
        response = APIErrorResponse(
            error=error_object,
            _links={
                "self": {"href": str(request.url), "method": request.method},
                "home": {"href": "/", "method": "GET"},
                "docs": {"href": "/docs", "method": "GET"}
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=response.model_dump(by_alias=True, exclude_none=True)
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        details = []
        for error in exc.errors():
            # error["loc"] is a tuple like ('body', 'ticker') or ('query', 'limit')
            pointer = "/" + "/".join(str(p) for p in error["loc"])
            details.append(ErrorDetail(
                code="INVALID_ATTRIBUTE",
                title=error["msg"],
                source=ErrorSource(pointer=pointer)
            ))

        error_object = ErrorObject(
            status=422,
            code="VALIDATION_ERROR",
            title="Validation failed for the request payload.",
            docs_url="https://docs.tito.ai/errors#VALIDATION_ERROR",
            details=details
        )

        response = APIErrorResponse(
            error=error_object,
            _links={
                "self": {"href": str(request.url), "method": request.method},
                "home": {"href": "/", "method": "GET"},
                "docs": {"href": "/docs", "method": "GET"}
            }
        )

        return JSONResponse(
            status_code=422,
            content=response.model_dump(by_alias=True, exclude_none=True)
        )
