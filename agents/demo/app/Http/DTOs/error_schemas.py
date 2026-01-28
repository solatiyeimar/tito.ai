"""
Error response schemas for the Tito.ai Agent Service.

This module defines the standardized error response structure
used across the application, matching the custom exception handlers.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ErrorLink(BaseModel):
    """Link object for HATEOAS navigation in errors."""
    href: str
    method: str


class ErrorSource(BaseModel):
    """Pointer to the source of the error (e.g., specific field)."""
    pointer: Optional[str] = None
    parameter: Optional[str] = None


class ErrorDetail(BaseModel):
    """Detailed breakdown of a specific error (used in validation errors)."""
    code: str = Field(..., description="Machine-readable error code for this specific detail.")
    title: str = Field(..., description="Human-readable error description.")
    source: Optional[ErrorSource] = Field(None, description="Location of the error.")


class ErrorObject(BaseModel):
    """Core error information."""
    status: int = Field(..., description="HTTP status code.")
    code: str = Field(..., description="High-level machine-readable error code.")
    title: str = Field(..., description="Brief human-readable summary of the problem.")
    docs_url: Optional[str] = Field(None, description="Link to documentation for this error.")
    details: Optional[List[ErrorDetail]] = Field(None, description="List of specific error details (e.g., validation failures).")


class APIErrorResponse(BaseModel):
    """
    Standardized API Error Response.
    
    Replaces the default FastAPI/Pydantic validation error structure.
    """
    error: ErrorObject
    links: Optional[Dict[str, ErrorLink]] = Field(None, alias="_links", description="Navigation links related to the error.")
    debug: Optional[Dict[str, Any]] = Field(None, description="Debug information (only in debug mode).")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "error": {
                    "status": 422,
                    "code": "VALIDATION_ERROR",
                    "title": "Validation failed for the request payload.",
                    "docs_url": "https://docs.tito.ai/errors#VALIDATION_ERROR",
                    "details": [
                        {
                            "code": "INVALID_ATTRIBUTE",
                            "title": "field required",
                            "source": {"pointer": "/query/ticker"}
                        }
                    ]
                },
                "_links": {
                    "self": {"href": "/assistants", "method": "GET"},
                    "home": {"href": "/", "method": "GET"},
                    "docs": {"href": "/docs", "method": "GET"}
                }
            }
        }
