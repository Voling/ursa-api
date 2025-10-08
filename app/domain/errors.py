"""Domain error hierarchy for clean exception handling."""
from __future__ import annotations


class DomainError(Exception):
    """Base for all domain errors."""


class NotFoundError(DomainError):
    """Resource not found."""


class ValidationError(DomainError):
    """Invalid input or state."""


class ConflictError(DomainError):
    """Resource conflict (e.g., duplicate name)."""

