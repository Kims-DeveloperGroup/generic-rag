"""Public exception types for generic RAG contracts and collaboration."""

__all__ = (
    "GenericRagError",
    "ContractValidationError",
    "CollaborationError",
    "StateCompatibilityError",
)


class GenericRagError(Exception):
    """Base class for errors intentionally exposed by generic-rag."""


class ContractValidationError(GenericRagError):
    """Raised when a public immutable value violates its contract."""


class CollaborationError(GenericRagError):
    """Raised when orchestration translates a collaborator operation failure."""


class StateCompatibilityError(GenericRagError):
    """Raised when derived state has an incompatible projection identity."""
