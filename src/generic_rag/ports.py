"""Synchronous injected collaborator contracts and borrowing semantics."""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Generic, Literal, Protocol, TypeVar, runtime_checkable

from .contracts import (
    DocumentIdentity,
    DocumentKey,
    EmbeddingIdentity,
    EmbeddingVector,
    Fragment,
    RetrievalQuery,
    VectorRecord,
)

__all__ = (
    "Borrowed",
    "Embedder",
    "VectorIndexWriter",
    "VectorIndexResetter",
    "VectorIndexReader",
    "LexicalRetriever",
)

_T_co = TypeVar("_T_co", covariant=True)


@dataclass(frozen=True, slots=True)
class Borrowed(Generic[_T_co]):
    """A no-op scope that marks a resource as caller-owned.

    Entry returns the exact wrapped resource. Exit never calls lifecycle methods
    on it and never suppresses an exception.
    """

    resource: _T_co

    def __enter__(self) -> _T_co:
        return self.resource

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        return False


@runtime_checkable
class Embedder(Protocol):
    """Synchronously embeds an ordered tuple without owning provider lifecycle."""

    @property
    def identity(self) -> EmbeddingIdentity:
        """Return the exact model identity used for produced vectors."""
        ...

    def embed(self, texts: tuple[str, ...], /) -> tuple[EmbeddingVector, ...]:
        """Return one same-order vector per input text."""
        ...


@runtime_checkable
class VectorIndexWriter(Protocol):
    """Synchronously replaces and deletes complete document projections."""

    def replace_document(
        self,
        document: DocumentIdentity,
        records: tuple[VectorRecord, ...],
        /,
    ) -> None:
        """Replace all derived vectors for the document's stable key."""
        ...

    def delete_document(self, document: DocumentKey, /) -> None:
        """Delete every derived revision for a stable document key."""
        ...


@runtime_checkable
class VectorIndexResetter(Protocol):
    """Synchronously removes every projected document for one corpus."""

    def reset_corpus(self, corpus_id: str, /) -> None:
        """Remove the complete derived vector projection for the corpus."""
        ...


@runtime_checkable
class VectorIndexReader(Protocol):
    """Returns best-first vector candidates without exposing raw scores."""

    def search(
        self,
        query: RetrievalQuery,
        embedding: EmbeddingVector,
        /,
    ) -> tuple[Fragment, ...]:
        """Return at most query.candidate_limit fragments in provider rank order."""
        ...


@runtime_checkable
class LexicalRetriever(Protocol):
    """Returns best-first lexical candidates without exposing raw scores."""

    def search(self, query: RetrievalQuery, /) -> tuple[Fragment, ...]:
        """Return at most query.candidate_limit fragments in provider rank order."""
        ...
