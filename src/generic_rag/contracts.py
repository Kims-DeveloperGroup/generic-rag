"""Immutable, validated values shared by generic RAG workflows and ports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from .errors import ContractValidationError

__all__ = (
    "DocumentKey",
    "DocumentIdentity",
    "Document",
    "FragmentIdentity",
    "Fragment",
    "EmbeddingIdentity",
    "EmbeddingVector",
    "VectorRecord",
    "ProjectionIdentity",
    "ProjectionCheckpoint",
    "ProjectionOutcome",
    "ProjectionReceipt",
    "RetrievalQuery",
    "RetrievalOutcome",
    "RetrievalHit",
    "RetrievalResult",
)


def _require_exact_type(name: str, value: object, expected: type[object]) -> None:
    if type(value) is not expected:
        raise ContractValidationError(
            f"{name} must be exactly {expected.__name__}, not {type(value).__name__}"
        )


def _require_nonblank_string(name: str, value: object) -> str:
    _require_exact_type(name, value, str)
    assert isinstance(value, str)
    if not value or value.isspace():
        raise ContractValidationError(f"{name} must not be empty or whitespace-only")
    return value


def _require_nonnegative_integer(name: str, value: object) -> int:
    _require_exact_type(name, value, int)
    assert isinstance(value, int)
    if value < 0:
        raise ContractValidationError(f"{name} must be nonnegative")
    return value


def _require_positive_integer(name: str, value: object) -> int:
    result = _require_nonnegative_integer(name, value)
    if result == 0:
        raise ContractValidationError(f"{name} must be positive")
    return result


def _validate_attributes(name: str, attributes: object) -> tuple[tuple[str, str], ...]:
    _require_exact_type(name, attributes, tuple)
    assert isinstance(attributes, tuple)
    for index, pair in enumerate(attributes):
        if type(pair) is not tuple or len(pair) != 2:
            raise ContractValidationError(
                f"{name}[{index}] must be exactly a two-element tuple"
            )
        key, value = pair
        _require_exact_type(f"{name}[{index}][0]", key, str)
        _require_exact_type(f"{name}[{index}][1]", value, str)
    return attributes


@dataclass(frozen=True, slots=True)
class DocumentKey:
    """Stable opaque corpus/document identity independent of revision."""

    corpus_id: str
    document_id: str

    def __post_init__(self) -> None:
        _require_nonblank_string("corpus_id", self.corpus_id)
        _require_nonblank_string("document_id", self.document_id)


@dataclass(frozen=True, slots=True)
class DocumentIdentity:
    """A stable document key at one opaque authoritative revision."""

    key: DocumentKey
    revision_id: str

    def __post_init__(self) -> None:
        _require_exact_type("key", self.key, DocumentKey)
        _require_nonblank_string("revision_id", self.revision_id)


@dataclass(frozen=True, slots=True)
class Document:
    """Exact caller-supplied document text and ordered opaque attributes."""

    identity: DocumentIdentity
    text: str
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_exact_type("identity", self.identity, DocumentIdentity)
        _require_exact_type("text", self.text, str)
        _validate_attributes("attributes", self.attributes)


@dataclass(frozen=True, slots=True)
class FragmentIdentity:
    """Opaque fragment identity and half-open code-point source range."""

    document: DocumentIdentity
    fragment_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        _require_exact_type("document", self.document, DocumentIdentity)
        _require_nonblank_string("fragment_id", self.fragment_id)
        start = _require_nonnegative_integer("start", self.start)
        end = _require_nonnegative_integer("end", self.end)
        if start >= end:
            raise ContractValidationError("fragment range must satisfy start < end")


@dataclass(frozen=True, slots=True)
class Fragment:
    """An exact nonempty document slice carried as non-authoritative derived data."""

    identity: FragmentIdentity
    text: str
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_exact_type("identity", self.identity, FragmentIdentity)
        _require_exact_type("text", self.text, str)
        if not self.text:
            raise ContractValidationError("fragment text must not be empty")
        if len(self.text) != self.identity.end - self.identity.start:
            raise ContractValidationError(
                "fragment text length must equal its half-open code-point range"
            )
        _validate_attributes("attributes", self.attributes)


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    """Opaque embedding-model identity and its required vector dimension."""

    model_id: str
    dimensions: int

    def __post_init__(self) -> None:
        _require_nonblank_string("model_id", self.model_id)
        _require_positive_integer("dimensions", self.dimensions)


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    """A finite, nonempty embedding vector stored canonically as floats."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_exact_type("values", self.values, tuple)
        if not self.values:
            raise ContractValidationError("embedding vector must not be empty")

        canonical: list[float] = []
        for index, coordinate in enumerate(self.values):
            if type(coordinate) not in (int, float):
                raise ContractValidationError(
                    f"values[{index}] must be int or float, excluding bool"
                )
            try:
                numeric_coordinate = float(coordinate)
            except OverflowError:
                raise ContractValidationError(
                    f"values[{index}] must be finite"
                ) from None
            if not isfinite(numeric_coordinate):
                raise ContractValidationError(f"values[{index}] must be finite")
            canonical.append(numeric_coordinate)
        object.__setattr__(self, "values", tuple(canonical))


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """A fragment paired with the vector written for it."""

    fragment: Fragment
    embedding: EmbeddingVector

    def __post_init__(self) -> None:
        _require_exact_type("fragment", self.fragment, Fragment)
        _require_exact_type("embedding", self.embedding, EmbeddingVector)


@dataclass(frozen=True, slots=True)
class ProjectionIdentity:
    """Schema and embedding identity that determine projection compatibility."""

    schema_id: str
    embedding: EmbeddingIdentity

    def __post_init__(self) -> None:
        _require_nonblank_string("schema_id", self.schema_id)
        _require_exact_type("embedding", self.embedding, EmbeddingIdentity)


@dataclass(frozen=True, slots=True)
class ProjectionCheckpoint:
    """Opaque completed checkpoint bound to one corpus and projection identity."""

    corpus_id: str
    projection: ProjectionIdentity
    token: str

    def __post_init__(self) -> None:
        _require_nonblank_string("corpus_id", self.corpus_id)
        _require_exact_type("projection", self.projection, ProjectionIdentity)
        _require_nonblank_string("token", self.token)


class ProjectionOutcome(StrEnum):
    """Truthful completion state for a projection attempt."""

    COMPLETED = "completed"
    UNCHANGED = "unchanged"
    PARTIAL = "partial"
    FAILED = "failed"

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise ContractValidationError(f"{cls.__name__} value is not a defined member")


@dataclass(frozen=True, slots=True)
class ProjectionReceipt:
    """Projection counts and checkpoint subject to completion invariants."""

    corpus_id: str
    projection: ProjectionIdentity
    outcome: ProjectionOutcome
    attempted_documents: int
    completed_documents: int
    checkpoint: ProjectionCheckpoint | None

    def __post_init__(self) -> None:
        _require_nonblank_string("corpus_id", self.corpus_id)
        _require_exact_type("projection", self.projection, ProjectionIdentity)
        _require_exact_type("outcome", self.outcome, ProjectionOutcome)
        attempted = _require_nonnegative_integer(
            "attempted_documents", self.attempted_documents
        )
        completed = _require_nonnegative_integer(
            "completed_documents", self.completed_documents
        )
        if completed > attempted:
            raise ContractValidationError(
                "completed_documents must not exceed attempted_documents"
            )
        if self.checkpoint is not None:
            _require_exact_type("checkpoint", self.checkpoint, ProjectionCheckpoint)
            if self.checkpoint.corpus_id != self.corpus_id:
                raise ContractValidationError(
                    "checkpoint corpus_id must match the receipt corpus_id"
                )
            if self.checkpoint.projection != self.projection:
                raise ContractValidationError(
                    "checkpoint projection must match the receipt projection"
                )

        successful = self.outcome in (
            ProjectionOutcome.COMPLETED,
            ProjectionOutcome.UNCHANGED,
        )
        if successful:
            if completed != attempted:
                raise ContractValidationError(
                    "completed and unchanged receipts must complete every attempt"
                )
            if self.checkpoint is None:
                raise ContractValidationError(
                    "completed and unchanged receipts require a checkpoint"
                )
        elif self.outcome is ProjectionOutcome.PARTIAL:
            if not 0 < completed < attempted:
                raise ContractValidationError(
                    "partial receipts require 0 < completed_documents "
                    "< attempted_documents"
                )
            if self.checkpoint is not None:
                raise ContractValidationError(
                    "partial receipts must not claim a checkpoint"
                )
        else:
            if attempted == 0 or completed != 0:
                raise ContractValidationError(
                    "failed receipts require attempts and zero completed documents"
                )
            if self.checkpoint is not None:
                raise ContractValidationError(
                    "failed receipts must not claim a checkpoint"
                )


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """A bounded retrieval request for one opaque corpus."""

    corpus_id: str
    text: str
    hit_limit: int
    candidate_limit: int

    def __post_init__(self) -> None:
        _require_nonblank_string("corpus_id", self.corpus_id)
        _require_nonblank_string("text", self.text)
        hit_limit = _require_positive_integer("hit_limit", self.hit_limit)
        candidate_limit = _require_positive_integer(
            "candidate_limit", self.candidate_limit
        )
        if hit_limit > candidate_limit:
            raise ContractValidationError("hit_limit must not exceed candidate_limit")


class RetrievalOutcome(StrEnum):
    """Explicit availability and completion state of retrieval."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    FAILED = "failed"

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise ContractValidationError(f"{cls.__name__} value is not a defined member")


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """A final score-free ranked fragment."""

    fragment: Fragment
    rank: int

    def __post_init__(self) -> None:
        _require_exact_type("fragment", self.fragment, Fragment)
        _require_positive_integer("rank", self.rank)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A bounded, deduplicated, score-free retrieval result."""

    query: RetrievalQuery
    outcome: RetrievalOutcome
    hits: tuple[RetrievalHit, ...]
    truncated: bool

    def __post_init__(self) -> None:
        _require_exact_type("query", self.query, RetrievalQuery)
        _require_exact_type("outcome", self.outcome, RetrievalOutcome)
        _require_exact_type("hits", self.hits, tuple)
        _require_exact_type("truncated", self.truncated, bool)
        if len(self.hits) > self.query.hit_limit:
            raise ContractValidationError("hits must not exceed query.hit_limit")

        seen: set[FragmentIdentity] = set()
        for expected_rank, hit in enumerate(self.hits, start=1):
            _require_exact_type(f"hits[{expected_rank - 1}]", hit, RetrievalHit)
            if hit.rank != expected_rank:
                raise ContractValidationError("hit ranks must be contiguous from one")
            if hit.fragment.identity.document.key.corpus_id != self.query.corpus_id:
                raise ContractValidationError(
                    "every hit corpus_id must match the query corpus_id"
                )
            if hit.fragment.identity in seen:
                raise ContractValidationError(
                    "retrieval hits must have unique fragment identities"
                )
            seen.add(hit.fragment.identity)

        empty_outcomes = (
            RetrievalOutcome.UNAVAILABLE,
            RetrievalOutcome.STALE,
            RetrievalOutcome.FAILED,
        )
        if self.outcome in empty_outcomes:
            if self.hits:
                raise ContractValidationError(
                    "unavailable, stale, and failed results must not contain hits"
                )
            if self.truncated:
                raise ContractValidationError(
                    "unavailable, stale, and failed results must not be truncated"
                )
        elif self.outcome is RetrievalOutcome.PARTIAL and not self.hits:
            raise ContractValidationError("partial results require at least one hit")
