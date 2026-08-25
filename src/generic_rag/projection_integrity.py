"""Deterministic integrity values shared by projection and retrieval."""

from __future__ import annotations

import hashlib

from .contracts import (
    ChunkingPolicy,
    Document,
    DocumentIdentity,
    DocumentKey,
    EmbeddingIdentity,
    ProjectionCheckpoint,
    ProjectionIdentity,
    ProjectionManifest,
    ProjectionManifestEntry,
)
from .errors import ContractValidationError

__all__ = (
    "derive_source_digest",
    "derive_fragment_id",
    "derive_projection_checkpoint_token",
    "has_valid_projection_checkpoint",
)

_SOURCE_DIGEST_DOMAIN = "generic-rag:projection-source:v1"
_FRAGMENT_ID_DOMAIN = "generic-rag:fragment-id:v1"
_CHECKPOINT_DOMAIN = "generic-rag:projection-checkpoint:v1"


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


def _validate_document_key(value: object) -> DocumentKey:
    _require_exact_type("document key", value, DocumentKey)
    assert isinstance(value, DocumentKey)
    return DocumentKey(value.corpus_id, value.document_id)


def _validate_document_identity(value: object) -> DocumentIdentity:
    _require_exact_type("document identity", value, DocumentIdentity)
    assert isinstance(value, DocumentIdentity)
    return DocumentIdentity(
        _validate_document_key(value.key),
        value.revision_id,
    )


def _validate_document(value: object) -> Document:
    _require_exact_type("document", value, Document)
    assert isinstance(value, Document)
    return Document(
        _validate_document_identity(value.identity),
        value.text,
        value.attributes,
    )


def _validate_embedding_identity(value: object) -> EmbeddingIdentity:
    _require_exact_type("embedding identity", value, EmbeddingIdentity)
    assert isinstance(value, EmbeddingIdentity)
    return EmbeddingIdentity(value.model_id, value.dimensions)


def _validate_projection_identity(value: object) -> ProjectionIdentity:
    _require_exact_type("projection identity", value, ProjectionIdentity)
    assert isinstance(value, ProjectionIdentity)
    return ProjectionIdentity(
        value.schema_id,
        _validate_embedding_identity(value.embedding),
    )


def _validate_chunking(value: object) -> ChunkingPolicy:
    _require_exact_type("chunking", value, ChunkingPolicy)
    assert isinstance(value, ChunkingPolicy)
    return ChunkingPolicy(
        value.max_fragment_codepoints,
        value.overlap_codepoints,
    )


def _validate_manifest_entry(value: object) -> ProjectionManifestEntry:
    _require_exact_type("manifest entry", value, ProjectionManifestEntry)
    assert isinstance(value, ProjectionManifestEntry)
    return ProjectionManifestEntry(
        _validate_document_identity(value.document),
        value.source_digest,
        value.fragment_count,
    )


def _validate_checkpoint(value: object) -> ProjectionCheckpoint:
    _require_exact_type("checkpoint", value, ProjectionCheckpoint)
    assert isinstance(value, ProjectionCheckpoint)
    return ProjectionCheckpoint(
        value.corpus_id,
        _validate_projection_identity(value.projection),
        value.token,
    )


def _validate_manifest(value: object) -> ProjectionManifest:
    _require_exact_type("manifest", value, ProjectionManifest)
    assert isinstance(value, ProjectionManifest)
    _require_exact_type("manifest entries", value.entries, tuple)
    entries = tuple(_validate_manifest_entry(entry) for entry in value.entries)
    canonical = ProjectionManifest(
        value.corpus_id,
        _validate_projection_identity(value.projection),
        _validate_chunking(value.chunking),
        entries,
        _validate_checkpoint(value.checkpoint),
    )
    if entries != canonical.entries:
        raise ContractValidationError("manifest entries must be in canonical order")
    return canonical


def _sha256_fields(fields: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for field in fields:
        encoded = field.encode("utf-8", "surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"


def derive_source_digest(document: Document, /) -> str:
    """Derive the stable v1 digest of exact source text and attributes."""

    canonical = _validate_document(document)
    fields = [
        _SOURCE_DIGEST_DOMAIN,
        "text",
        canonical.text,
        "attributes_count",
        str(len(canonical.attributes)),
    ]
    for key, value in canonical.attributes:
        fields.extend(("attribute_key", key, "attribute_value", value))
    return _sha256_fields(tuple(fields))


def derive_fragment_id(
    document: DocumentIdentity,
    start: int,
    end: int,
    /,
) -> str:
    """Derive the stable v1 identity for one valid half-open source range."""

    canonical = _validate_document_identity(document)
    canonical_start = _require_nonnegative_integer("start", start)
    canonical_end = _require_nonnegative_integer("end", end)
    if canonical_start >= canonical_end:
        raise ContractValidationError("fragment range must satisfy start < end")
    return _sha256_fields(
        (
            _FRAGMENT_ID_DOMAIN,
            "corpus_id",
            canonical.key.corpus_id,
            "document_id",
            canonical.key.document_id,
            "revision_id",
            canonical.revision_id,
            "start",
            str(canonical_start),
            "end",
            str(canonical_end),
        )
    )


def derive_projection_checkpoint_token(
    corpus_id: str,
    projection: ProjectionIdentity,
    chunking: ChunkingPolicy,
    entries: tuple[ProjectionManifestEntry, ...],
    /,
) -> str:
    """Derive the stable v1 token for one canonical projection manifest."""

    canonical_corpus_id = _require_nonblank_string("corpus_id", corpus_id)
    canonical_projection = _validate_projection_identity(projection)
    canonical_chunking = _validate_chunking(chunking)
    _require_exact_type("entries", entries, tuple)
    canonical_entries = tuple(_validate_manifest_entry(entry) for entry in entries)
    ordering_witness = ProjectionManifest(
        canonical_corpus_id,
        canonical_projection,
        canonical_chunking,
        canonical_entries,
        ProjectionCheckpoint(
            canonical_corpus_id,
            canonical_projection,
            "integrity-validation",
        ),
    )
    if canonical_entries != ordering_witness.entries:
        raise ContractValidationError("manifest entries must be in canonical order")

    fields = [
        _CHECKPOINT_DOMAIN,
        "corpus_id",
        canonical_corpus_id,
        "schema_id",
        canonical_projection.schema_id,
        "embedding_model_id",
        canonical_projection.embedding.model_id,
        "embedding_dimensions",
        str(canonical_projection.embedding.dimensions),
        "max_fragment_codepoints",
        str(canonical_chunking.max_fragment_codepoints),
        "overlap_codepoints",
        str(canonical_chunking.overlap_codepoints),
        "entry_count",
        str(len(canonical_entries)),
    ]
    for entry in canonical_entries:
        fields.extend(
            (
                "document_id",
                entry.document.key.document_id,
                "revision_id",
                entry.document.revision_id,
                "source_digest",
                entry.source_digest,
                "fragment_count",
                str(entry.fragment_count),
            )
        )
    return _sha256_fields(tuple(fields))


def has_valid_projection_checkpoint(manifest: ProjectionManifest, /) -> bool:
    """Return whether a structurally valid manifest has its exact v1 token."""

    canonical = _validate_manifest(manifest)
    expected = derive_projection_checkpoint_token(
        canonical.corpus_id,
        canonical.projection,
        canonical.chunking,
        canonical.entries,
    )
    return canonical.checkpoint.token == expected
