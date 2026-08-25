"""Deterministic bounded document projection orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import NoReturn, cast

from .contracts import (
    ChunkingPolicy,
    Document,
    DocumentIdentity,
    DocumentKey,
    EmbeddingIdentity,
    EmbeddingVector,
    Fragment,
    FragmentIdentity,
    ProjectionCheckpoint,
    ProjectionIdentity,
    ProjectionLimits,
    ProjectionManifest,
    ProjectionManifestEntry,
    ProjectionOutcome,
    ProjectionReceipt,
    ProjectionRequest,
    ProjectionResult,
    ProjectionStateAvailability,
    ProjectionStateSnapshot,
    ProjectionStateStatus,
    VectorRecord,
)
from .errors import (
    CollaborationError,
    ContractValidationError,
    StateCompatibilityError,
)
from .ports import Borrowed, Embedder, VectorIndexResetter, VectorIndexWriter
from .projection_integrity import (
    derive_fragment_id,
    derive_projection_checkpoint_token,
    derive_source_digest,
    has_valid_projection_checkpoint,
)

__all__ = (
    "ProjectionFailureStage",
    "ProjectionStateError",
    "ProjectionOperationError",
    "project_documents",
    "rebuild_projection",
)


class ProjectionFailureStage(StrEnum):
    """The collaborator stage at which a projection operation failed."""

    EMBEDDER_IDENTITY = "embedder_identity"
    EMBEDDING = "embedding"
    REPLACEMENT = "replacement"
    DELETION = "deletion"
    RESET = "reset"

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise ContractValidationError(f"{cls.__name__} value is not a defined member")


class ProjectionStateError(StateCompatibilityError):
    """Raised before effects when incremental projection cannot use state."""

    status: ProjectionStateStatus

    def __init__(self, status: ProjectionStateStatus) -> None:
        _require_exact_type("status", status, ProjectionStateStatus)
        self.status = status
        super().__init__("projection state is not compatible with incremental update")


class ProjectionOperationError(CollaborationError):
    """A content-free translation of one collaborator operation failure."""

    stage: ProjectionFailureStage
    affected_document: DocumentKey | None
    receipt: ProjectionReceipt | None

    def __init__(
        self,
        stage: ProjectionFailureStage,
        affected_document: DocumentKey | None,
        receipt: ProjectionReceipt | None,
    ) -> None:
        _require_exact_type("stage", stage, ProjectionFailureStage)
        if affected_document is not None:
            _validate_document_key(affected_document)
        if receipt is not None:
            _require_exact_type("receipt", receipt, ProjectionReceipt)
            if receipt.outcome not in (
                ProjectionOutcome.FAILED,
                ProjectionOutcome.PARTIAL,
            ):
                raise ContractValidationError(
                    "operation error receipt must be failed or partial"
                )
        self.stage = stage
        self.affected_document = affected_document
        self.receipt = receipt
        super().__init__(f"projection collaborator failed during {stage.value}")


@dataclass(frozen=True, slots=True)
class _PreparedDocument:
    source: Document
    fragments: tuple[Fragment, ...]
    entry: ProjectionManifestEntry


@dataclass(frozen=True, slots=True)
class _PreparedTarget:
    request: ProjectionRequest
    documents: tuple[_PreparedDocument, ...]
    manifest: ProjectionManifest


@dataclass(frozen=True, slots=True)
class _Mutation:
    key: DocumentKey
    replacement: _PreparedDocument | None


def _require_exact_type(name: str, value: object, expected: type[object]) -> None:
    if type(value) is not expected:
        raise ContractValidationError(
            f"{name} must be exactly {expected.__name__}, not {type(value).__name__}"
        )


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


def _validate_limits(value: object) -> ProjectionLimits:
    _require_exact_type("limits", value, ProjectionLimits)
    assert isinstance(value, ProjectionLimits)
    return ProjectionLimits(
        value.max_documents,
        value.max_document_codepoints,
        value.max_embedding_batch_size,
    )


def _validate_request(value: object) -> ProjectionRequest:
    _require_exact_type("request", value, ProjectionRequest)
    assert isinstance(value, ProjectionRequest)
    _require_exact_type("request documents", value.documents, tuple)
    documents = tuple(_validate_document(document) for document in value.documents)
    canonical = ProjectionRequest(
        value.corpus_id,
        _validate_projection_identity(value.projection),
        _validate_chunking(value.chunking),
        _validate_limits(value.limits),
        documents,
    )
    if documents != canonical.documents:
        raise ContractValidationError("request documents must be in canonical order")
    return canonical


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


def _validate_state(value: object) -> ProjectionStateSnapshot:
    _require_exact_type("state", value, ProjectionStateSnapshot)
    assert isinstance(value, ProjectionStateSnapshot)
    _require_exact_type(
        "state availability",
        value.availability,
        ProjectionStateAvailability,
    )
    manifest = None if value.manifest is None else _validate_manifest(value.manifest)
    return ProjectionStateSnapshot(value.availability, manifest)


def _validate_borrowed(name: str, value: object) -> None:
    _require_exact_type(name, value, Borrowed)


def _fragments(document: Document, chunking: ChunkingPolicy) -> tuple[Fragment, ...]:
    fragments: list[Fragment] = []
    start = 0
    text_length = len(document.text)
    while start < text_length:
        end = min(start + chunking.max_fragment_codepoints, text_length)
        identity = FragmentIdentity(
            document.identity,
            derive_fragment_id(document.identity, start, end),
            start,
            end,
        )
        fragments.append(
            Fragment(
                identity,
                document.text[start:end],
                document.attributes,
            )
        )
        if end == text_length:
            break
        start = end - chunking.overlap_codepoints
    return tuple(fragments)


def _prepare_target(request: ProjectionRequest) -> _PreparedTarget:
    prepared: list[_PreparedDocument] = []
    for document in request.documents:
        fragments = _fragments(document, request.chunking)
        entry = ProjectionManifestEntry(
            document.identity,
            derive_source_digest(document),
            len(fragments),
        )
        prepared.append(_PreparedDocument(document, fragments, entry))
    entries = tuple(item.entry for item in prepared)
    checkpoint = ProjectionCheckpoint(
        request.corpus_id,
        request.projection,
        derive_projection_checkpoint_token(
            request.corpus_id,
            request.projection,
            request.chunking,
            entries,
        ),
    )
    manifest = ProjectionManifest(
        request.corpus_id,
        request.projection,
        request.chunking,
        entries,
        checkpoint,
    )
    return _PreparedTarget(request, tuple(prepared), manifest)


def _state_status(
    state: ProjectionStateSnapshot,
    target: _PreparedTarget,
) -> ProjectionStateStatus:
    if state.availability is ProjectionStateAvailability.MISSING:
        return ProjectionStateStatus.MISSING
    if state.availability is ProjectionStateAvailability.CORRUPT:
        return ProjectionStateStatus.CORRUPT

    manifest = state.manifest
    assert manifest is not None
    if not has_valid_projection_checkpoint(manifest):
        return ProjectionStateStatus.CORRUPT
    if manifest.corpus_id != target.request.corpus_id:
        return ProjectionStateStatus.CORRUPT
    if manifest.projection.schema_id != target.request.projection.schema_id:
        return ProjectionStateStatus.SCHEMA_MISMATCH
    if manifest.projection.embedding != target.request.projection.embedding:
        return ProjectionStateStatus.EMBEDDING_MISMATCH

    target_by_key = {item.entry.document.key: item.entry for item in target.documents}
    for previous in manifest.entries:
        current = target_by_key.get(previous.document.key)
        if current is None or previous.document != current.document:
            continue
        if previous.source_digest != current.source_digest:
            return ProjectionStateStatus.CORRUPT
        if (
            manifest.chunking == target.request.chunking
            and previous.fragment_count != current.fragment_count
        ):
            return ProjectionStateStatus.CORRUPT

    if (
        manifest.chunking == target.request.chunking
        and manifest.entries == target.manifest.entries
    ):
        return ProjectionStateStatus.CURRENT
    return ProjectionStateStatus.STALE


def _incremental_plan(
    manifest: ProjectionManifest,
    target: _PreparedTarget,
) -> tuple[_Mutation, ...]:
    previous_by_key = {entry.document.key: entry for entry in manifest.entries}
    target_by_key = {item.entry.document.key: item for item in target.documents}
    mutations: list[_Mutation] = []

    for key, item in target_by_key.items():
        previous = previous_by_key.get(key)
        if manifest.chunking != target.request.chunking or previous != item.entry:
            mutations.append(_Mutation(key, item))
    for key in previous_by_key.keys() - target_by_key.keys():
        mutations.append(_Mutation(key, None))
    mutations.sort(key=lambda mutation: mutation.key.document_id)
    return tuple(mutations)


def _rebuild_plan(target: _PreparedTarget) -> tuple[_Mutation, ...]:
    return tuple(_Mutation(item.entry.document.key, item) for item in target.documents)


def _failure_receipt(
    target: _PreparedTarget,
    attempted: int,
    completed: int,
) -> ProjectionReceipt | None:
    if attempted == 0:
        return None
    outcome = ProjectionOutcome.FAILED if completed == 0 else ProjectionOutcome.PARTIAL
    return ProjectionReceipt(
        target.request.corpus_id,
        target.request.projection,
        outcome,
        attempted,
        completed,
        None,
    )


def _raise_operation_error(
    stage: ProjectionFailureStage,
    affected_document: DocumentKey | None,
    receipt: ProjectionReceipt | None,
    cause: Exception | None = None,
) -> NoReturn:
    error = ProjectionOperationError(stage, affected_document, receipt)
    if cause is None:
        raise error
    raise error from cause


def _require_embedder_identity(
    embedder: Embedder,
    expected: EmbeddingIdentity,
    receipt: ProjectionReceipt | None,
) -> None:
    try:
        identity = embedder.identity
    except Exception as exc:
        _raise_operation_error(
            ProjectionFailureStage.EMBEDDER_IDENTITY,
            None,
            receipt,
            exc,
        )
    try:
        validated = _validate_embedding_identity(identity)
    except ContractValidationError:
        _raise_operation_error(
            ProjectionFailureStage.EMBEDDER_IDENTITY,
            None,
            receipt,
        )
    if validated != expected:
        _raise_operation_error(
            ProjectionFailureStage.EMBEDDER_IDENTITY,
            None,
            receipt,
        )


def _validate_embeddings(
    value: object,
    expected_count: int,
    expected_dimensions: int,
) -> tuple[EmbeddingVector, ...]:
    if type(value) is not tuple or len(value) != expected_count:
        raise ContractValidationError(
            "embedder output must be an exact same-count tuple"
        )
    assert isinstance(value, tuple)
    for vector in value:
        _require_exact_type("embedding", vector, EmbeddingVector)
        assert isinstance(vector, EmbeddingVector)
        if (
            type(vector.values) is not tuple
            or len(vector.values) != expected_dimensions
        ):
            raise ContractValidationError(
                "embedding vector must have the requested dimensions"
            )
        if any(
            type(coordinate) is not float or not isfinite(coordinate)
            for coordinate in vector.values
        ):
            raise ContractValidationError(
                "embedding vector coordinates must be canonical finite floats"
            )
    return value


def _records_for_document(
    embedder: Embedder,
    item: _PreparedDocument,
    batch_size: int,
    target: _PreparedTarget,
    attempted: int,
    completed: int,
) -> tuple[VectorRecord, ...]:
    records: list[VectorRecord] = []
    for start in range(0, len(item.fragments), batch_size):
        fragments = item.fragments[start : start + batch_size]
        texts = tuple(fragment.text for fragment in fragments)
        try:
            output = embedder.embed(texts)
        except Exception as exc:
            _raise_operation_error(
                ProjectionFailureStage.EMBEDDING,
                item.entry.document.key,
                _failure_receipt(target, attempted, completed),
                exc,
            )
        try:
            vectors = _validate_embeddings(
                output,
                len(fragments),
                target.request.projection.embedding.dimensions,
            )
        except ContractValidationError:
            _raise_operation_error(
                ProjectionFailureStage.EMBEDDING,
                item.entry.document.key,
                _failure_receipt(target, attempted, completed),
            )
        records.extend(
            VectorRecord(fragment, vector)
            for fragment, vector in zip(fragments, vectors, strict=True)
        )
    return tuple(records)


def _execute_prevalidated_mutations(
    target: _PreparedTarget,
    mutations: tuple[_Mutation, ...],
    embedder: Embedder | None,
    writer_scope: Borrowed[VectorIndexWriter],
) -> None:
    if not mutations:
        return
    attempted = len(mutations)
    completed = 0
    with writer_scope as writer:
        for mutation in mutations:
            if mutation.replacement is None:
                try:
                    delete_document = cast(
                        Callable[[DocumentKey], object],
                        writer.delete_document,
                    )
                    command_result = delete_document(mutation.key)
                except Exception as exc:
                    _raise_operation_error(
                        ProjectionFailureStage.DELETION,
                        mutation.key,
                        _failure_receipt(target, attempted, completed),
                        exc,
                    )
                if command_result is not None:
                    _raise_operation_error(
                        ProjectionFailureStage.DELETION,
                        mutation.key,
                        _failure_receipt(target, attempted, completed),
                    )
            else:
                assert embedder is not None
                records = _records_for_document(
                    embedder,
                    mutation.replacement,
                    target.request.limits.max_embedding_batch_size,
                    target,
                    attempted,
                    completed,
                )
                try:
                    replace_document = cast(
                        Callable[
                            [DocumentIdentity, tuple[VectorRecord, ...]],
                            object,
                        ],
                        writer.replace_document,
                    )
                    command_result = replace_document(
                        mutation.replacement.entry.document,
                        records,
                    )
                except Exception as exc:
                    _raise_operation_error(
                        ProjectionFailureStage.REPLACEMENT,
                        mutation.key,
                        _failure_receipt(target, attempted, completed),
                        exc,
                    )
                if command_result is not None:
                    _raise_operation_error(
                        ProjectionFailureStage.REPLACEMENT,
                        mutation.key,
                        _failure_receipt(target, attempted, completed),
                    )
            completed += 1


def _execute_mutations(
    target: _PreparedTarget,
    mutations: tuple[_Mutation, ...],
    embedder_scope: Borrowed[Embedder],
    writer_scope: Borrowed[VectorIndexWriter],
) -> None:
    needs_embedder = any(mutation.replacement is not None for mutation in mutations)
    if not needs_embedder:
        _execute_prevalidated_mutations(target, mutations, None, writer_scope)
        return

    with embedder_scope as embedder:
        _require_embedder_identity(
            embedder,
            target.request.projection.embedding,
            _failure_receipt(target, len(mutations), 0),
        )
        _execute_prevalidated_mutations(target, mutations, embedder, writer_scope)


def _reset_projection(
    target: _PreparedTarget,
    attempted: int,
    resetter_scope: Borrowed[VectorIndexResetter],
) -> None:
    try:
        with resetter_scope as resetter:
            reset_corpus = cast(
                Callable[[str], object],
                resetter.reset_corpus,
            )
            command_result = reset_corpus(target.request.corpus_id)
    except Exception as exc:
        _raise_operation_error(
            ProjectionFailureStage.RESET,
            None,
            _failure_receipt(target, attempted, 0),
            exc,
        )
    if command_result is not None:
        _raise_operation_error(
            ProjectionFailureStage.RESET,
            None,
            _failure_receipt(target, attempted, 0),
        )


def _successful_result(
    target: _PreparedTarget,
    status_before: ProjectionStateStatus,
    outcome: ProjectionOutcome,
    attempted: int,
) -> ProjectionResult:
    receipt = ProjectionReceipt(
        target.request.corpus_id,
        target.request.projection,
        outcome,
        attempted,
        attempted,
        target.manifest.checkpoint,
    )
    return ProjectionResult(status_before, receipt, target.manifest)


def project_documents(
    request: ProjectionRequest,
    state: ProjectionStateSnapshot,
    embedder: Borrowed[Embedder],
    writer: Borrowed[VectorIndexWriter],
    /,
) -> ProjectionResult:
    """Project a compatible present state to the complete target.

    Incompatible state raises ``ProjectionStateError`` before collaborator effects.
    """

    canonical_request = _validate_request(request)
    canonical_state = _validate_state(state)
    _validate_borrowed("embedder", embedder)
    _validate_borrowed("writer", writer)
    target = _prepare_target(canonical_request)
    status = _state_status(canonical_state, target)
    if status is ProjectionStateStatus.CURRENT:
        return _successful_result(target, status, ProjectionOutcome.UNCHANGED, 0)
    if status is not ProjectionStateStatus.STALE:
        raise ProjectionStateError(status)

    manifest = canonical_state.manifest
    assert manifest is not None
    mutations = _incremental_plan(manifest, target)
    _execute_mutations(target, mutations, embedder, writer)
    return _successful_result(
        target,
        status,
        ProjectionOutcome.COMPLETED,
        len(mutations),
    )


def rebuild_projection(
    request: ProjectionRequest,
    state: ProjectionStateSnapshot,
    embedder: Borrowed[Embedder],
    writer: Borrowed[VectorIndexWriter],
    resetter: Borrowed[VectorIndexResetter],
    /,
) -> ProjectionResult:
    """Reset one corpus and write its complete canonical target projection."""

    canonical_request = _validate_request(request)
    canonical_state = _validate_state(state)
    _validate_borrowed("embedder", embedder)
    _validate_borrowed("writer", writer)
    _validate_borrowed("resetter", resetter)
    target = _prepare_target(canonical_request)
    status = _state_status(canonical_state, target)
    mutations = _rebuild_plan(target)

    if mutations:
        with embedder as embedder_resource:
            _require_embedder_identity(
                embedder_resource,
                target.request.projection.embedding,
                _failure_receipt(target, len(mutations), 0),
            )
            _reset_projection(target, len(mutations), resetter)
            _execute_prevalidated_mutations(
                target,
                mutations,
                embedder_resource,
                writer,
            )
    else:
        _reset_projection(target, 0, resetter)
    return _successful_result(
        target,
        status,
        ProjectionOutcome.COMPLETED,
        len(mutations),
    )
