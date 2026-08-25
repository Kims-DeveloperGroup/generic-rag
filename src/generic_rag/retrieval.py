"""Bounded semantic retrieval and deterministic rank-based hybrid fusion."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite

from .contracts import (
    ChunkingPolicy,
    DocumentIdentity,
    DocumentKey,
    EmbeddingIdentity,
    EmbeddingVector,
    Fragment,
    FragmentIdentity,
    ProjectionCheckpoint,
    ProjectionIdentity,
    ProjectionManifest,
    ProjectionManifestEntry,
    ProjectionStateAvailability,
    ProjectionStateSnapshot,
    RetrievalHit,
    RetrievalLimits,
    RetrievalOutcome,
    RetrievalQuery,
    RetrievalResult,
)
from .errors import ContractValidationError
from .ports import Borrowed, Embedder, LexicalRetriever, VectorIndexReader
from .projection_integrity import (
    derive_fragment_id,
    has_valid_projection_checkpoint,
)

__all__ = ("retrieve_semantic", "retrieve_hybrid")

_RRF_OFFSET = 60


@dataclass(frozen=True, slots=True)
class _RankedFragment:
    fragment: Fragment
    source_rank: int


@dataclass(frozen=True, slots=True)
class _BranchResult:
    all_candidates: tuple[_RankedFragment, ...]
    current_candidates: tuple[_RankedFragment, ...]
    stale: bool
    failed: bool


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


def _validate_query(value: object) -> RetrievalQuery:
    _require_exact_type("query", value, RetrievalQuery)
    assert isinstance(value, RetrievalQuery)
    return RetrievalQuery(
        value.corpus_id,
        value.text,
        value.hit_limit,
        value.candidate_limit,
    )


def _validate_limits(value: object) -> RetrievalLimits:
    _require_exact_type("limits", value, RetrievalLimits)
    assert isinstance(value, RetrievalLimits)
    return RetrievalLimits(value.max_query_codepoints)


def _validate_borrowed(name: str, value: object) -> None:
    _require_exact_type(name, value, Borrowed)


def _validate_embedding(
    value: object,
    expected_dimensions: int,
) -> EmbeddingVector:
    _require_exact_type("embedding", value, EmbeddingVector)
    assert isinstance(value, EmbeddingVector)
    _require_exact_type("embedding values", value.values, tuple)
    if len(value.values) != expected_dimensions:
        raise ContractValidationError(
            "embedding vector must have the requested dimensions"
        )
    if any(
        type(coordinate) is not float or not isfinite(coordinate)
        for coordinate in value.values
    ):
        raise ContractValidationError(
            "embedding vector coordinates must be canonical finite floats"
        )
    return EmbeddingVector(value.values)


def _validate_fragment(
    value: object,
    query: RetrievalQuery,
    chunking: ChunkingPolicy,
) -> Fragment:
    _require_exact_type("candidate", value, Fragment)
    assert isinstance(value, Fragment)
    identity_value = value.identity
    _require_exact_type("candidate identity", identity_value, FragmentIdentity)
    assert isinstance(identity_value, FragmentIdentity)
    identity = FragmentIdentity(
        _validate_document_identity(identity_value.document),
        identity_value.fragment_id,
        identity_value.start,
        identity_value.end,
    )
    canonical = Fragment(identity, value.text, value.attributes)
    if canonical.identity.document.key.corpus_id != query.corpus_id:
        raise ContractValidationError("candidate corpus_id must match the query")
    if len(canonical.text) > chunking.max_fragment_codepoints:
        raise ContractValidationError(
            "candidate width must not exceed the published chunk bound"
        )
    if canonical.identity.fragment_id != derive_fragment_id(
        canonical.identity.document,
        canonical.identity.start,
        canonical.identity.end,
    ):
        raise ContractValidationError(
            "candidate fragment_id must match its deterministic identity"
        )
    return canonical


def _validate_provider_candidates(
    value: object,
    query: RetrievalQuery,
    manifest: ProjectionManifest,
) -> _BranchResult:
    if type(value) is not tuple:
        raise ContractValidationError("retriever output must be exactly tuple")
    assert isinstance(value, tuple)
    if len(value) > query.candidate_limit:
        raise ContractValidationError(
            "retriever output must not exceed query.candidate_limit"
        )

    manifest_by_key = {entry.document.key: entry for entry in manifest.entries}
    first_by_identity: dict[FragmentIdentity, _RankedFragment] = {}
    ordered: list[_RankedFragment] = []
    for source_rank, candidate in enumerate(value, start=1):
        fragment = _validate_fragment(candidate, query, manifest.chunking)
        previous = first_by_identity.get(fragment.identity)
        if previous is not None:
            if previous.fragment != fragment:
                raise ContractValidationError(
                    "one provider returned conflicting candidate payloads"
                )
            continue
        ranked = _RankedFragment(fragment, source_rank)
        first_by_identity[fragment.identity] = ranked
        ordered.append(ranked)

    current: list[_RankedFragment] = []
    stale = False
    for candidate in ordered:
        published = manifest_by_key.get(candidate.fragment.identity.document.key)
        if (
            published is None
            or candidate.fragment.identity.document != published.document
        ):
            stale = True
            continue
        current.append(candidate)
    return _BranchResult(tuple(ordered), tuple(current), stale, False)


def _failed_branch() -> _BranchResult:
    return _BranchResult((), (), False, True)


def _stale_branch() -> _BranchResult:
    return _BranchResult((), (), True, False)


def _semantic_branch(
    query: RetrievalQuery,
    manifest: ProjectionManifest,
    embedder: Borrowed[Embedder],
    vector_reader: Borrowed[VectorIndexReader],
) -> _BranchResult:
    try:
        with embedder as embedder_resource:
            identity_value = embedder_resource.identity
            identity = _validate_embedding_identity(identity_value)
            if identity != manifest.projection.embedding:
                return _stale_branch()
            embedding_values = embedder_resource.embed((query.text,))
            if type(embedding_values) is not tuple or len(embedding_values) != 1:
                raise ContractValidationError(
                    "embedder output must be an exact one-vector tuple"
                )
            embedding = _validate_embedding(
                embedding_values[0],
                identity.dimensions,
            )
    except Exception:
        return _failed_branch()

    try:
        with vector_reader as reader_resource:
            candidates = reader_resource.search(query, embedding)
        return _validate_provider_candidates(candidates, query, manifest)
    except Exception:
        return _failed_branch()


def _lexical_branch(
    query: RetrievalQuery,
    manifest: ProjectionManifest,
    lexical_retriever: Borrowed[LexicalRetriever],
) -> _BranchResult:
    try:
        with lexical_retriever as lexical_resource:
            candidates = lexical_resource.search(query)
        return _validate_provider_candidates(candidates, query, manifest)
    except Exception:
        return _failed_branch()


def _terminal_result(
    query: RetrievalQuery,
    outcome: RetrievalOutcome,
) -> RetrievalResult:
    return RetrievalResult(query, outcome, (), False)


def _state_manifest_or_result(
    query: RetrievalQuery,
    state: ProjectionStateSnapshot,
) -> ProjectionManifest | RetrievalResult:
    if state.availability is ProjectionStateAvailability.MISSING:
        return _terminal_result(query, RetrievalOutcome.UNAVAILABLE)
    if state.availability is ProjectionStateAvailability.CORRUPT:
        return _terminal_result(query, RetrievalOutcome.FAILED)

    manifest = state.manifest
    assert manifest is not None
    if manifest.corpus_id != query.corpus_id:
        return _terminal_result(query, RetrievalOutcome.UNAVAILABLE)
    if not has_valid_projection_checkpoint(manifest):
        return _terminal_result(query, RetrievalOutcome.FAILED)
    if not manifest.entries:
        return _terminal_result(query, RetrievalOutcome.COMPLETE)
    return manifest


def _outcome_for_hits(
    has_hits: bool,
    *,
    stale: bool,
    failed: bool,
) -> RetrievalOutcome:
    if has_hits:
        if stale or failed:
            return RetrievalOutcome.PARTIAL
        return RetrievalOutcome.COMPLETE
    if failed:
        return RetrievalOutcome.FAILED
    if stale:
        return RetrievalOutcome.STALE
    return RetrievalOutcome.COMPLETE


def _ranked_result(
    query: RetrievalQuery,
    fragments: tuple[Fragment, ...],
    *,
    stale: bool,
    failed: bool,
) -> RetrievalResult:
    truncated = len(fragments) > query.hit_limit
    selected = fragments[: query.hit_limit]
    hits = tuple(
        RetrievalHit(fragment, rank) for rank, fragment in enumerate(selected, start=1)
    )
    outcome = _outcome_for_hits(bool(hits), stale=stale, failed=failed)
    return RetrievalResult(query, outcome, hits, truncated)


def _identity_order(fragment: Fragment) -> tuple[str, str, str, str, int, int]:
    identity = fragment.identity
    return (
        identity.document.key.corpus_id,
        identity.document.key.document_id,
        identity.document.revision_id,
        identity.fragment_id,
        identity.start,
        identity.end,
    )


def _has_cross_provider_conflict(
    semantic: _BranchResult,
    lexical: _BranchResult,
) -> bool:
    semantic_by_identity = {
        candidate.fragment.identity: candidate.fragment
        for candidate in semantic.all_candidates
    }
    for candidate in lexical.all_candidates:
        previous = semantic_by_identity.get(candidate.fragment.identity)
        if previous is not None and previous != candidate.fragment:
            return True
    return False


def _fuse_candidates(
    semantic: _BranchResult,
    lexical: _BranchResult,
) -> tuple[Fragment, ...]:
    fragments: dict[FragmentIdentity, Fragment] = {}
    scores: dict[FragmentIdentity, Fraction] = {}
    for branch in (semantic, lexical):
        for candidate in branch.current_candidates:
            identity = candidate.fragment.identity
            fragments.setdefault(identity, candidate.fragment)
            scores[identity] = scores.get(identity, Fraction()) + Fraction(
                1,
                _RRF_OFFSET + candidate.source_rank,
            )
    ranked_identities = sorted(
        scores,
        key=lambda identity: (
            -scores[identity],
            _identity_order(fragments[identity]),
        ),
    )
    return tuple(fragments[identity] for identity in ranked_identities)


def retrieve_semantic(
    query: RetrievalQuery,
    state: ProjectionStateSnapshot,
    limits: RetrievalLimits,
    embedder: Borrowed[Embedder],
    vector_reader: Borrowed[VectorIndexReader],
    /,
) -> RetrievalResult:
    """Retrieve bounded current fragments from one borrowed semantic index."""

    canonical_query = _validate_query(query)
    canonical_state = _validate_state(state)
    canonical_limits = _validate_limits(limits)
    _validate_borrowed("embedder", embedder)
    _validate_borrowed("vector_reader", vector_reader)
    if len(canonical_query.text) > canonical_limits.max_query_codepoints:
        raise ContractValidationError(
            "query text must not exceed limits.max_query_codepoints"
        )

    state_result = _state_manifest_or_result(canonical_query, canonical_state)
    if type(state_result) is RetrievalResult:
        return state_result
    assert isinstance(state_result, ProjectionManifest)
    branch = _semantic_branch(
        canonical_query,
        state_result,
        embedder,
        vector_reader,
    )
    fragments = tuple(candidate.fragment for candidate in branch.current_candidates)
    return _ranked_result(
        canonical_query,
        fragments,
        stale=branch.stale,
        failed=branch.failed,
    )


def retrieve_hybrid(
    query: RetrievalQuery,
    state: ProjectionStateSnapshot,
    limits: RetrievalLimits,
    embedder: Borrowed[Embedder],
    vector_reader: Borrowed[VectorIndexReader],
    lexical_retriever: Borrowed[LexicalRetriever],
    /,
) -> RetrievalResult:
    """Fuse bounded semantic and lexical ranks without comparing raw scores."""

    canonical_query = _validate_query(query)
    canonical_state = _validate_state(state)
    canonical_limits = _validate_limits(limits)
    _validate_borrowed("embedder", embedder)
    _validate_borrowed("vector_reader", vector_reader)
    _validate_borrowed("lexical_retriever", lexical_retriever)
    if len(canonical_query.text) > canonical_limits.max_query_codepoints:
        raise ContractValidationError(
            "query text must not exceed limits.max_query_codepoints"
        )

    state_result = _state_manifest_or_result(canonical_query, canonical_state)
    if type(state_result) is RetrievalResult:
        return state_result
    assert isinstance(state_result, ProjectionManifest)
    semantic = _semantic_branch(
        canonical_query,
        state_result,
        embedder,
        vector_reader,
    )
    lexical = _lexical_branch(
        canonical_query,
        state_result,
        lexical_retriever,
    )
    if _has_cross_provider_conflict(semantic, lexical):
        return _terminal_result(canonical_query, RetrievalOutcome.FAILED)
    fragments = _fuse_candidates(semantic, lexical)
    return _ranked_result(
        canonical_query,
        fragments,
        stale=semantic.stale or lexical.stale,
        failed=semantic.failed or lexical.failed,
    )
