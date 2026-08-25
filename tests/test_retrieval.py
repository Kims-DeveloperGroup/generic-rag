"""Behavioral tests for bounded semantic and deterministic hybrid retrieval."""

from __future__ import annotations

import inspect
import itertools
import json
import os
import subprocess
import sys
import textwrap
import unittest
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import cast

import generic_rag.retrieval as retrieval_module
from generic_rag.contracts import (
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
from generic_rag.errors import ContractValidationError
from generic_rag.ports import Borrowed, Embedder, LexicalRetriever, VectorIndexReader
from generic_rag.projection_integrity import (
    derive_fragment_id,
    derive_projection_checkpoint_token,
)
from generic_rag.retrieval import retrieve_hybrid, retrieve_semantic

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
_DEFAULT = object()


class _TupleSubclass(tuple[object, ...]):
    pass


class _ControlFlow(BaseException):
    pass


def _projection(
    *,
    model_id: str = "model-v1",
    dimensions: int = 2,
) -> ProjectionIdentity:
    return ProjectionIdentity(
        "schema-v1",
        EmbeddingIdentity(model_id, dimensions),
    )


def _document(
    document_id: str,
    *,
    corpus_id: str = "corpus",
    revision_id: str = "revision-1",
) -> DocumentIdentity:
    return DocumentIdentity(DocumentKey(corpus_id, document_id), revision_id)


def _manifest(
    documents: tuple[DocumentIdentity, ...],
    *,
    corpus_id: str = "corpus",
    projection: ProjectionIdentity | None = None,
    chunking: ChunkingPolicy | None = None,
    token: str | None = None,
) -> ProjectionManifest:
    selected_projection = projection or _projection()
    selected_chunking = chunking or ChunkingPolicy(4, 1)
    entries = tuple(
        sorted(
            (
                ProjectionManifestEntry(
                    document,
                    "sha256:" + f"{index:064x}",
                    1,
                )
                for index, document in enumerate(documents, start=1)
            ),
            key=lambda entry: entry.document.key.document_id,
        )
    )
    checkpoint_token = token or derive_projection_checkpoint_token(
        corpus_id,
        selected_projection,
        selected_chunking,
        entries,
    )
    return ProjectionManifest(
        corpus_id,
        selected_projection,
        selected_chunking,
        entries,
        ProjectionCheckpoint(corpus_id, selected_projection, checkpoint_token),
    )


def _present(manifest: ProjectionManifest) -> ProjectionStateSnapshot:
    return ProjectionStateSnapshot(ProjectionStateAvailability.PRESENT, manifest)


def _fragment(
    document: DocumentIdentity,
    *,
    text: str = "x",
    start: int = 0,
    fragment_id: str | None = None,
    attributes: tuple[tuple[str, str], ...] = (),
) -> Fragment:
    end = start + len(text)
    identity = FragmentIdentity(
        document,
        fragment_id or derive_fragment_id(document, start, end),
        start,
        end,
    )
    return Fragment(identity, text, attributes)


def _query(
    *,
    corpus_id: str = "corpus",
    text: str = "query",
    hit_limit: int = 4,
    candidate_limit: int = 8,
) -> RetrievalQuery:
    return RetrievalQuery(corpus_id, text, hit_limit, candidate_limit)


def _limits(max_query_codepoints: int = 100) -> RetrievalLimits:
    return RetrievalLimits(max_query_codepoints)


class _LifecycleResource:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.lifecycle_calls: list[str] = []

    def __enter__(self) -> _LifecycleResource:
        self.lifecycle_calls.append("enter")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.lifecycle_calls.append("exit")
        return True

    def close(self) -> None:
        self.lifecycle_calls.append("close")

    def shutdown(self) -> None:
        self.lifecycle_calls.append("shutdown")


class _FakeEmbedder(_LifecycleResource):
    def __init__(
        self,
        expected_identity: EmbeddingIdentity,
        events: list[str],
        *,
        identity_value: object = _DEFAULT,
        output: object = _DEFAULT,
        identity_failure: BaseException | None = None,
        embed_failure: BaseException | None = None,
    ) -> None:
        super().__init__(events)
        self.expected_identity = expected_identity
        self.identity_value = identity_value
        self.output = output
        self.identity_failure = identity_failure
        self.embed_failure = embed_failure
        self.identity_calls = 0
        self.embed_calls: list[tuple[str, ...]] = []

    @property
    def identity(self) -> EmbeddingIdentity:
        self.identity_calls += 1
        self.events.append("identity")
        if self.identity_failure is not None:
            raise self.identity_failure
        if self.identity_value is _DEFAULT:
            return self.expected_identity
        return cast(EmbeddingIdentity, self.identity_value)

    def embed(self, texts: tuple[str, ...], /) -> tuple[EmbeddingVector, ...]:
        self.embed_calls.append(texts)
        self.events.append("embed:" + "|".join(texts))
        if self.embed_failure is not None:
            raise self.embed_failure
        if self.output is _DEFAULT:
            return (EmbeddingVector((1.0, 2.0)),)
        return cast(tuple[EmbeddingVector, ...], self.output)


class _FakeVectorReader(_LifecycleResource):
    def __init__(
        self,
        events: list[str],
        output: object = (),
        *,
        failure: BaseException | None = None,
    ) -> None:
        super().__init__(events)
        self.output = output
        self.failure = failure
        self.calls: list[tuple[RetrievalQuery, EmbeddingVector]] = []

    def search(
        self,
        query: RetrievalQuery,
        embedding: EmbeddingVector,
        /,
    ) -> tuple[Fragment, ...]:
        self.calls.append((query, embedding))
        self.events.append("vector")
        if self.failure is not None:
            raise self.failure
        return cast(tuple[Fragment, ...], self.output)


class _FakeLexicalRetriever(_LifecycleResource):
    def __init__(
        self,
        events: list[str],
        output: object = (),
        *,
        failure: BaseException | None = None,
    ) -> None:
        super().__init__(events)
        self.output = output
        self.failure = failure
        self.calls: list[RetrievalQuery] = []

    def search(self, query: RetrievalQuery, /) -> tuple[Fragment, ...]:
        self.calls.append(query)
        self.events.append("lexical")
        if self.failure is not None:
            raise self.failure
        return cast(tuple[Fragment, ...], self.output)


def _borrow_embedder(value: _FakeEmbedder) -> Borrowed[Embedder]:
    return Borrowed(cast(Embedder, value))


def _borrow_vector(value: _FakeVectorReader) -> Borrowed[VectorIndexReader]:
    return Borrowed(cast(VectorIndexReader, value))


def _borrow_lexical(value: _FakeLexicalRetriever) -> Borrowed[LexicalRetriever]:
    return Borrowed(cast(LexicalRetriever, value))


def _hit_fragments(result: RetrievalResult) -> tuple[Fragment, ...]:
    return tuple(hit.fragment for hit in result.hits)


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


def _independent_fusion(
    semantic: tuple[Fragment, ...],
    lexical: tuple[Fragment, ...],
) -> tuple[Fragment, ...]:
    fragments: dict[FragmentIdentity, Fragment] = {}
    scores: dict[FragmentIdentity, Fraction] = {}
    for candidates in (semantic, lexical):
        seen: set[FragmentIdentity] = set()
        for rank, fragment in enumerate(candidates, start=1):
            if fragment.identity in seen:
                continue
            seen.add(fragment.identity)
            fragments.setdefault(fragment.identity, fragment)
            scores[fragment.identity] = scores.get(
                fragment.identity,
                Fraction(),
            ) + Fraction(1, 60 + rank)
    identities = sorted(
        scores,
        key=lambda identity: (
            -scores[identity],
            _identity_order(fragments[identity]),
        ),
    )
    return tuple(fragments[identity] for identity in identities)


class RetrievalPublicAndInputTests(unittest.TestCase):
    def test_exports_and_function_shapes_are_exact_and_owned(self) -> None:
        expected = ("retrieve_semantic", "retrieve_hybrid")
        signatures = {
            retrieve_semantic: (
                "query",
                "state",
                "limits",
                "embedder",
                "vector_reader",
            ),
            retrieve_hybrid: (
                "query",
                "state",
                "limits",
                "embedder",
                "vector_reader",
                "lexical_retriever",
            ),
        }

        self.assertEqual(retrieval_module.__all__, expected)
        for function, names in signatures.items():
            with self.subTest(function=function.__name__):
                self.assertEqual(function.__module__, "generic_rag.retrieval")
                self.assertFalse(inspect.iscoroutinefunction(function))
                parameters = tuple(
                    inspect.signature(
                        cast(Callable[..., object], function)
                    ).parameters.values()
                )
                self.assertEqual(tuple(item.name for item in parameters), names)
                self.assertTrue(
                    all(
                        item.kind is inspect.Parameter.POSITIONAL_ONLY
                        for item in parameters
                    )
                )

    def test_invalid_top_level_inputs_fail_before_any_collaborator_effect(self) -> None:
        document = _document("document")
        state = _present(_manifest((document,)))
        query = _query(hit_limit=1, candidate_limit=1)
        events: list[str] = []
        embedder = _FakeEmbedder(_projection().embedding, events)
        vector = _FakeVectorReader(events, (_fragment(document),))
        operations: tuple[Callable[[], object], ...] = (
            lambda: retrieve_semantic(
                cast(RetrievalQuery, object()),
                state,
                _limits(),
                _borrow_embedder(embedder),
                _borrow_vector(vector),
            ),
            lambda: retrieve_semantic(
                query,
                cast(ProjectionStateSnapshot, object()),
                _limits(),
                _borrow_embedder(embedder),
                _borrow_vector(vector),
            ),
            lambda: retrieve_semantic(
                query,
                state,
                cast(RetrievalLimits, object()),
                _borrow_embedder(embedder),
                _borrow_vector(vector),
            ),
            lambda: retrieve_semantic(
                query,
                state,
                _limits(),
                cast(Borrowed[Embedder], object()),
                _borrow_vector(vector),
            ),
            lambda: retrieve_semantic(
                query,
                state,
                _limits(),
                _borrow_embedder(embedder),
                cast(Borrowed[VectorIndexReader], object()),
            ),
            lambda: retrieve_hybrid(
                query,
                state,
                _limits(),
                _borrow_embedder(embedder),
                _borrow_vector(vector),
                cast(Borrowed[LexicalRetriever], object()),
            ),
        )

        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ContractValidationError):
                    operation()
                self.assertEqual(events, [])

    def test_corrupted_frozen_query_and_state_fail_before_effects(self) -> None:
        document = _document("document")
        query = _query(hit_limit=1, candidate_limit=1)
        object.__setattr__(query, "candidate_limit", True)
        state = _present(_manifest((document,)))
        malformed_state = _present(_manifest((document,)))
        assert malformed_state.manifest is not None
        object.__setattr__(malformed_state.manifest, "entries", [])

        for malformed_query, malformed_snapshot in (
            (query, state),
            (_query(hit_limit=1, candidate_limit=1), malformed_state),
        ):
            events: list[str] = []
            embedder = _FakeEmbedder(_projection().embedding, events)
            vector = _FakeVectorReader(events)
            with self.subTest(query=malformed_query, state=malformed_snapshot):
                with self.assertRaises(ContractValidationError):
                    retrieve_semantic(
                        malformed_query,
                        malformed_snapshot,
                        _limits(),
                        _borrow_embedder(embedder),
                        _borrow_vector(vector),
                    )
                self.assertEqual(events, [])

    def test_query_codepoint_cap_is_independent_and_checked_before_effects(
        self,
    ) -> None:
        document = _document("document")
        fragment = _fragment(document)
        state = _present(_manifest((document,)))
        exact_text = "😀e\u0301\0"
        self.assertEqual(len(exact_text), 4)

        events: list[str] = []
        embedder = _FakeEmbedder(_projection().embedding, events)
        vector = _FakeVectorReader(events, (fragment,))
        result = retrieve_semantic(
            _query(text=exact_text, hit_limit=1, candidate_limit=1),
            state,
            RetrievalLimits(4),
            _borrow_embedder(embedder),
            _borrow_vector(vector),
        )
        self.assertIs(result.outcome, RetrievalOutcome.COMPLETE)
        self.assertEqual(embedder.embed_calls, [(exact_text,)])

        blocked_events: list[str] = []
        blocked_embedder = _FakeEmbedder(_projection().embedding, blocked_events)
        blocked_vector = _FakeVectorReader(blocked_events, (fragment,))
        with self.assertRaises(ContractValidationError):
            retrieve_semantic(
                _query(text=exact_text + "Z", hit_limit=1, candidate_limit=1),
                state,
                RetrievalLimits(4),
                _borrow_embedder(blocked_embedder),
                _borrow_vector(blocked_vector),
            )
        self.assertEqual(blocked_events, [])


class RetrievalStateAndEffectTests(unittest.TestCase):
    def test_terminal_state_outcomes_never_touch_collaborators(self) -> None:
        document = _document("document")
        valid = _manifest((document,))
        invalid_checkpoint = ProjectionManifest(
            valid.corpus_id,
            valid.projection,
            valid.chunking,
            valid.entries,
            ProjectionCheckpoint(valid.corpus_id, valid.projection, "wrong-token"),
        )
        other_document = _document("document", corpus_id="other")
        cases = (
            (
                ProjectionStateSnapshot(ProjectionStateAvailability.MISSING, None),
                RetrievalOutcome.UNAVAILABLE,
            ),
            (
                ProjectionStateSnapshot(ProjectionStateAvailability.CORRUPT, None),
                RetrievalOutcome.FAILED,
            ),
            (
                _present(_manifest((other_document,), corpus_id="other")),
                RetrievalOutcome.UNAVAILABLE,
            ),
            (_present(invalid_checkpoint), RetrievalOutcome.FAILED),
            (_present(_manifest(())), RetrievalOutcome.COMPLETE),
        )

        for state, expected in cases:
            events: list[str] = []
            embedder = _FakeEmbedder(
                _projection().embedding,
                events,
                identity_failure=AssertionError("must not run"),
            )
            vector = _FakeVectorReader(
                events,
                failure=AssertionError("must not run"),
            )
            lexical = _FakeLexicalRetriever(
                events,
                failure=AssertionError("must not run"),
            )
            with self.subTest(outcome=expected):
                result = retrieve_hybrid(
                    _query(),
                    state,
                    _limits(),
                    _borrow_embedder(embedder),
                    _borrow_vector(vector),
                    _borrow_lexical(lexical),
                )
                self.assertIs(result.outcome, expected)
                self.assertEqual(result.hits, ())
                self.assertFalse(result.truncated)
                self.assertEqual(events, [])

    def test_semantic_and_hybrid_effect_order_and_call_bounds_are_exact(self) -> None:
        semantic_document = _document("semantic")
        lexical_document = _document("lexical")
        state = _present(_manifest((semantic_document, lexical_document)))
        query = _query(hit_limit=2, candidate_limit=2)

        semantic_events: list[str] = []
        semantic_embedder = _FakeEmbedder(_projection().embedding, semantic_events)
        semantic_vector = _FakeVectorReader(
            semantic_events,
            (_fragment(semantic_document),),
        )
        semantic_result = retrieve_semantic(
            query,
            state,
            _limits(),
            _borrow_embedder(semantic_embedder),
            _borrow_vector(semantic_vector),
        )
        self.assertEqual(semantic_events, ["identity", "embed:query", "vector"])
        self.assertEqual(semantic_embedder.identity_calls, 1)
        self.assertEqual(semantic_embedder.embed_calls, [(query.text,)])
        self.assertEqual(len(semantic_vector.calls), 1)
        self.assertEqual(semantic_vector.calls[0][0], query)
        self.assertEqual(semantic_vector.calls[0][1], EmbeddingVector((1.0, 2.0)))
        self.assertIs(semantic_result.outcome, RetrievalOutcome.COMPLETE)

        hybrid_events: list[str] = []
        hybrid_embedder = _FakeEmbedder(_projection().embedding, hybrid_events)
        hybrid_vector = _FakeVectorReader(
            hybrid_events,
            (_fragment(semantic_document),),
        )
        hybrid_lexical = _FakeLexicalRetriever(
            hybrid_events,
            (_fragment(lexical_document),),
        )
        hybrid_result = retrieve_hybrid(
            query,
            state,
            _limits(),
            _borrow_embedder(hybrid_embedder),
            _borrow_vector(hybrid_vector),
            _borrow_lexical(hybrid_lexical),
        )
        self.assertEqual(
            hybrid_events,
            ["identity", "embed:query", "vector", "lexical"],
        )
        self.assertEqual(hybrid_embedder.identity_calls, 1)
        self.assertEqual(hybrid_embedder.embed_calls, [(query.text,)])
        self.assertEqual(len(hybrid_vector.calls), 1)
        self.assertEqual(hybrid_lexical.calls, [query])
        self.assertIs(hybrid_result.outcome, RetrievalOutcome.COMPLETE)
        self.assertEqual(tuple(hit.rank for hit in hybrid_result.hits), (1, 2))


class RetrievalProviderValidationTests(unittest.TestCase):
    def test_embedder_identity_and_vector_outputs_are_strictly_validated(self) -> None:
        document = _document("document")
        state = _present(_manifest((document,)))
        malformed_identity = EmbeddingIdentity("model-v1", 2)
        object.__setattr__(malformed_identity, "dimensions", True)
        wrong_dimensions = EmbeddingVector((1.0,))
        noncanonical_number = EmbeddingVector((1.0, 2.0))
        object.__setattr__(noncanonical_number, "values", (1, 2.0))
        nonfinite = EmbeddingVector((1.0, 2.0))
        object.__setattr__(nonfinite, "values", (float("nan"), 2.0))
        malformed_container = EmbeddingVector((1.0, 2.0))
        object.__setattr__(malformed_container, "values", [1.0, 2.0])
        cases = (
            ("identity type", object(), _DEFAULT),
            ("identity shape", malformed_identity, _DEFAULT),
            ("output list", _DEFAULT, [EmbeddingVector((1.0, 2.0))]),
            ("output empty", _DEFAULT, ()),
            (
                "output count",
                _DEFAULT,
                (EmbeddingVector((1.0, 2.0)), EmbeddingVector((3.0, 4.0))),
            ),
            ("vector type", _DEFAULT, (object(),)),
            ("dimensions", _DEFAULT, (wrong_dimensions,)),
            ("coordinate type", _DEFAULT, (noncanonical_number,)),
            ("nonfinite", _DEFAULT, (nonfinite,)),
            ("values container", _DEFAULT, (malformed_container,)),
        )

        for label, identity_value, output in cases:
            events: list[str] = []
            embedder = _FakeEmbedder(
                _projection().embedding,
                events,
                identity_value=identity_value,
                output=output,
            )
            vector = _FakeVectorReader(events, (_fragment(document),))
            with self.subTest(case=label):
                result = retrieve_semantic(
                    _query(hit_limit=1, candidate_limit=1),
                    state,
                    _limits(),
                    _borrow_embedder(embedder),
                    _borrow_vector(vector),
                )
                self.assertIs(result.outcome, RetrievalOutcome.FAILED)
                self.assertEqual(result.hits, ())
                self.assertFalse(result.truncated)
                self.assertLessEqual(embedder.identity_calls, 1)
                self.assertLessEqual(len(embedder.embed_calls), 1)
                self.assertEqual(vector.calls, [])

    def test_vector_and_lexical_candidate_outputs_share_strict_validation(self) -> None:
        documents = tuple(_document(name) for name in ("a", "b", "c"))
        state = _present(_manifest(documents))
        valid = tuple(_fragment(document) for document in documents)
        wrong_corpus = _fragment(_document("wrong", corpus_id="other"))
        too_wide = _fragment(documents[0], text="abcde")
        wrong_id = _fragment(documents[0], fragment_id="wrong-fragment-id")
        malformed_outputs: tuple[tuple[str, object], ...] = (
            ("list", list(valid[:1])),
            ("tuple subclass", _TupleSubclass(valid[:1])),
            ("candidate count", valid),
            ("candidate type", (object(),)),
            ("wrong corpus", (wrong_corpus,)),
            ("fragment width", (too_wide,)),
            ("fragment id", (wrong_id,)),
        )
        query = _query(hit_limit=2, candidate_limit=2)

        for pathway in ("vector", "lexical"):
            for label, malformed in malformed_outputs:
                events: list[str] = []
                embedder = _FakeEmbedder(_projection().embedding, events)
                vector = _FakeVectorReader(
                    events,
                    malformed if pathway == "vector" else (),
                )
                lexical = _FakeLexicalRetriever(
                    events,
                    malformed if pathway == "lexical" else (),
                )
                with self.subTest(pathway=pathway, case=label):
                    if pathway == "vector":
                        result = retrieve_semantic(
                            query,
                            state,
                            _limits(),
                            _borrow_embedder(embedder),
                            _borrow_vector(vector),
                        )
                    else:
                        result = retrieve_hybrid(
                            query,
                            state,
                            _limits(),
                            _borrow_embedder(embedder),
                            _borrow_vector(vector),
                            _borrow_lexical(lexical),
                        )
                    self.assertIs(result.outcome, RetrievalOutcome.FAILED)
                    self.assertEqual(result.hits, ())
                    self.assertFalse(result.truncated)
                    if pathway == "vector":
                        self.assertEqual(len(vector.calls), 1)
                        self.assertEqual(lexical.calls, [])
                    else:
                        self.assertEqual(len(vector.calls), 1)
                        self.assertEqual(len(lexical.calls), 1)

    def test_revision_and_document_membership_staleness_preserve_current_hits(
        self,
    ) -> None:
        published_a = _document("a", revision_id="current")
        published_c = _document("c", revision_id="current")
        stale_revision = _fragment(_document("a", revision_id="old"))
        absent_document = _fragment(_document("b", revision_id="current"))
        current = _fragment(published_c)
        state = _present(_manifest((published_a, published_c)))
        events: list[str] = []
        embedder = _FakeEmbedder(_projection().embedding, events)
        vector = _FakeVectorReader(
            events,
            (stale_revision, absent_document, current),
        )

        result = retrieve_semantic(
            _query(hit_limit=2, candidate_limit=3),
            state,
            _limits(),
            _borrow_embedder(embedder),
            _borrow_vector(vector),
        )

        self.assertIs(result.outcome, RetrievalOutcome.PARTIAL)
        self.assertEqual(_hit_fragments(result), (current,))
        self.assertEqual(result.hits[0], RetrievalHit(current, 1))
        self.assertFalse(result.truncated)

    def test_embedder_identity_mismatch_is_stale_without_embedding_or_vector_read(
        self,
    ) -> None:
        document = _document("document")
        state = _present(_manifest((document,)))
        events: list[str] = []
        embedder = _FakeEmbedder(
            _projection().embedding,
            events,
            identity_value=EmbeddingIdentity("other-model", 2),
        )
        vector = _FakeVectorReader(events, (_fragment(document),))

        result = retrieve_semantic(
            _query(hit_limit=1, candidate_limit=1),
            state,
            _limits(),
            _borrow_embedder(embedder),
            _borrow_vector(vector),
        )

        self.assertIs(result.outcome, RetrievalOutcome.STALE)
        self.assertEqual(events, ["identity"])
        self.assertEqual(embedder.embed_calls, [])
        self.assertEqual(vector.calls, [])


class RetrievalDeduplicationAndFusionTests(unittest.TestCase):
    def test_identical_provider_duplicates_are_deduplicated_without_truncation(
        self,
    ) -> None:
        documents = (_document("a"), _document("b"))
        first, second = tuple(_fragment(document) for document in documents)
        events: list[str] = []
        result = retrieve_semantic(
            _query(hit_limit=2, candidate_limit=3),
            _present(_manifest(documents)),
            _limits(),
            _borrow_embedder(_FakeEmbedder(_projection().embedding, events)),
            _borrow_vector(_FakeVectorReader(events, (first, first, second))),
        )

        self.assertIs(result.outcome, RetrievalOutcome.COMPLETE)
        self.assertEqual(_hit_fragments(result), (first, second))
        self.assertFalse(result.truncated)

    def test_within_and_cross_provider_payload_conflicts_fail_content_free(
        self,
    ) -> None:
        document = _document("document")
        original = _fragment(document, attributes=(("variant", "one"),))
        conflict = _fragment(document, attributes=(("sentinel-secret", "two"),))
        state = _present(_manifest((document,)))

        events: list[str] = []
        within = retrieve_semantic(
            _query(hit_limit=2, candidate_limit=2),
            state,
            _limits(),
            _borrow_embedder(_FakeEmbedder(_projection().embedding, events)),
            _borrow_vector(_FakeVectorReader(events, (original, conflict))),
        )
        self.assertIs(within.outcome, RetrievalOutcome.FAILED)
        self.assertEqual(within.hits, ())
        self.assertNotIn("sentinel-secret", repr(within))

        cross_events: list[str] = []
        cross = retrieve_hybrid(
            _query(hit_limit=1, candidate_limit=1),
            state,
            _limits(),
            _borrow_embedder(_FakeEmbedder(_projection().embedding, cross_events)),
            _borrow_vector(_FakeVectorReader(cross_events, (original,))),
            _borrow_lexical(_FakeLexicalRetriever(cross_events, (conflict,))),
        )
        self.assertIs(cross.outcome, RetrievalOutcome.FAILED)
        self.assertEqual(cross.hits, ())
        self.assertFalse(cross.truncated)
        self.assertNotIn("sentinel-secret", repr(cross))

    def test_exact_cross_provider_duplicate_is_one_fused_hit(self) -> None:
        document = _document("document")
        fragment = _fragment(document)
        events: list[str] = []
        result = retrieve_hybrid(
            _query(hit_limit=1, candidate_limit=1),
            _present(_manifest((document,))),
            _limits(),
            _borrow_embedder(_FakeEmbedder(_projection().embedding, events)),
            _borrow_vector(_FakeVectorReader(events, (fragment,))),
            _borrow_lexical(_FakeLexicalRetriever(events, (fragment,))),
        )

        self.assertIs(result.outcome, RetrievalOutcome.COMPLETE)
        self.assertEqual(_hit_fragments(result), (fragment,))
        self.assertFalse(result.truncated)

    def test_duplicate_gaps_retain_original_provider_ranks(self) -> None:
        documents = tuple(_document(name) for name in ("a", "b", "c", "d"))
        by_id = {
            document.key.document_id: _fragment(document) for document in documents
        }
        semantic = (by_id["d"], by_id["d"], by_id["b"])
        lexical = (by_id["a"], by_id["c"])
        events: list[str] = []
        result = retrieve_hybrid(
            _query(hit_limit=4, candidate_limit=4),
            _present(_manifest(documents)),
            _limits(),
            _borrow_embedder(_FakeEmbedder(_projection().embedding, events)),
            _borrow_vector(_FakeVectorReader(events, semantic)),
            _borrow_lexical(_FakeLexicalRetriever(events, lexical)),
        )

        self.assertEqual(_hit_fragments(result), _independent_fusion(semantic, lexical))
        document_ids = tuple(
            hit.fragment.identity.document.key.document_id for hit in result.hits
        )
        self.assertLess(document_ids.index("c"), document_ids.index("b"))

    def test_rrf_uses_exact_fraction_offset_sixty_witness(self) -> None:
        semantic_ids = (
            "A",
            "s02",
            "s03",
            "s04",
            "B",
            "s06",
            "s07",
            "s08",
            "s09",
            "s10",
            "s11",
            "s12",
            "s13",
        )
        lexical_ids = (
            "l01",
            "l02",
            "l03",
            "l04",
            "l05",
            "l06",
            "l07",
            "B",
            "l09",
            "l10",
            "l11",
            "l12",
            "A",
        )
        all_ids = tuple(sorted(set((*semantic_ids, *lexical_ids))))
        documents = tuple(_document(document_id) for document_id in all_ids)
        fragments = {
            document.key.document_id: _fragment(document) for document in documents
        }
        semantic = tuple(fragments[document_id] for document_id in semantic_ids)
        lexical = tuple(fragments[document_id] for document_id in lexical_ids)
        events: list[str] = []
        result = retrieve_hybrid(
            _query(hit_limit=13, candidate_limit=13),
            _present(_manifest(documents)),
            _limits(),
            _borrow_embedder(_FakeEmbedder(_projection().embedding, events)),
            _borrow_vector(_FakeVectorReader(events, semantic)),
            _borrow_lexical(_FakeLexicalRetriever(events, lexical)),
        )

        expected = _independent_fusion(semantic, lexical)
        self.assertEqual(_hit_fragments(result), expected[:13])
        self.assertEqual(
            tuple(
                hit.fragment.identity.document.key.document_id
                for hit in result.hits[:2]
            ),
            ("A", "B"),
        )
        self.assertGreater(
            Fraction(1, 60 + 1) + Fraction(1, 60 + 13),
            Fraction(1, 60 + 5) + Fraction(1, 60 + 8),
        )
        self.assertLess(
            Fraction(1, 61 + 1) + Fraction(1, 61 + 13),
            Fraction(1, 61 + 5) + Fraction(1, 61 + 8),
        )
        self.assertTrue(result.truncated)

    def test_unicode_ties_and_branch_permutation_use_identity_order(self) -> None:
        decomposed = _document("e\u0301")
        composed = _document("é")
        first = _fragment(decomposed)
        second = _fragment(composed)
        state = _present(_manifest((decomposed, composed)))

        outputs: list[tuple[Fragment, ...]] = []
        for semantic, lexical in (((first,), (second,)), ((second,), (first,))):
            events: list[str] = []
            result = retrieve_hybrid(
                _query(hit_limit=2, candidate_limit=2),
                state,
                _limits(),
                _borrow_embedder(_FakeEmbedder(_projection().embedding, events)),
                _borrow_vector(_FakeVectorReader(events, semantic)),
                _borrow_lexical(_FakeLexicalRetriever(events, lexical)),
            )
            outputs.append(_hit_fragments(result))

        self.assertEqual(outputs, [(first, second), (first, second)])

    def test_all_small_provider_permutations_match_fraction_oracle(self) -> None:
        documents = tuple(_document(name) for name in ("a", "b", "c"))
        fragments = tuple(_fragment(document) for document in documents)
        state = _present(_manifest(documents))

        for semantic in itertools.permutations(fragments):
            for lexical in itertools.permutations(fragments):
                events: list[str] = []
                with self.subTest(
                    semantic=tuple(
                        item.identity.document.key.document_id for item in semantic
                    ),
                    lexical=tuple(
                        item.identity.document.key.document_id for item in lexical
                    ),
                ):
                    result = retrieve_hybrid(
                        _query(hit_limit=3, candidate_limit=3),
                        state,
                        _limits(),
                        _borrow_embedder(
                            _FakeEmbedder(_projection().embedding, events)
                        ),
                        _borrow_vector(_FakeVectorReader(events, semantic)),
                        _borrow_lexical(_FakeLexicalRetriever(events, lexical)),
                    )
                    self.assertEqual(
                        _hit_fragments(result),
                        _independent_fusion(semantic, lexical),
                    )

    def test_truncation_reflects_post_validation_unique_current_candidates(
        self,
    ) -> None:
        documents = tuple(_document(name) for name in ("a", "b", "c"))
        current = tuple(_fragment(document) for document in documents)

        events: list[str] = []
        truncated = retrieve_semantic(
            _query(hit_limit=2, candidate_limit=3),
            _present(_manifest(documents)),
            _limits(),
            _borrow_embedder(_FakeEmbedder(_projection().embedding, events)),
            _borrow_vector(_FakeVectorReader(events, current)),
        )
        self.assertEqual(len(truncated.hits), 2)
        self.assertTrue(truncated.truncated)

        stale = _fragment(_document("b", revision_id="old"))
        filtered_events: list[str] = []
        filtered = retrieve_semantic(
            _query(hit_limit=1, candidate_limit=3),
            _present(_manifest(documents[:2])),
            _limits(),
            _borrow_embedder(_FakeEmbedder(_projection().embedding, filtered_events)),
            _borrow_vector(
                _FakeVectorReader(filtered_events, (current[0], current[0], stale))
            ),
        )
        self.assertIs(filtered.outcome, RetrievalOutcome.PARTIAL)
        self.assertEqual(_hit_fragments(filtered), (current[0],))
        self.assertFalse(filtered.truncated)


class RetrievalFailureAndLifecycleTests(unittest.TestCase):
    def assert_no_lifecycle(self, *resources: _LifecycleResource) -> None:
        for resource in resources:
            self.assertEqual(resource.lifecycle_calls, [])

    def test_failure_and_stale_precedence_is_truthful(self) -> None:
        document = _document("document")
        fragment = _fragment(document)
        state = _present(_manifest((document,)))
        wrong_identity = EmbeddingIdentity("other-model", 2)
        cases = (
            (wrong_identity, None, (), RetrievalOutcome.STALE),
            (
                wrong_identity,
                RuntimeError("lexical sentinel"),
                (),
                RetrievalOutcome.FAILED,
            ),
            (wrong_identity, None, (fragment,), RetrievalOutcome.PARTIAL),
            (
                RuntimeError("identity sentinel"),
                None,
                (fragment,),
                RetrievalOutcome.PARTIAL,
            ),
            (_DEFAULT, None, (), RetrievalOutcome.COMPLETE),
        )

        for semantic_state, lexical_failure, lexical_output, expected in cases:
            events: list[str] = []
            embedder = _FakeEmbedder(
                _projection().embedding,
                events,
                identity_value=(
                    semantic_state
                    if isinstance(semantic_state, EmbeddingIdentity)
                    else _DEFAULT
                ),
                identity_failure=(
                    semantic_state
                    if isinstance(semantic_state, BaseException)
                    else None
                ),
            )
            vector = _FakeVectorReader(events)
            lexical = _FakeLexicalRetriever(
                events,
                lexical_output,
                failure=lexical_failure,
            )
            with self.subTest(expected=expected, semantic=semantic_state):
                result = retrieve_hybrid(
                    _query(hit_limit=1, candidate_limit=1),
                    state,
                    _limits(),
                    _borrow_embedder(embedder),
                    _borrow_vector(vector),
                    _borrow_lexical(lexical),
                )
                self.assertIs(result.outcome, expected)
                self.assertEqual(
                    bool(result.hits), expected is RetrievalOutcome.PARTIAL
                )
                self.assertNotIn("sentinel", repr(result))

    def test_ordinary_exceptions_are_content_free_and_never_retried(self) -> None:
        document = _document("document")
        state = _present(_manifest((document,)))
        stages = ("identity", "embedding", "vector", "lexical")
        expected_calls = {
            "identity": (1, 0, 0, 0),
            "embedding": (1, 1, 0, 0),
            "vector": (1, 1, 1, 0),
            "lexical": (1, 1, 1, 1),
        }

        for stage in stages:
            failure = RuntimeError(f"secret-{stage}-sentinel")
            events: list[str] = []
            embedder = _FakeEmbedder(
                _projection().embedding,
                events,
                identity_failure=failure if stage == "identity" else None,
                embed_failure=failure if stage == "embedding" else None,
            )
            vector = _FakeVectorReader(
                events,
                failure=failure if stage == "vector" else None,
            )
            lexical = _FakeLexicalRetriever(
                events,
                failure=failure if stage == "lexical" else None,
            )
            with self.subTest(stage=stage):
                if stage == "lexical":
                    result = retrieve_hybrid(
                        _query(hit_limit=1, candidate_limit=1),
                        state,
                        _limits(),
                        _borrow_embedder(embedder),
                        _borrow_vector(vector),
                        _borrow_lexical(lexical),
                    )
                else:
                    result = retrieve_semantic(
                        _query(hit_limit=1, candidate_limit=1),
                        state,
                        _limits(),
                        _borrow_embedder(embedder),
                        _borrow_vector(vector),
                    )
                self.assertIs(result.outcome, RetrievalOutcome.FAILED)
                self.assertEqual(result.hits, ())
                self.assertFalse(result.truncated)
                self.assertNotIn(str(failure), repr(result))
                self.assertEqual(
                    (
                        embedder.identity_calls,
                        len(embedder.embed_calls),
                        len(vector.calls),
                        len(lexical.calls),
                    ),
                    expected_calls[stage],
                )
                self.assert_no_lifecycle(embedder, vector, lexical)

    def test_base_exceptions_propagate_exactly_once_without_lifecycle_actions(
        self,
    ) -> None:
        document = _document("document")
        state = _present(_manifest((document,)))
        expected_calls = {
            "identity": (1, 0, 0, 0),
            "embedding": (1, 1, 0, 0),
            "vector": (1, 1, 1, 0),
            "lexical": (1, 1, 1, 1),
        }
        for stage in ("identity", "embedding", "vector", "lexical"):
            failure = _ControlFlow(f"control-{stage}")
            events: list[str] = []
            embedder = _FakeEmbedder(
                _projection().embedding,
                events,
                identity_failure=failure if stage == "identity" else None,
                embed_failure=failure if stage == "embedding" else None,
            )
            vector = _FakeVectorReader(
                events,
                failure=failure if stage == "vector" else None,
            )
            lexical = _FakeLexicalRetriever(
                events,
                failure=failure if stage == "lexical" else None,
            )
            with self.subTest(stage=stage):
                with self.assertRaises(_ControlFlow) as raised:
                    if stage == "lexical":
                        retrieve_hybrid(
                            _query(hit_limit=1, candidate_limit=1),
                            state,
                            _limits(),
                            _borrow_embedder(embedder),
                            _borrow_vector(vector),
                            _borrow_lexical(lexical),
                        )
                    else:
                        retrieve_semantic(
                            _query(hit_limit=1, candidate_limit=1),
                            state,
                            _limits(),
                            _borrow_embedder(embedder),
                            _borrow_vector(vector),
                        )
                self.assertIs(raised.exception, failure)
                self.assertEqual(
                    (
                        embedder.identity_calls,
                        len(embedder.embed_calls),
                        len(vector.calls),
                        len(lexical.calls),
                    ),
                    expected_calls[stage],
                )
                self.assert_no_lifecycle(embedder, vector, lexical)

    def test_success_never_enters_closes_or_shuts_down_borrowed_resources(self) -> None:
        document = _document("document")
        fragment = _fragment(document)
        events: list[str] = []
        embedder = _FakeEmbedder(_projection().embedding, events)
        vector = _FakeVectorReader(events, (fragment,))
        lexical = _FakeLexicalRetriever(events, (fragment,))

        result = retrieve_hybrid(
            _query(hit_limit=1, candidate_limit=1),
            _present(_manifest((document,))),
            _limits(),
            _borrow_embedder(embedder),
            _borrow_vector(vector),
            _borrow_lexical(lexical),
        )

        self.assertIs(result.outcome, RetrievalOutcome.COMPLETE)
        self.assert_no_lifecycle(embedder, vector, lexical)


class RetrievalProcessDeterminismTests(unittest.TestCase):
    def test_hash_seed_does_not_change_hybrid_rank_order(self) -> None:
        script = textwrap.dedent(
            f"""
            import json
            import sys
            sys.path.insert(0, {os.fspath(_SOURCE_ROOT)!r})
            from generic_rag.contracts import (
                ChunkingPolicy, DocumentIdentity, DocumentKey,
                EmbeddingIdentity, EmbeddingVector, Fragment, FragmentIdentity,
                ProjectionCheckpoint, ProjectionIdentity, ProjectionManifest,
                ProjectionManifestEntry, ProjectionStateAvailability,
                ProjectionStateSnapshot, RetrievalLimits, RetrievalQuery,
            )
            from generic_rag.ports import Borrowed
            from generic_rag.projection_integrity import (
                derive_fragment_id, derive_projection_checkpoint_token,
            )
            from generic_rag.retrieval import retrieve_hybrid

            projection = ProjectionIdentity(
                'schema-v1', EmbeddingIdentity('model-v1', 2)
            )
            chunking = ChunkingPolicy(4, 1)
            documents = tuple(
                DocumentIdentity(DocumentKey('corpus', name), 'revision-1')
                for name in ('e\\u0301', 'é', '한')
            )
            entries = tuple(
                ProjectionManifestEntry(
                    document, 'sha256:' + format(index, '064x'), 1
                )
                for index, document in enumerate(documents, start=1)
            )
            token = derive_projection_checkpoint_token(
                'corpus', projection, chunking, entries
            )
            manifest = ProjectionManifest(
                'corpus', projection, chunking, entries,
                ProjectionCheckpoint('corpus', projection, token),
            )
            fragments = tuple(
                Fragment(
                    FragmentIdentity(
                        document, derive_fragment_id(document, 0, 1), 0, 1
                    ),
                    'x',
                )
                for document in documents
            )

            class E:
                identity = projection.embedding
                def embed(self, texts, /):
                    return (EmbeddingVector((1.0, 2.0)),)
            class V:
                def search(self, query, embedding, /):
                    return (fragments[2], fragments[0], fragments[1])
            class L:
                def search(self, query, /):
                    return (fragments[1], fragments[0], fragments[2])

            result = retrieve_hybrid(
                RetrievalQuery('corpus', 'query', 3, 3),
                ProjectionStateSnapshot(ProjectionStateAvailability.PRESENT, manifest),
                RetrievalLimits(100), Borrowed(E()), Borrowed(V()), Borrowed(L()),
            )
            print(json.dumps([
                hit.fragment.identity.document.key.document_id
                for hit in result.hits
            ], ensure_ascii=False))
            """
        )
        outputs: list[list[str]] = []
        for seed in ("1", "17", "987654"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                (sys.executable, "-B", "-c", script),
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            outputs.append(cast(list[str], json.loads(completed.stdout)))

        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])
        self.assertEqual(outputs[0], ["é", "한", "e\u0301"])


if __name__ == "__main__":
    unittest.main()
