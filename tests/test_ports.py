"""Contract tests and deterministic witnesses for injected collaborator ports."""

from __future__ import annotations

import inspect
import unittest
from dataclasses import FrozenInstanceError, fields
from types import TracebackType
from typing import get_type_hints

import generic_rag.ports as ports
from generic_rag.contracts import (
    DocumentIdentity,
    DocumentKey,
    EmbeddingIdentity,
    EmbeddingVector,
    Fragment,
    FragmentIdentity,
    RetrievalQuery,
    VectorRecord,
)
from generic_rag.ports import (
    Borrowed,
    Embedder,
    LexicalRetriever,
    VectorIndexReader,
    VectorIndexWriter,
)


def _document_identity(
    *,
    corpus_id: str = "corpus",
    document_id: str = "document",
    revision_id: str = "revision",
) -> DocumentIdentity:
    return DocumentIdentity(
        DocumentKey(corpus_id, document_id),
        revision_id,
    )


def _fragment(
    fragment_id: str,
    *,
    corpus_id: str = "corpus",
    document_id: str = "document",
) -> Fragment:
    document = _document_identity(
        corpus_id=corpus_id,
        document_id=document_id,
    )
    identity = FragmentIdentity(document, fragment_id, 0, 1)
    return Fragment(identity, "x")


def _query(
    *,
    corpus_id: str = "corpus",
    candidate_limit: int = 3,
) -> RetrievalQuery:
    return RetrievalQuery(corpus_id, "query", 2, candidate_limit)


class _FakeEmbedder:
    def __init__(self) -> None:
        self._identity = EmbeddingIdentity("fake-model", 2)
        self.calls: list[tuple[str, ...]] = []

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    def embed(self, texts: tuple[str, ...], /) -> tuple[EmbeddingVector, ...]:
        self.calls.append(texts)
        return tuple(
            EmbeddingVector((index, len(text))) for index, text in enumerate(texts)
        )


class _ShapeOnlyInvalidEmbedder:
    @property
    def identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity("shape-only", 2)

    def embed(self, texts: tuple[str, ...], /) -> tuple[EmbeddingVector, ...]:
        del texts
        return (EmbeddingVector((1.0,)),)


class _FakeVectorWriter:
    def __init__(self) -> None:
        self.replacements: list[tuple[DocumentIdentity, tuple[VectorRecord, ...]]] = []
        self.deletions: list[DocumentKey] = []

    def replace_document(
        self,
        document: DocumentIdentity,
        records: tuple[VectorRecord, ...],
        /,
    ) -> None:
        if any(record.fragment.identity.document != document for record in records):
            raise AssertionError("records must belong to the supplied document")
        self.replacements.append((document, records))

    def delete_document(self, document: DocumentKey, /) -> None:
        self.deletions.append(document)


class _FakeVectorReader:
    def __init__(self, ranked: tuple[Fragment, ...]) -> None:
        self.ranked = ranked
        self.calls: list[tuple[RetrievalQuery, EmbeddingVector]] = []

    def search(
        self,
        query: RetrievalQuery,
        embedding: EmbeddingVector,
        /,
    ) -> tuple[Fragment, ...]:
        self.calls.append((query, embedding))
        matching = tuple(
            fragment
            for fragment in self.ranked
            if fragment.identity.document.key.corpus_id == query.corpus_id
        )
        return matching[: query.candidate_limit]


class _FakeLexicalRetriever:
    def __init__(self, ranked: tuple[Fragment, ...]) -> None:
        self.ranked = ranked
        self.calls: list[RetrievalQuery] = []

    def search(self, query: RetrievalQuery, /) -> tuple[Fragment, ...]:
        self.calls.append(query)
        matching = tuple(
            fragment
            for fragment in self.ranked
            if fragment.identity.document.key.corpus_id == query.corpus_id
        )
        return matching[: query.candidate_limit]


class _RaisingLexicalRetriever:
    def __init__(self, failure: RuntimeError) -> None:
        self.failure = failure

    def search(self, query: RetrievalQuery, /) -> tuple[Fragment, ...]:
        del query
        raise self.failure


class _LifecycleSentinel:
    def __init__(self) -> None:
        self.enter_calls = 0
        self.exit_calls = 0
        self.close_calls = 0
        self.shutdown_calls = 0

    def __enter__(self) -> _LifecycleSentinel:
        self.enter_calls += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.exit_calls += 1
        return True

    def close(self) -> None:
        self.close_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _MissingMethods:
    pass


class PortContractTests(unittest.TestCase):
    def test_exports_are_exact_and_owned_by_the_module(self) -> None:
        expected = (
            "Borrowed",
            "Embedder",
            "VectorIndexWriter",
            "VectorIndexReader",
            "LexicalRetriever",
        )

        self.assertEqual(ports.__all__, expected)
        for name in expected:
            exported = getattr(ports, name)
            self.assertEqual(exported.__module__, "generic_rag.ports")

    def test_protocols_are_runtime_checkable_structural_shapes(self) -> None:
        self.assertIsInstance(_FakeEmbedder(), Embedder)
        self.assertIsInstance(_FakeVectorWriter(), VectorIndexWriter)
        self.assertIsInstance(_FakeVectorReader(()), VectorIndexReader)
        self.assertIsInstance(_FakeLexicalRetriever(()), LexicalRetriever)

        missing = _MissingMethods()
        self.assertNotIsInstance(missing, Embedder)
        self.assertNotIsInstance(missing, VectorIndexWriter)
        self.assertNotIsInstance(missing, VectorIndexReader)
        self.assertNotIsInstance(missing, LexicalRetriever)

    def test_runtime_protocol_check_does_not_claim_semantic_enforcement(
        self,
    ) -> None:
        shape_only = _ShapeOnlyInvalidEmbedder()

        self.assertIsInstance(shape_only, Embedder)
        self.assertNotEqual(
            len(shape_only.embed(())),
            0,
            "runtime_checkable verifies names, not the empty-input obligation",
        )
        self.assertNotEqual(
            len(shape_only.embed(("text",))[0].values),
            shape_only.identity.dimensions,
            "runtime_checkable does not verify embedding dimensions",
        )

    def test_protocol_methods_are_synchronous_and_positional_only(self) -> None:
        methods = (
            (Embedder.embed, ("self", "texts")),
            (
                VectorIndexWriter.replace_document,
                ("self", "document", "records"),
            ),
            (VectorIndexWriter.delete_document, ("self", "document")),
            (
                VectorIndexReader.search,
                ("self", "query", "embedding"),
            ),
            (LexicalRetriever.search, ("self", "query")),
        )
        for method, names in methods:
            with self.subTest(method=method.__qualname__):
                self.assertFalse(inspect.iscoroutinefunction(method))
                parameters = tuple(inspect.signature(method).parameters.values())
                self.assertEqual(
                    tuple(parameter.name for parameter in parameters),
                    names,
                )
                self.assertTrue(
                    all(
                        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
                        for parameter in parameters
                    )
                )

    def test_protocol_annotations_keep_scores_and_lifecycle_out(self) -> None:
        self.assertEqual(
            get_type_hints(Embedder.embed),
            {
                "texts": tuple[str, ...],
                "return": tuple[EmbeddingVector, ...],
            },
        )
        self.assertEqual(
            get_type_hints(VectorIndexWriter.replace_document),
            {
                "document": DocumentIdentity,
                "records": tuple[VectorRecord, ...],
                "return": type(None),
            },
        )
        self.assertEqual(
            get_type_hints(VectorIndexWriter.delete_document),
            {
                "document": DocumentKey,
                "return": type(None),
            },
        )
        self.assertEqual(
            get_type_hints(VectorIndexReader.search),
            {
                "query": RetrievalQuery,
                "embedding": EmbeddingVector,
                "return": tuple[Fragment, ...],
            },
        )
        self.assertEqual(
            get_type_hints(LexicalRetriever.search),
            {
                "query": RetrievalQuery,
                "return": tuple[Fragment, ...],
            },
        )
        lifecycle_names = {"close", "shutdown", "__enter__", "__exit__"}
        for protocol in (
            Embedder,
            VectorIndexWriter,
            VectorIndexReader,
            LexicalRetriever,
        ):
            with self.subTest(protocol=protocol.__name__):
                self.assertTrue(lifecycle_names.isdisjoint(protocol.__dict__))

    def test_embedder_witness_preserves_order_count_dimensions_and_empty(self) -> None:
        embedder = _FakeEmbedder()

        self.assertEqual(embedder.embed(()), ())
        vectors = embedder.embed(("a", "longer"))

        self.assertEqual(embedder.calls, [(), ("a", "longer")])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(vectors[0].values, (0.0, 1.0))
        self.assertEqual(vectors[1].values, (1.0, 6.0))
        self.assertTrue(
            all(
                len(vector.values) == embedder.identity.dimensions for vector in vectors
            )
        )

    def test_writer_witness_replaces_complete_sets_and_accepts_empty(self) -> None:
        writer = _FakeVectorWriter()
        document = _document_identity()
        fragment = _fragment("fragment")
        records = (VectorRecord(fragment, EmbeddingVector((1.0, 2.0))),)

        writer.replace_document(document, records)
        writer.replace_document(document, ())
        writer.delete_document(document.key)

        self.assertEqual(
            writer.replacements,
            [(document, records), (document, ())],
        )
        self.assertEqual(writer.deletions, [document.key])

        wrong_document = _document_identity(document_id="other")
        with self.assertRaises(AssertionError):
            writer.replace_document(wrong_document, records)

    def test_reader_witnesses_preserve_rank_and_enforce_candidate_bound(self) -> None:
        ranked = (
            _fragment("first"),
            _fragment("wrong-corpus", corpus_id="other"),
            _fragment("second", document_id="second-document"),
            _fragment("third", document_id="third-document"),
        )
        query = _query(candidate_limit=2)
        embedding = EmbeddingVector((1.0, 2.0))
        vector_reader = _FakeVectorReader(ranked)
        lexical_reader = _FakeLexicalRetriever(ranked)

        self.assertEqual(
            vector_reader.search(query, embedding),
            (ranked[0], ranked[2]),
        )
        self.assertEqual(
            lexical_reader.search(query),
            (ranked[0], ranked[2]),
        )
        self.assertEqual(vector_reader.calls, [(query, embedding)])
        self.assertEqual(lexical_reader.calls, [query])

    def test_borrowed_never_invokes_or_owns_resource_lifecycle(self) -> None:
        resource = _LifecycleSentinel()
        borrowed = Borrowed(resource)

        self.assertEqual(
            tuple(field.name for field in fields(Borrowed)),
            ("resource",),
        )
        self.assertFalse(hasattr(borrowed, "__dict__"))
        self.assertIs(borrowed.resource, resource)
        with self.assertRaises(FrozenInstanceError):
            setattr(borrowed, "resource", object())

        with borrowed as entered:
            self.assertIs(entered, resource)

        self.assertEqual(
            (
                resource.enter_calls,
                resource.exit_calls,
                resource.close_calls,
                resource.shutdown_calls,
            ),
            (0, 0, 0, 0),
        )

    def test_borrowed_never_suppresses_collaborator_failure(self) -> None:
        failure = RuntimeError("sentinel collaborator failure")
        retriever = _RaisingLexicalRetriever(failure)

        with self.assertRaises(RuntimeError) as raised:
            with Borrowed(retriever) as borrowed:
                borrowed.search(_query())

        self.assertIs(raised.exception, failure)


if __name__ == "__main__":
    unittest.main()
