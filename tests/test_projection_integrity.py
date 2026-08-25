"""Independent v1 goldens and validation for projection integrity values."""

from __future__ import annotations

import hashlib
import inspect
import unittest
from collections.abc import Callable
from typing import cast

import generic_rag.projection_integrity as integrity
from generic_rag.contracts import (
    ChunkingPolicy,
    Document,
    DocumentIdentity,
    DocumentKey,
    EmbeddingIdentity,
    EmbeddingVector,
    ProjectionCheckpoint,
    ProjectionIdentity,
    ProjectionLimits,
    ProjectionManifest,
    ProjectionManifestEntry,
    ProjectionRequest,
    ProjectionStateAvailability,
    ProjectionStateSnapshot,
    VectorRecord,
)
from generic_rag.errors import ContractValidationError
from generic_rag.ports import Borrowed, Embedder, VectorIndexResetter, VectorIndexWriter
from generic_rag.projection import rebuild_projection
from generic_rag.projection_integrity import (
    derive_fragment_id,
    derive_projection_checkpoint_token,
    derive_source_digest,
    has_valid_projection_checkpoint,
)

_CORPUS_ID = " Corpus/../e\u0301 "
_SOURCE_GOLDEN = (
    "sha256:19390f5d911704efab45a06821296d2309454cfb8bdca63799d1c451afa1a9ea"
)
_FRAGMENT_GOLDEN = (
    "sha256:49d28767699855dbf6778396488946303a62dc8394c46adcdd22bef653950293"
)
_CHECKPOINT_GOLDEN = (
    "sha256:f62fca95057be15ab4e4bab8eebde94b7f92eb87773ae2d82d849664c80ba59f"
)


class _TupleSubclass(tuple[object, ...]):
    pass


def _golden_document() -> Document:
    return Document(
        DocumentIdentity(
            DocumentKey(_CORPUS_ID, "doc/\0😀"),
            "rev/한",
        ),
        "A\0😀e\u0301\n",
        (("k", "v"), ("k", "v"), ("", "한")),
    )


def _projection() -> ProjectionIdentity:
    return ProjectionIdentity(
        "schema/😀",
        EmbeddingIdentity("model/e\u0301", 3),
    )


def _checkpoint_entries() -> tuple[ProjectionManifestEntry, ...]:
    return (
        ProjectionManifestEntry(
            DocumentIdentity(DocumentKey(_CORPUS_ID, "alpha/e\u0301"), "r1😀"),
            _SOURCE_GOLDEN,
            2,
        ),
        ProjectionManifestEntry(
            DocumentIdentity(DocumentKey(_CORPUS_ID, "한"), "r2\0"),
            "sha256:" + "b" * 64,
            0,
        ),
    )


def _manifest(token: str = _CHECKPOINT_GOLDEN) -> ProjectionManifest:
    projection = _projection()
    return ProjectionManifest(
        _CORPUS_ID,
        projection,
        ChunkingPolicy(4, 1),
        _checkpoint_entries(),
        ProjectionCheckpoint(_CORPUS_ID, projection, token),
    )


def _independent_hash(fields: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for field in fields:
        encoded = field.encode("utf-8", "surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"


class _ProjectionEmbedder:
    identity = _projection().embedding

    def embed(self, texts: tuple[str, ...], /) -> tuple[EmbeddingVector, ...]:
        return tuple(EmbeddingVector((float(len(text)), 0.0, 1.0)) for text in texts)


class _ProjectionWriter:
    def __init__(self) -> None:
        self.records: tuple[VectorRecord, ...] = ()

    def replace_document(
        self,
        document: DocumentIdentity,
        records: tuple[VectorRecord, ...],
        /,
    ) -> None:
        del document
        self.records = records

    def delete_document(self, document: DocumentKey, /) -> None:
        del document


class _ProjectionResetter:
    def reset_corpus(self, corpus_id: str, /) -> None:
        del corpus_id


class ProjectionIntegrityPublicTests(unittest.TestCase):
    def test_exports_are_exact_owned_synchronous_and_positional_only(self) -> None:
        expected = (
            "derive_source_digest",
            "derive_fragment_id",
            "derive_projection_checkpoint_token",
            "has_valid_projection_checkpoint",
        )
        expected_parameters = {
            derive_source_digest: ("document",),
            derive_fragment_id: ("document", "start", "end"),
            derive_projection_checkpoint_token: (
                "corpus_id",
                "projection",
                "chunking",
                "entries",
            ),
            has_valid_projection_checkpoint: ("manifest",),
        }

        self.assertEqual(integrity.__all__, expected)
        for function, parameter_names in expected_parameters.items():
            with self.subTest(function=function.__name__):
                self.assertEqual(
                    function.__module__, "generic_rag.projection_integrity"
                )
                self.assertFalse(inspect.iscoroutinefunction(function))
                parameters = tuple(
                    inspect.signature(
                        cast(Callable[..., object], function)
                    ).parameters.values()
                )
                self.assertEqual(
                    tuple(parameter.name for parameter in parameters),
                    parameter_names,
                )
                self.assertTrue(
                    all(
                        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
                        for parameter in parameters
                    )
                )

    def test_fixed_unicode_nul_and_duplicate_attribute_v1_goldens(self) -> None:
        document = _golden_document()
        entries = _checkpoint_entries()

        source_fields = [
            "generic-rag:projection-source:v1",
            "text",
            document.text,
            "attributes_count",
            str(len(document.attributes)),
        ]
        for key, value in document.attributes:
            source_fields.extend(("attribute_key", key, "attribute_value", value))
        self.assertEqual(_independent_hash(tuple(source_fields)), _SOURCE_GOLDEN)
        self.assertEqual(derive_source_digest(document), _SOURCE_GOLDEN)

        fragment_fields = (
            "generic-rag:fragment-id:v1",
            "corpus_id",
            document.identity.key.corpus_id,
            "document_id",
            document.identity.key.document_id,
            "revision_id",
            document.identity.revision_id,
            "start",
            "1",
            "end",
            "5",
        )
        self.assertEqual(_independent_hash(fragment_fields), _FRAGMENT_GOLDEN)
        self.assertEqual(derive_fragment_id(document.identity, 1, 5), _FRAGMENT_GOLDEN)

        checkpoint_fields = [
            "generic-rag:projection-checkpoint:v1",
            "corpus_id",
            _CORPUS_ID,
            "schema_id",
            _projection().schema_id,
            "embedding_model_id",
            _projection().embedding.model_id,
            "embedding_dimensions",
            "3",
            "max_fragment_codepoints",
            "4",
            "overlap_codepoints",
            "1",
            "entry_count",
            "2",
        ]
        for entry in entries:
            checkpoint_fields.extend(
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
        self.assertEqual(
            _independent_hash(tuple(checkpoint_fields)), _CHECKPOINT_GOLDEN
        )
        self.assertEqual(
            derive_projection_checkpoint_token(
                _CORPUS_ID,
                _projection(),
                ChunkingPolicy(4, 1),
                entries,
            ),
            _CHECKPOINT_GOLDEN,
        )

    def test_checkpoint_validation_is_exact_and_does_not_mutate_manifest(self) -> None:
        manifest = _manifest()
        before = repr(manifest)

        self.assertTrue(has_valid_projection_checkpoint(manifest))
        self.assertFalse(has_valid_projection_checkpoint(_manifest("wrong-token")))
        self.assertEqual(repr(manifest), before)

    def test_projection_workflow_preserves_the_extracted_v1_algorithms(self) -> None:
        document = _golden_document()
        projection = _projection()
        chunking = ChunkingPolicy(4, 1)
        request = ProjectionRequest(
            _CORPUS_ID,
            projection,
            chunking,
            ProjectionLimits(1, len(document.text), 4),
            (document,),
        )
        writer = _ProjectionWriter()

        result = rebuild_projection(
            request,
            ProjectionStateSnapshot(ProjectionStateAvailability.MISSING, None),
            Borrowed(cast(Embedder, _ProjectionEmbedder())),
            Borrowed(cast(VectorIndexWriter, writer)),
            Borrowed(cast(VectorIndexResetter, _ProjectionResetter())),
        )

        entry = result.manifest.entries[0]
        self.assertEqual(entry.source_digest, derive_source_digest(document))
        self.assertEqual(
            tuple(record.fragment.identity.fragment_id for record in writer.records),
            tuple(
                derive_fragment_id(document.identity, start, end)
                for start, end in ((0, 4), (3, 6))
            ),
        )
        self.assertEqual(
            result.manifest.checkpoint.token,
            derive_projection_checkpoint_token(
                request.corpus_id,
                request.projection,
                request.chunking,
                result.manifest.entries,
            ),
        )


class ProjectionIntegrityValidationTests(unittest.TestCase):
    def assert_contract_failure(self, operation: Callable[[], object]) -> None:
        with self.assertRaises(ContractValidationError) as raised:
            operation()
        self.assertIs(type(raised.exception), ContractValidationError)

    def test_top_level_values_require_exact_public_types(self) -> None:
        operations: tuple[Callable[[], object], ...] = (
            lambda: derive_source_digest(cast(Document, object())),
            lambda: derive_fragment_id(
                cast(DocumentIdentity, object()),
                0,
                1,
            ),
            lambda: derive_projection_checkpoint_token(
                cast(str, object()),
                _projection(),
                ChunkingPolicy(4, 1),
                (),
            ),
            lambda: derive_projection_checkpoint_token(
                _CORPUS_ID,
                cast(ProjectionIdentity, object()),
                ChunkingPolicy(4, 1),
                (),
            ),
            lambda: derive_projection_checkpoint_token(
                _CORPUS_ID,
                _projection(),
                cast(ChunkingPolicy, object()),
                (),
            ),
            lambda: derive_projection_checkpoint_token(
                _CORPUS_ID,
                _projection(),
                ChunkingPolicy(4, 1),
                cast(tuple[ProjectionManifestEntry, ...], []),
            ),
            lambda: has_valid_projection_checkpoint(cast(ProjectionManifest, object())),
        )

        for operation in operations:
            with self.subTest(operation=operation):
                self.assert_contract_failure(operation)

    def test_corrupted_frozen_nested_values_fail_as_contract_errors(self) -> None:
        malformed_documents = (_golden_document(), _golden_document())
        object.__setattr__(malformed_documents[0], "identity", object())
        object.__setattr__(malformed_documents[1], "attributes", [("k", "v")])

        malformed_identity = _golden_document().identity
        object.__setattr__(malformed_identity, "key", object())

        malformed_projection = _projection()
        object.__setattr__(malformed_projection.embedding, "dimensions", True)

        malformed_chunking = ChunkingPolicy(4, 1)
        object.__setattr__(malformed_chunking, "overlap_codepoints", 4)

        malformed_entry = _checkpoint_entries()[0]
        object.__setattr__(malformed_entry, "fragment_count", -1)

        operations: tuple[Callable[[], object], ...] = (
            *(
                lambda value=value: derive_source_digest(value)
                for value in malformed_documents
            ),
            lambda: derive_fragment_id(malformed_identity, 0, 1),
            lambda: derive_projection_checkpoint_token(
                _CORPUS_ID,
                malformed_projection,
                ChunkingPolicy(4, 1),
                (),
            ),
            lambda: derive_projection_checkpoint_token(
                _CORPUS_ID,
                _projection(),
                malformed_chunking,
                (),
            ),
            lambda: derive_projection_checkpoint_token(
                _CORPUS_ID,
                _projection(),
                ChunkingPolicy(4, 1),
                (malformed_entry,),
            ),
        )

        for operation in operations:
            with self.subTest(operation=operation):
                self.assert_contract_failure(operation)

    def test_fragment_ranges_are_nonnegative_ordered_exact_integers(self) -> None:
        document = _golden_document().identity
        for start, end in (
            (-1, 1),
            (0, 0),
            (2, 1),
            (True, 1),
            (0, False),
            (0.0, 1),
            (0, 1.0),
        ):
            with self.subTest(start=start, end=end):
                self.assert_contract_failure(
                    lambda: derive_fragment_id(
                        document,
                        cast(int, start),
                        cast(int, end),
                    )
                )

    def test_checkpoint_entries_must_be_an_exact_canonical_tuple(self) -> None:
        entries = _checkpoint_entries()
        wrong_corpus = ProjectionManifestEntry(
            DocumentIdentity(DocumentKey("other", "document"), "revision"),
            _SOURCE_GOLDEN,
            1,
        )
        duplicate = ProjectionManifestEntry(
            entries[0].document,
            "sha256:" + "c" * 64,
            1,
        )
        malformed_cases: tuple[object, ...] = (
            list(entries),
            _TupleSubclass(entries),
            tuple(reversed(entries)),
            (entries[0], duplicate),
            (wrong_corpus,),
            (object(),),
        )

        for malformed in malformed_cases:
            with self.subTest(container=type(malformed).__name__, value=malformed):
                self.assert_contract_failure(
                    lambda: derive_projection_checkpoint_token(
                        _CORPUS_ID,
                        _projection(),
                        ChunkingPolicy(4, 1),
                        cast(tuple[ProjectionManifestEntry, ...], malformed),
                    )
                )

    def test_checkpoint_validation_rejects_corrupted_manifest_shapes(self) -> None:
        malformed_manifests = (_manifest(), _manifest(), _manifest())
        object.__setattr__(
            malformed_manifests[0], "entries", list(_checkpoint_entries())
        )
        object.__setattr__(malformed_manifests[1], "checkpoint", object())
        object.__setattr__(malformed_manifests[2].projection, "schema_id", 1)

        for malformed in malformed_manifests:
            with self.subTest(manifest=malformed):
                self.assert_contract_failure(
                    lambda: has_valid_projection_checkpoint(malformed)
                )


if __name__ == "__main__":
    unittest.main()
