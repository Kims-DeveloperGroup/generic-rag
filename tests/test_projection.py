"""Deterministic projection orchestration, failure, and lifecycle tests."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
import textwrap
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import cast

import generic_rag.projection as projection_module
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
    ProjectionOutcome,
    ProjectionReceipt,
    ProjectionRequest,
    ProjectionStateAvailability,
    ProjectionStateSnapshot,
    ProjectionStateStatus,
    VectorRecord,
)
from generic_rag.errors import (
    CollaborationError,
    ContractValidationError,
    StateCompatibilityError,
)
from generic_rag.ports import (
    Borrowed,
    Embedder,
    VectorIndexResetter,
    VectorIndexWriter,
)
from generic_rag.projection import (
    ProjectionFailureStage,
    ProjectionOperationError,
    ProjectionStateError,
    project_documents,
    rebuild_projection,
)

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


def _document(
    document_id: str,
    *,
    revision_id: str = "revision-1",
    text: str = "abcdefgh",
    corpus_id: str = "corpus",
    attributes: tuple[tuple[str, str], ...] = (),
) -> Document:
    return Document(
        DocumentIdentity(DocumentKey(corpus_id, document_id), revision_id),
        text,
        attributes,
    )


def _identity(
    *,
    schema_id: str = "schema-v1",
    model_id: str = "model-v1",
    dimensions: int = 2,
) -> ProjectionIdentity:
    return ProjectionIdentity(
        schema_id,
        EmbeddingIdentity(model_id, dimensions),
    )


def _request(
    documents: tuple[Document, ...],
    *,
    projection: ProjectionIdentity | None = None,
    chunking: ChunkingPolicy | None = None,
    batch_size: int = 2,
    corpus_id: str = "corpus",
) -> ProjectionRequest:
    return ProjectionRequest(
        corpus_id,
        projection or _identity(),
        chunking or ChunkingPolicy(4, 1),
        ProjectionLimits(max(1, len(documents)), 100, batch_size),
        documents,
    )


def _hash_fields(fields_to_hash: tuple[str, ...]) -> str:
    """Independent length-delimited SHA-256 oracle from the public contract."""

    digest = hashlib.sha256()
    for value in fields_to_hash:
        encoded = value.encode("utf-8", "surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"


def _source_digest(document: Document) -> str:
    source_fields = [
        "generic-rag:projection-source:v1",
        "text",
        document.text,
        "attributes_count",
        str(len(document.attributes)),
    ]
    for key, value in document.attributes:
        source_fields.extend(("attribute_key", key, "attribute_value", value))
    return _hash_fields(tuple(source_fields))


def _fragment_id(document: DocumentIdentity, start: int, end: int) -> str:
    return _hash_fields(
        (
            "generic-rag:fragment-id:v1",
            "corpus_id",
            document.key.corpus_id,
            "document_id",
            document.key.document_id,
            "revision_id",
            document.revision_id,
            "start",
            str(start),
            "end",
            str(end),
        )
    )


def _expected_ranges(
    text: str, chunking: ChunkingPolicy
) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + chunking.max_fragment_codepoints, len(text))
        ranges.append((start, end))
        if end == len(text):
            break
        start = end - chunking.overlap_codepoints
    return tuple(ranges)


def _checkpoint_token(
    request: ProjectionRequest,
    entries: tuple[ProjectionManifestEntry, ...],
) -> str:
    token_fields = [
        "generic-rag:projection-checkpoint:v1",
        "corpus_id",
        request.corpus_id,
        "schema_id",
        request.projection.schema_id,
        "embedding_model_id",
        request.projection.embedding.model_id,
        "embedding_dimensions",
        str(request.projection.embedding.dimensions),
        "max_fragment_codepoints",
        str(request.chunking.max_fragment_codepoints),
        "overlap_codepoints",
        str(request.chunking.overlap_codepoints),
        "entry_count",
        str(len(entries)),
    ]
    for entry in entries:
        token_fields.extend(
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
    return _hash_fields(tuple(token_fields))


def _manifest(request: ProjectionRequest) -> ProjectionManifest:
    entries = tuple(
        ProjectionManifestEntry(
            document.identity,
            _source_digest(document),
            len(_expected_ranges(document.text, request.chunking)),
        )
        for document in request.documents
    )
    checkpoint = ProjectionCheckpoint(
        request.corpus_id,
        request.projection,
        _checkpoint_token(request, entries),
    )
    return ProjectionManifest(
        request.corpus_id,
        request.projection,
        request.chunking,
        entries,
        checkpoint,
    )


def _present(request: ProjectionRequest) -> ProjectionStateSnapshot:
    return ProjectionStateSnapshot(
        ProjectionStateAvailability.PRESENT,
        _manifest(request),
    )


class _FakeEmbedder:
    def __init__(
        self,
        expected_identity: EmbeddingIdentity,
        events: list[str],
        *,
        identity_value: object | None = None,
        identity_failure: BaseException | None = None,
        output: Callable[[tuple[str, ...]], object] | None = None,
        embed_failure_at: int | None = None,
        embed_failure: BaseException | None = None,
    ) -> None:
        self.expected_identity = expected_identity
        self.events = events
        self.identity_value = (
            expected_identity if identity_value is None else identity_value
        )
        self.identity_failure = identity_failure
        self.output = output
        self.embed_failure_at = embed_failure_at
        self.embed_failure = embed_failure or RuntimeError("embedding failed")
        self.identity_calls = 0
        self.embed_calls: list[tuple[str, ...]] = []
        self.lifecycle_calls: list[str] = []

    @property
    def identity(self) -> EmbeddingIdentity:
        self.identity_calls += 1
        self.events.append("identity")
        if self.identity_failure is not None:
            raise self.identity_failure
        return cast(EmbeddingIdentity, self.identity_value)

    def embed(self, texts: tuple[str, ...], /) -> tuple[EmbeddingVector, ...]:
        call_index = len(self.embed_calls)
        self.embed_calls.append(texts)
        self.events.append("embed:" + "|".join(texts))
        if self.embed_failure_at == call_index:
            raise self.embed_failure
        if self.output is not None:
            return cast(tuple[EmbeddingVector, ...], self.output(texts))
        return tuple(
            EmbeddingVector((float(len(text)), float(index)))
            for index, text in enumerate(texts)
        )

    def __enter__(self) -> _FakeEmbedder:
        self.lifecycle_calls.append("enter")
        return self

    def __exit__(self, *arguments: object) -> bool:
        del arguments
        self.lifecycle_calls.append("exit")
        return True

    def close(self) -> None:
        self.lifecycle_calls.append("close")

    def shutdown(self) -> None:
        self.lifecycle_calls.append("shutdown")


class _FakeWriter:
    def __init__(
        self,
        events: list[str],
        *,
        replace_failure_at: int | None = None,
        replace_failure: BaseException | None = None,
        replace_result: object = None,
        delete_failure_at: int | None = None,
        delete_failure: BaseException | None = None,
        delete_result: object = None,
    ) -> None:
        self.events = events
        self.replace_failure_at = replace_failure_at
        self.replace_failure = replace_failure or RuntimeError("replacement failed")
        self.replace_result = replace_result
        self.delete_failure_at = delete_failure_at
        self.delete_failure = delete_failure or RuntimeError("deletion failed")
        self.delete_result = delete_result
        self.replacements: list[tuple[DocumentIdentity, tuple[VectorRecord, ...]]] = []
        self.deletions: list[DocumentKey] = []
        self.lifecycle_calls: list[str] = []

    def replace_document(
        self,
        document: DocumentIdentity,
        records: tuple[VectorRecord, ...],
        /,
    ) -> None:
        call_index = len(self.replacements)
        self.replacements.append((document, records))
        self.events.append(f"replace:{document.key.document_id}")
        if self.replace_failure_at == call_index:
            raise self.replace_failure
        return cast(None, self.replace_result)

    def delete_document(self, document: DocumentKey, /) -> None:
        call_index = len(self.deletions)
        self.deletions.append(document)
        self.events.append(f"delete:{document.document_id}")
        if self.delete_failure_at == call_index:
            raise self.delete_failure
        return cast(None, self.delete_result)

    def __enter__(self) -> _FakeWriter:
        self.lifecycle_calls.append("enter")
        return self

    def __exit__(self, *arguments: object) -> bool:
        del arguments
        self.lifecycle_calls.append("exit")
        return True

    def close(self) -> None:
        self.lifecycle_calls.append("close")

    def shutdown(self) -> None:
        self.lifecycle_calls.append("shutdown")


class _FakeResetter:
    def __init__(
        self,
        events: list[str],
        *,
        failure: BaseException | None = None,
        result: object = None,
    ) -> None:
        self.events = events
        self.failure = failure
        self.result = result
        self.corpora: list[str] = []
        self.lifecycle_calls: list[str] = []

    def reset_corpus(self, corpus_id: str, /) -> None:
        self.corpora.append(corpus_id)
        self.events.append(f"reset:{corpus_id}")
        if self.failure is not None:
            raise self.failure
        return cast(None, self.result)

    def __enter__(self) -> _FakeResetter:
        self.lifecycle_calls.append("enter")
        return self

    def __exit__(self, *arguments: object) -> bool:
        del arguments
        self.lifecycle_calls.append("exit")
        return True

    def close(self) -> None:
        self.lifecycle_calls.append("close")

    def shutdown(self) -> None:
        self.lifecycle_calls.append("shutdown")


def _borrow_embedder(embedder: _FakeEmbedder) -> Borrowed[Embedder]:
    return Borrowed(cast(Embedder, embedder))


def _borrow_writer(writer: _FakeWriter) -> Borrowed[VectorIndexWriter]:
    return Borrowed(cast(VectorIndexWriter, writer))


def _borrow_resetter(resetter: _FakeResetter) -> Borrowed[VectorIndexResetter]:
    return Borrowed(cast(VectorIndexResetter, resetter))


def _constant_output(value: object) -> Callable[[tuple[str, ...]], object]:
    def output(texts: tuple[str, ...]) -> object:
        del texts
        return value

    return output


def _collaborators(
    request: ProjectionRequest,
) -> tuple[_FakeEmbedder, _FakeWriter, _FakeResetter, list[str]]:
    events: list[str] = []
    return (
        _FakeEmbedder(request.projection.embedding, events),
        _FakeWriter(events),
        _FakeResetter(events),
        events,
    )


class ProjectionPublicApiTests(unittest.TestCase):
    def test_exports_are_exact_and_owned_by_projection_module(self) -> None:
        expected = (
            "ProjectionFailureStage",
            "ProjectionStateError",
            "ProjectionOperationError",
            "project_documents",
            "rebuild_projection",
        )

        self.assertEqual(projection_module.__all__, expected)
        for name in expected:
            self.assertEqual(
                getattr(projection_module, name).__module__,
                "generic_rag.projection",
            )

    def test_public_workflows_are_synchronous_and_positional_only(self) -> None:
        expected_parameters = {
            project_documents: ("request", "state", "embedder", "writer"),
            rebuild_projection: (
                "request",
                "state",
                "embedder",
                "writer",
                "resetter",
            ),
        }
        for function, expected in expected_parameters.items():
            with self.subTest(function=function.__name__):
                self.assertFalse(inspect.iscoroutinefunction(function))
                parameters = tuple(
                    inspect.signature(
                        cast(Callable[..., object], function)
                    ).parameters.values()
                )
                self.assertEqual(tuple(item.name for item in parameters), expected)
                self.assertTrue(
                    all(
                        item.kind is inspect.Parameter.POSITIONAL_ONLY
                        for item in parameters
                    )
                )

    def test_failure_stage_is_an_exact_closed_string_enum(self) -> None:
        self.assertEqual(
            tuple((member.name, member.value) for member in ProjectionFailureStage),
            (
                ("EMBEDDER_IDENTITY", "embedder_identity"),
                ("EMBEDDING", "embedding"),
                ("REPLACEMENT", "replacement"),
                ("DELETION", "deletion"),
                ("RESET", "reset"),
            ),
        )
        for value in ("EMBEDDING", "unknown", "", None, 1, object()):
            with self.subTest(value=value):
                with self.assertRaises(ContractValidationError):
                    ProjectionFailureStage(cast(str, value))

    def test_state_error_is_typed_content_free_and_requires_exact_status(self) -> None:
        error = ProjectionStateError(ProjectionStateStatus.SCHEMA_MISMATCH)

        self.assertIsInstance(error, StateCompatibilityError)
        self.assertIs(error.status, ProjectionStateStatus.SCHEMA_MISMATCH)
        self.assertNotIn("schema", str(error).lower())
        with self.assertRaises(ContractValidationError):
            ProjectionStateError(cast(ProjectionStateStatus, "schema_mismatch"))

    def test_operation_error_fields_and_receipt_are_exact_and_truthful(self) -> None:
        request = _request((_document("alpha"),))
        receipt = ProjectionReceipt(
            request.corpus_id,
            request.projection,
            ProjectionOutcome.FAILED,
            1,
            0,
            None,
        )
        key = request.documents[0].identity.key
        error = ProjectionOperationError(
            ProjectionFailureStage.REPLACEMENT,
            key,
            receipt,
        )

        self.assertIsInstance(error, CollaborationError)
        self.assertEqual(
            ProjectionOperationError.__annotations__,
            {
                "stage": "ProjectionFailureStage",
                "affected_document": "DocumentKey | None",
                "receipt": "ProjectionReceipt | None",
            },
        )
        self.assertIs(error.stage, ProjectionFailureStage.REPLACEMENT)
        self.assertEqual(error.affected_document, key)
        self.assertIs(error.receipt, receipt)
        for stage, affected, value_receipt in (
            ("replacement", key, receipt),
            (ProjectionFailureStage.REPLACEMENT, object(), receipt),
            (
                ProjectionFailureStage.REPLACEMENT,
                key,
                ProjectionReceipt(
                    request.corpus_id,
                    request.projection,
                    ProjectionOutcome.COMPLETED,
                    1,
                    1,
                    _manifest(request).checkpoint,
                ),
            ),
            (ProjectionFailureStage.REPLACEMENT, key, object()),
        ):
            with self.subTest(stage=stage, affected=affected):
                with self.assertRaises(ContractValidationError):
                    ProjectionOperationError(
                        cast(ProjectionFailureStage, stage),
                        cast(DocumentKey, affected),
                        cast(ProjectionReceipt, value_receipt),
                    )

    def test_invalid_top_level_inputs_are_rejected_before_any_effect(self) -> None:
        request = _request((_document("alpha"),))
        state = _present(_request((), projection=request.projection))
        embedder, writer, resetter, events = _collaborators(request)
        invalid_calls: tuple[Callable[[], object], ...] = (
            lambda: project_documents(
                cast(ProjectionRequest, object()),
                state,
                _borrow_embedder(embedder),
                _borrow_writer(writer),
            ),
            lambda: project_documents(
                request,
                cast(ProjectionStateSnapshot, object()),
                _borrow_embedder(embedder),
                _borrow_writer(writer),
            ),
            lambda: project_documents(
                request,
                state,
                cast(Borrowed[Embedder], object()),
                _borrow_writer(writer),
            ),
            lambda: project_documents(
                request,
                state,
                _borrow_embedder(embedder),
                cast(Borrowed[VectorIndexWriter], object()),
            ),
            lambda: rebuild_projection(
                request,
                state,
                _borrow_embedder(embedder),
                _borrow_writer(writer),
                cast(Borrowed[VectorIndexResetter], object()),
            ),
        )

        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ContractValidationError):
                    call()
                self.assertEqual(events, [])


class ProjectionDeterminismTests(unittest.TestCase):
    def test_unicode_codepoint_chunks_overlap_and_preserve_source_attributes(
        self,
    ) -> None:
        document = _document(
            "unicode",
            text="A😀e\u0301한Z",
            attributes=(("", ""), ("tag", "one"), ("tag", "one")),
        )
        request = _request(
            (document,),
            chunking=ChunkingPolicy(3, 1),
            batch_size=8,
        )
        previous = _request(
            (), projection=request.projection, chunking=request.chunking
        )
        embedder, writer, _, _ = _collaborators(request)

        result = project_documents(
            request,
            _present(previous),
            _borrow_embedder(embedder),
            _borrow_writer(writer),
        )

        records = writer.replacements[0][1]
        self.assertEqual(
            tuple(
                (record.fragment.identity.start, record.fragment.identity.end)
                for record in records
            ),
            ((0, 3), (2, 5), (4, 6)),
        )
        self.assertEqual(
            tuple(record.fragment.text for record in records),
            ("A😀e", "e\u0301한", "한Z"),
        )
        self.assertTrue(
            all(record.fragment.attributes == document.attributes for record in records)
        )
        self.assertTrue(
            all(
                record.fragment.identity.document == document.identity
                for record in records
            )
        )
        self.assertEqual(result.manifest.entries[0].fragment_count, 3)

    def test_fragment_ids_source_digest_and_checkpoint_match_independent_oracle(
        self,
    ) -> None:
        document = _document(
            "doc/../A",
            revision_id=" rev e\u0301 ",
            text="abcdef",
            attributes=(("k", "v"), ("k", "v"), ("", "")),
        )
        request = _request((document,), chunking=ChunkingPolicy(4, 1))
        embedder, writer, resetter, _ = _collaborators(request)

        result = rebuild_projection(
            request,
            ProjectionStateSnapshot(ProjectionStateAvailability.MISSING, None),
            _borrow_embedder(embedder),
            _borrow_writer(writer),
            _borrow_resetter(resetter),
        )

        expected_manifest = _manifest(request)
        self.assertEqual(result.manifest, expected_manifest)
        self.assertEqual(
            tuple(
                record.fragment.identity.fragment_id
                for record in writer.replacements[0][1]
            ),
            tuple(
                _fragment_id(document.identity, start, end)
                for start, end in _expected_ranges(document.text, request.chunking)
            ),
        )
        self.assertEqual(
            result.manifest.entries[0].source_digest,
            _source_digest(document),
        )
        self.assertEqual(
            result.manifest.checkpoint.token,
            _checkpoint_token(request, expected_manifest.entries),
        )

    def test_source_digest_distinguishes_attribute_order_duplicates_and_text(
        self,
    ) -> None:
        base = _document("doc", text="same", attributes=(("a", "1"), ("b", "2")))
        variants = (
            _document("doc", text="same", attributes=(("b", "2"), ("a", "1"))),
            _document("doc", text="same", attributes=(("a", "1"), ("a", "1"))),
            _document("doc", text="same!", attributes=(("a", "1"), ("b", "2"))),
        )

        self.assertEqual(len({_source_digest(base), *map(_source_digest, variants)}), 4)

    def test_embedding_batches_are_bounded_ordered_and_never_empty(self) -> None:
        document = _document("doc", text="abcdefghijklmn")
        request = _request(
            (document,),
            chunking=ChunkingPolicy(3, 0),
            batch_size=2,
        )
        previous = _request(
            (), projection=request.projection, chunking=request.chunking
        )
        embedder, writer, _, _ = _collaborators(request)

        project_documents(
            request,
            _present(previous),
            _borrow_embedder(embedder),
            _borrow_writer(writer),
        )

        self.assertEqual(
            embedder.embed_calls,
            [("abc", "def"), ("ghi", "jkl"), ("mn",)],
        )
        self.assertTrue(all(batch for batch in embedder.embed_calls))
        self.assertEqual(
            tuple(record.fragment.text for record in writer.replacements[0][1]),
            ("abc", "def", "ghi", "jkl", "mn"),
        )

    def test_empty_document_is_replaced_by_an_explicit_empty_record_tuple(self) -> None:
        document = _document("doc", revision_id="revision-2", text="")
        request = _request((document,))
        previous = _request(
            (_document("doc", revision_id="revision-1", text="old"),),
            projection=request.projection,
            chunking=request.chunking,
        )
        embedder, writer, _, _ = _collaborators(request)

        result = project_documents(
            request,
            _present(previous),
            _borrow_embedder(embedder),
            _borrow_writer(writer),
        )

        self.assertEqual(embedder.embed_calls, [])
        self.assertEqual(writer.replacements, [(document.identity, ())])
        self.assertEqual(result.manifest.entries[0].fragment_count, 0)

    def test_document_and_mutation_order_is_canonical_by_opaque_document_id(
        self,
    ) -> None:
        request = _request(
            (
                _document("c", revision_id="new"),
                _document("a", revision_id="new"),
            )
        )
        previous = _request(
            (
                _document("d", revision_id="old"),
                _document("b", revision_id="old"),
            ),
            projection=request.projection,
            chunking=request.chunking,
        )
        embedder, writer, _, events = _collaborators(request)

        project_documents(
            request,
            _present(previous),
            _borrow_embedder(embedder),
            _borrow_writer(writer),
        )

        mutation_events = [
            event for event in events if event.startswith(("replace:", "delete:"))
        ]
        self.assertEqual(
            mutation_events,
            ["replace:a", "delete:b", "replace:c", "delete:d"],
        )

    def test_fixed_projection_is_identical_in_clean_processes(self) -> None:
        script = textwrap.dedent(
            f"""
            import json
            import sys
            sys.path.insert(0, {os.fspath(_SOURCE_ROOT)!r})
            from generic_rag.contracts import *
            from generic_rag.ports import Borrowed
            from generic_rag.projection import rebuild_projection

            class Embedder:
                identity = EmbeddingIdentity('model-v1', 2)
                def embed(self, texts, /):
                    return tuple(
                        EmbeddingVector((len(text), index))
                        for index, text in enumerate(texts)
                    )
            class Writer:
                def __init__(self): self.records = ()
                def replace_document(self, document, records, /): self.records = records
                def delete_document(self, document, /): return None
            class Resetter:
                def reset_corpus(self, corpus_id, /): return None

            document = Document(
                DocumentIdentity(DocumentKey('corpus', 'doc/../A'), ' rev e\\u0301 '),
                'abcdef',
                (('k', 'v'), ('k', 'v'), ('', '')),
            )
            request = ProjectionRequest(
                'corpus',
                ProjectionIdentity('schema-v1', EmbeddingIdentity('model-v1', 2)),
                ChunkingPolicy(4, 1),
                ProjectionLimits(1, 100, 2),
                (document,),
            )
            writer = Writer()
            result = rebuild_projection(
                request,
                ProjectionStateSnapshot(ProjectionStateAvailability.MISSING, None),
                Borrowed(Embedder()),
                Borrowed(writer),
                Borrowed(Resetter()),
            )
            print(json.dumps({{
                'source': result.manifest.entries[0].source_digest,
                'checkpoint': result.manifest.checkpoint.token,
                'fragments': [
                    record.fragment.identity.fragment_id
                    for record in writer.records
                ],
            }}, sort_keys=True))
            """
        )
        outputs: list[dict[str, object]] = []
        for _ in range(2):
            completed = subprocess.run(
                (sys.executable, "-I", "-B", "-c", script),
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            outputs.append(cast(dict[str, object], json.loads(completed.stdout)))

        request = _request(
            (
                _document(
                    "doc/../A",
                    revision_id=" rev e\u0301 ",
                    text="abcdef",
                    attributes=(("k", "v"), ("k", "v"), ("", "")),
                ),
            ),
            chunking=ChunkingPolicy(4, 1),
        )
        expected = _manifest(request)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0]["source"], expected.entries[0].source_digest)
        self.assertEqual(outputs[0]["checkpoint"], expected.checkpoint.token)
        self.assertEqual(
            outputs[0]["fragments"],
            [
                _fragment_id(request.documents[0].identity, start, end)
                for start, end in _expected_ranges(
                    request.documents[0].text,
                    request.chunking,
                )
            ],
        )


class IncrementalProjectionTests(unittest.TestCase):
    def test_current_state_is_unchanged_without_touching_collaborators(self) -> None:
        request = _request((_document("alpha"), _document("beta")))
        embedder, writer, _, events = _collaborators(request)
        embedder.identity_failure = AssertionError("identity must not be read")
        writer.replace_failure_at = 0

        result = project_documents(
            request,
            _present(request),
            _borrow_embedder(embedder),
            _borrow_writer(writer),
        )

        self.assertIs(result.status_before, ProjectionStateStatus.CURRENT)
        self.assertIs(result.receipt.outcome, ProjectionOutcome.UNCHANGED)
        self.assertEqual(result.receipt.attempted_documents, 0)
        self.assertEqual(result.receipt.completed_documents, 0)
        self.assertEqual(result.manifest, _manifest(request))
        self.assertEqual(events, [])

    def test_incompatible_state_statuses_fail_before_collaborator_effects(self) -> None:
        request = _request((_document("alpha"),))
        schema_request = _request(
            request.documents,
            projection=_identity(schema_id="other-schema"),
        )
        embedding_request = _request(
            request.documents,
            projection=_identity(model_id="other-model"),
        )
        corpus_request = _request(
            (_document("alpha", corpus_id="other"),),
            projection=request.projection,
            corpus_id="other",
        )
        cases = (
            (
                ProjectionStateSnapshot(ProjectionStateAvailability.MISSING, None),
                ProjectionStateStatus.MISSING,
            ),
            (
                ProjectionStateSnapshot(ProjectionStateAvailability.CORRUPT, None),
                ProjectionStateStatus.CORRUPT,
            ),
            (_present(schema_request), ProjectionStateStatus.SCHEMA_MISMATCH),
            (_present(embedding_request), ProjectionStateStatus.EMBEDDING_MISMATCH),
            (_present(corpus_request), ProjectionStateStatus.CORRUPT),
        )

        for state, expected_status in cases:
            embedder, writer, _, events = _collaborators(request)
            with self.subTest(status=expected_status):
                with self.assertRaises(ProjectionStateError) as raised:
                    project_documents(
                        request,
                        state,
                        _borrow_embedder(embedder),
                        _borrow_writer(writer),
                    )
                self.assertIs(raised.exception.status, expected_status)
                self.assertEqual(events, [])

    def test_invalid_checkpoint_is_corrupt_before_any_effect(self) -> None:
        request = _request((_document("alpha"),))
        valid = _manifest(request)
        corrupt = ProjectionManifest(
            valid.corpus_id,
            valid.projection,
            valid.chunking,
            valid.entries,
            ProjectionCheckpoint(valid.corpus_id, valid.projection, "wrong-token"),
        )
        embedder, writer, _, events = _collaborators(request)

        with self.assertRaises(ProjectionStateError) as raised:
            project_documents(
                request,
                ProjectionStateSnapshot(
                    ProjectionStateAvailability.PRESENT,
                    corrupt,
                ),
                _borrow_embedder(embedder),
                _borrow_writer(writer),
            )

        self.assertIs(raised.exception.status, ProjectionStateStatus.CORRUPT)
        self.assertEqual(events, [])

    def test_same_revision_digest_or_fragment_count_drift_is_corrupt(self) -> None:
        request = _request((_document("alpha"),))
        valid = _manifest(request)
        original = valid.entries[0]
        changed_entries = (
            (
                ProjectionManifestEntry(
                    original.document,
                    "sha256:" + "f" * 64,
                    original.fragment_count,
                ),
            ),
            (
                ProjectionManifestEntry(
                    original.document,
                    original.source_digest,
                    original.fragment_count + 1,
                ),
            ),
        )

        for entries in changed_entries:
            previous_request = _request(
                request.documents,
                projection=request.projection,
                chunking=request.chunking,
            )
            manifest = ProjectionManifest(
                request.corpus_id,
                request.projection,
                request.chunking,
                entries,
                ProjectionCheckpoint(
                    request.corpus_id,
                    request.projection,
                    _checkpoint_token(previous_request, entries),
                ),
            )
            embedder, writer, _, events = _collaborators(request)
            with self.subTest(entries=entries):
                with self.assertRaises(ProjectionStateError) as raised:
                    project_documents(
                        request,
                        ProjectionStateSnapshot(
                            ProjectionStateAvailability.PRESENT,
                            manifest,
                        ),
                        _borrow_embedder(embedder),
                        _borrow_writer(writer),
                    )
                self.assertIs(
                    raised.exception.status,
                    ProjectionStateStatus.CORRUPT,
                )
                self.assertEqual(events, [])

    def test_revision_and_chunking_changes_are_stale_and_replaced(self) -> None:
        target = _request(
            (_document("alpha", revision_id="revision-2", text="new text"),),
            chunking=ChunkingPolicy(4, 1),
        )
        previous_requests = (
            _request(
                (_document("alpha", revision_id="revision-1", text="new text"),),
                projection=target.projection,
                chunking=target.chunking,
            ),
            _request(
                target.documents,
                projection=target.projection,
                chunking=ChunkingPolicy(5, 0),
            ),
        )

        for previous in previous_requests:
            embedder, writer, _, _ = _collaborators(target)
            with self.subTest(previous=previous):
                result = project_documents(
                    target,
                    _present(previous),
                    _borrow_embedder(embedder),
                    _borrow_writer(writer),
                )
                self.assertIs(result.status_before, ProjectionStateStatus.STALE)
                self.assertEqual(
                    [identity for identity, _ in writer.replacements],
                    [target.documents[0].identity],
                )

    def test_same_revision_source_change_is_corrupt_not_stale(self) -> None:
        target = _request(
            (_document("alpha", revision_id="revision-2", text="new text"),)
        )
        previous = _request(
            (_document("alpha", revision_id="revision-2", text="old text"),),
            projection=target.projection,
            chunking=target.chunking,
        )
        embedder, writer, _, events = _collaborators(target)

        with self.assertRaises(ProjectionStateError) as raised:
            project_documents(
                target,
                _present(previous),
                _borrow_embedder(embedder),
                _borrow_writer(writer),
            )

        self.assertIs(raised.exception.status, ProjectionStateStatus.CORRUPT)
        self.assertEqual(events, [])

    def test_delete_only_plan_does_not_access_embedder(self) -> None:
        target = _request(())
        previous = _request(
            (_document("zeta"), _document("alpha")),
            projection=target.projection,
            chunking=target.chunking,
        )
        embedder, writer, _, events = _collaborators(target)
        embedder.identity_failure = AssertionError("delete-only must not embed")

        result = project_documents(
            target,
            _present(previous),
            _borrow_embedder(embedder),
            _borrow_writer(writer),
        )

        self.assertEqual(
            writer.deletions,
            [DocumentKey("corpus", "alpha"), DocumentKey("corpus", "zeta")],
        )
        self.assertEqual(embedder.identity_calls, 0)
        self.assertFalse(any(event.startswith("embed:") for event in events))
        self.assertEqual(result.receipt.attempted_documents, 2)
        self.assertEqual(result.receipt.completed_documents, 2)

    def test_success_receipt_counts_only_changed_and_removed_documents(self) -> None:
        unchanged = _document("same", revision_id="r1", text="same")
        target = _request(
            (
                unchanged,
                _document("added", revision_id="r1", text="added"),
                _document("changed", revision_id="r2", text="new"),
            )
        )
        previous = _request(
            (
                unchanged,
                _document("changed", revision_id="r1", text="old"),
                _document("removed", revision_id="r1", text="gone"),
            ),
            projection=target.projection,
            chunking=target.chunking,
        )
        embedder, writer, _, _ = _collaborators(target)

        result = project_documents(
            target,
            _present(previous),
            _borrow_embedder(embedder),
            _borrow_writer(writer),
        )

        self.assertIs(result.receipt.outcome, ProjectionOutcome.COMPLETED)
        self.assertEqual(result.receipt.attempted_documents, 3)
        self.assertEqual(result.receipt.completed_documents, 3)
        self.assertEqual(result.receipt.checkpoint, result.manifest.checkpoint)
        self.assertNotIn(unchanged.identity, [item[0] for item in writer.replacements])


class RebuildProjectionTests(unittest.TestCase):
    def test_nonempty_rebuild_validates_identity_once_before_reset_and_reuses_it(
        self,
    ) -> None:
        request = _request((_document("alpha"), _document("beta")))
        embedder, writer, resetter, events = _collaborators(request)

        result = rebuild_projection(
            request,
            ProjectionStateSnapshot(ProjectionStateAvailability.MISSING, None),
            _borrow_embedder(embedder),
            _borrow_writer(writer),
            _borrow_resetter(resetter),
        )

        self.assertEqual(embedder.identity_calls, 1)
        self.assertEqual(resetter.corpora, ["corpus"])
        self.assertEqual(
            [identity.key.document_id for identity, _ in writer.replacements],
            ["alpha", "beta"],
        )
        self.assertEqual(events[0:2], ["identity", "reset:corpus"])
        self.assertEqual(sum(event == "identity" for event in events), 1)
        self.assertIs(result.status_before, ProjectionStateStatus.MISSING)
        self.assertEqual(result.receipt.attempted_documents, 2)

    def test_wrong_embedder_identity_fails_before_reset_without_a_cause(self) -> None:
        request = _request((_document("alpha"),))
        events: list[str] = []
        embedder = _FakeEmbedder(
            request.projection.embedding,
            events,
            identity_value=EmbeddingIdentity("other-model", 2),
        )
        writer = _FakeWriter(events)
        resetter = _FakeResetter(events)

        with self.assertRaises(ProjectionOperationError) as raised:
            rebuild_projection(
                request,
                ProjectionStateSnapshot(ProjectionStateAvailability.MISSING, None),
                _borrow_embedder(embedder),
                _borrow_writer(writer),
                _borrow_resetter(resetter),
            )

        self.assertIs(
            raised.exception.stage,
            ProjectionFailureStage.EMBEDDER_IDENTITY,
        )
        self.assertIsNone(raised.exception.affected_document)
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNotNone(raised.exception.receipt)
        assert raised.exception.receipt is not None
        self.assertIs(raised.exception.receipt.outcome, ProjectionOutcome.FAILED)
        self.assertEqual(resetter.corpora, [])
        self.assertEqual(writer.replacements, [])

    def test_identity_exception_is_preserved_as_cause_before_reset(self) -> None:
        request = _request((_document("alpha"),))
        failure = RuntimeError("identity provider failed")
        events: list[str] = []
        embedder = _FakeEmbedder(
            request.projection.embedding,
            events,
            identity_failure=failure,
        )
        writer = _FakeWriter(events)
        resetter = _FakeResetter(events)

        with self.assertRaises(ProjectionOperationError) as raised:
            rebuild_projection(
                request,
                ProjectionStateSnapshot(ProjectionStateAvailability.MISSING, None),
                _borrow_embedder(embedder),
                _borrow_writer(writer),
                _borrow_resetter(resetter),
            )

        self.assertIs(raised.exception.__cause__, failure)
        self.assertEqual(resetter.corpora, [])

    def test_empty_rebuild_resets_only_without_embedder_or_writer_access(self) -> None:
        request = _request(())
        embedder, writer, resetter, events = _collaborators(request)
        embedder.identity_failure = AssertionError("empty rebuild must not embed")
        writer.replace_failure_at = 0

        result = rebuild_projection(
            request,
            ProjectionStateSnapshot(ProjectionStateAvailability.CORRUPT, None),
            _borrow_embedder(embedder),
            _borrow_writer(writer),
            _borrow_resetter(resetter),
        )

        self.assertEqual(events, ["reset:corpus"])
        self.assertEqual(embedder.identity_calls, 0)
        self.assertEqual(writer.replacements, [])
        self.assertIs(result.status_before, ProjectionStateStatus.CORRUPT)
        self.assertEqual(result.receipt.attempted_documents, 0)
        self.assertEqual(result.receipt.completed_documents, 0)
        self.assertIs(result.receipt.outcome, ProjectionOutcome.COMPLETED)

    def test_rebuild_accepts_every_state_status_and_reports_it_truthfully(self) -> None:
        request = _request((_document("alpha"),))
        other_schema = _request(
            request.documents,
            projection=_identity(schema_id="other"),
        )
        other_embedding = _request(
            request.documents,
            projection=_identity(model_id="other"),
        )
        stale = _request(
            (_document("alpha", revision_id="old"),),
            projection=request.projection,
        )
        states = (
            (
                ProjectionStateSnapshot(ProjectionStateAvailability.MISSING, None),
                ProjectionStateStatus.MISSING,
            ),
            (
                ProjectionStateSnapshot(ProjectionStateAvailability.CORRUPT, None),
                ProjectionStateStatus.CORRUPT,
            ),
            (_present(request), ProjectionStateStatus.CURRENT),
            (_present(stale), ProjectionStateStatus.STALE),
            (_present(other_schema), ProjectionStateStatus.SCHEMA_MISMATCH),
            (_present(other_embedding), ProjectionStateStatus.EMBEDDING_MISMATCH),
        )

        for state, status in states:
            embedder, writer, resetter, _ = _collaborators(request)
            with self.subTest(status=status):
                result = rebuild_projection(
                    request,
                    state,
                    _borrow_embedder(embedder),
                    _borrow_writer(writer),
                    _borrow_resetter(resetter),
                )
                self.assertIs(result.status_before, status)
                self.assertEqual(resetter.corpora, ["corpus"])
                self.assertIs(result.receipt.outcome, ProjectionOutcome.COMPLETED)

    def test_reset_exception_has_failed_receipt_and_preserves_cause(self) -> None:
        request = _request((_document("alpha"), _document("beta")))
        failure = RuntimeError("reset failed")
        events: list[str] = []
        embedder = _FakeEmbedder(request.projection.embedding, events)
        writer = _FakeWriter(events)
        resetter = _FakeResetter(events, failure=failure)

        with self.assertRaises(ProjectionOperationError) as raised:
            rebuild_projection(
                request,
                ProjectionStateSnapshot(ProjectionStateAvailability.MISSING, None),
                _borrow_embedder(embedder),
                _borrow_writer(writer),
                _borrow_resetter(resetter),
            )

        error = raised.exception
        self.assertIs(error.stage, ProjectionFailureStage.RESET)
        self.assertIsNone(error.affected_document)
        self.assertIs(error.__cause__, failure)
        self.assertIsNotNone(error.receipt)
        assert error.receipt is not None
        self.assertIs(error.receipt.outcome, ProjectionOutcome.FAILED)
        self.assertEqual(error.receipt.attempted_documents, 2)
        self.assertEqual(error.receipt.completed_documents, 0)
        self.assertIsNone(error.receipt.checkpoint)
        self.assertEqual(writer.replacements, [])

    def test_non_none_reset_result_is_failure_without_cause(self) -> None:
        for documents, expected_receipt in (
            ((_document("alpha"),), True),
            ((), False),
        ):
            request = _request(documents)
            events: list[str] = []
            embedder = _FakeEmbedder(request.projection.embedding, events)
            writer = _FakeWriter(events)
            resetter = _FakeResetter(events, result=False)
            with self.subTest(documents=documents):
                with self.assertRaises(ProjectionOperationError) as raised:
                    rebuild_projection(
                        request,
                        ProjectionStateSnapshot(
                            ProjectionStateAvailability.MISSING,
                            None,
                        ),
                        _borrow_embedder(embedder),
                        _borrow_writer(writer),
                        _borrow_resetter(resetter),
                    )
                self.assertIs(raised.exception.stage, ProjectionFailureStage.RESET)
                self.assertIsNone(raised.exception.__cause__)
                self.assertEqual(
                    raised.exception.receipt is not None,
                    expected_receipt,
                )


class ProjectionFailureTests(unittest.TestCase):
    def _incremental_target(
        self,
        documents: tuple[Document, ...],
        *,
        batch_size: int = 2,
    ) -> tuple[ProjectionRequest, ProjectionStateSnapshot]:
        target = _request(documents, batch_size=batch_size)
        previous = _request(
            (),
            projection=target.projection,
            chunking=target.chunking,
            batch_size=batch_size,
        )
        return target, _present(previous)

    def test_embedding_exception_is_called_once_and_preserved_as_cause(self) -> None:
        request, state = self._incremental_target((_document("alpha", text="x"),))
        failure = RuntimeError("provider failed")
        events: list[str] = []
        embedder = _FakeEmbedder(
            request.projection.embedding,
            events,
            embed_failure_at=0,
            embed_failure=failure,
        )
        writer = _FakeWriter(events)

        with self.assertRaises(ProjectionOperationError) as raised:
            project_documents(
                request,
                state,
                _borrow_embedder(embedder),
                _borrow_writer(writer),
            )

        error = raised.exception
        self.assertIs(error.stage, ProjectionFailureStage.EMBEDDING)
        self.assertEqual(error.affected_document, DocumentKey("corpus", "alpha"))
        self.assertIs(error.__cause__, failure)
        self.assertEqual(embedder.embed_calls, [("x",)])
        self.assertEqual(writer.replacements, [])
        self.assertIsNotNone(error.receipt)
        assert error.receipt is not None
        self.assertIs(error.receipt.outcome, ProjectionOutcome.FAILED)
        self.assertEqual(error.receipt.completed_documents, 0)
        self.assertIsNone(error.receipt.checkpoint)

    def test_malformed_embedding_outputs_fail_without_internal_cause(self) -> None:
        request, state = self._incremental_target((_document("alpha", text="x"),))
        nonfinite = EmbeddingVector((1.0, 2.0))
        object.__setattr__(nonfinite, "values", (float("nan"), 2.0))
        noncanonical = EmbeddingVector((1.0, 2.0))
        object.__setattr__(noncanonical, "values", (1, 2.0))
        outputs: tuple[object, ...] = (
            [EmbeddingVector((1.0, 2.0))],
            (),
            (object(),),
            (EmbeddingVector((1.0,)),),
            (EmbeddingVector((1.0, 2.0, 3.0)),),
            (nonfinite,),
            (noncanonical,),
        )

        for output in outputs:
            events: list[str] = []
            embedder = _FakeEmbedder(
                request.projection.embedding,
                events,
                output=_constant_output(output),
            )
            writer = _FakeWriter(events)
            with self.subTest(output=output):
                with self.assertRaises(ProjectionOperationError) as raised:
                    project_documents(
                        request,
                        state,
                        _borrow_embedder(embedder),
                        _borrow_writer(writer),
                    )
                self.assertIs(
                    raised.exception.stage,
                    ProjectionFailureStage.EMBEDDING,
                )
                self.assertIsNone(raised.exception.__cause__)
                self.assertEqual(len(embedder.embed_calls), 1)
                self.assertEqual(writer.replacements, [])

    def test_second_document_embedding_failure_reports_partial_completion(self) -> None:
        request, state = self._incremental_target(
            (_document("alpha"), _document("beta")),
            batch_size=2,
        )
        events: list[str] = []
        failure = RuntimeError("second document failed")
        embedder = _FakeEmbedder(
            request.projection.embedding,
            events,
            embed_failure_at=2,
            embed_failure=failure,
        )
        writer = _FakeWriter(events)

        with self.assertRaises(ProjectionOperationError) as raised:
            project_documents(
                request,
                state,
                _borrow_embedder(embedder),
                _borrow_writer(writer),
            )

        error = raised.exception
        self.assertEqual(error.affected_document, DocumentKey("corpus", "beta"))
        self.assertIsNotNone(error.receipt)
        assert error.receipt is not None
        self.assertIs(error.receipt.outcome, ProjectionOutcome.PARTIAL)
        self.assertEqual(error.receipt.attempted_documents, 2)
        self.assertEqual(error.receipt.completed_documents, 1)
        self.assertEqual(
            [identity.key.document_id for identity, _ in writer.replacements],
            ["alpha"],
        )
        self.assertIs(error.__cause__, failure)

    def test_replacement_exception_has_matching_failed_receipt_and_cause(self) -> None:
        request, state = self._incremental_target((_document("alpha", text="x"),))
        events: list[str] = []
        failure = RuntimeError("writer failed")
        embedder = _FakeEmbedder(request.projection.embedding, events)
        writer = _FakeWriter(
            events,
            replace_failure_at=0,
            replace_failure=failure,
        )

        with self.assertRaises(ProjectionOperationError) as raised:
            project_documents(
                request,
                state,
                _borrow_embedder(embedder),
                _borrow_writer(writer),
            )

        error = raised.exception
        self.assertIs(error.stage, ProjectionFailureStage.REPLACEMENT)
        self.assertEqual(error.affected_document, DocumentKey("corpus", "alpha"))
        self.assertIs(error.__cause__, failure)
        self.assertIsNotNone(error.receipt)
        assert error.receipt is not None
        self.assertIs(error.receipt.outcome, ProjectionOutcome.FAILED)
        self.assertEqual(len(writer.replacements), 1)

    def test_non_none_replacement_result_is_failure_without_cause(self) -> None:
        request, state = self._incremental_target((_document("alpha", text="x"),))
        events: list[str] = []
        embedder = _FakeEmbedder(request.projection.embedding, events)
        writer = _FakeWriter(events, replace_result=False)

        with self.assertRaises(ProjectionOperationError) as raised:
            project_documents(
                request,
                state,
                _borrow_embedder(embedder),
                _borrow_writer(writer),
            )

        self.assertIs(raised.exception.stage, ProjectionFailureStage.REPLACEMENT)
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNotNone(raised.exception.receipt)
        assert raised.exception.receipt is not None
        self.assertEqual(raised.exception.receipt.completed_documents, 0)

    def test_deletion_exception_after_replacement_reports_partial_receipt(self) -> None:
        target = _request((_document("alpha", text="x"),))
        previous = _request(
            (_document("beta", text="x"),),
            projection=target.projection,
            chunking=target.chunking,
        )
        events: list[str] = []
        failure = RuntimeError("delete failed")
        embedder = _FakeEmbedder(target.projection.embedding, events)
        writer = _FakeWriter(
            events,
            delete_failure_at=0,
            delete_failure=failure,
        )

        with self.assertRaises(ProjectionOperationError) as raised:
            project_documents(
                target,
                _present(previous),
                _borrow_embedder(embedder),
                _borrow_writer(writer),
            )

        error = raised.exception
        self.assertIs(error.stage, ProjectionFailureStage.DELETION)
        self.assertEqual(error.affected_document, DocumentKey("corpus", "beta"))
        self.assertIs(error.__cause__, failure)
        self.assertIsNotNone(error.receipt)
        assert error.receipt is not None
        self.assertIs(error.receipt.outcome, ProjectionOutcome.PARTIAL)
        self.assertEqual(error.receipt.attempted_documents, 2)
        self.assertEqual(error.receipt.completed_documents, 1)
        self.assertIsNone(error.receipt.checkpoint)
        self.assertEqual(
            [event for event in events if event.startswith(("replace:", "delete:"))],
            ["replace:alpha", "delete:beta"],
        )

    def test_non_none_deletion_result_is_failed_without_cause(self) -> None:
        target = _request(())
        previous = _request(
            (_document("alpha", text="x"),),
            projection=target.projection,
            chunking=target.chunking,
        )
        events: list[str] = []
        embedder = _FakeEmbedder(target.projection.embedding, events)
        writer = _FakeWriter(events, delete_result=0)

        with self.assertRaises(ProjectionOperationError) as raised:
            project_documents(
                target,
                _present(previous),
                _borrow_embedder(embedder),
                _borrow_writer(writer),
            )

        self.assertIs(raised.exception.stage, ProjectionFailureStage.DELETION)
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNotNone(raised.exception.receipt)
        assert raised.exception.receipt is not None
        self.assertIs(raised.exception.receipt.outcome, ProjectionOutcome.FAILED)

    def test_rebuild_replacement_failure_never_claims_a_checkpoint(self) -> None:
        request = _request((_document("alpha"), _document("beta")))
        events: list[str] = []
        embedder = _FakeEmbedder(request.projection.embedding, events)
        writer = _FakeWriter(events, replace_failure_at=1)
        resetter = _FakeResetter(events)

        with self.assertRaises(ProjectionOperationError) as raised:
            rebuild_projection(
                request,
                ProjectionStateSnapshot(ProjectionStateAvailability.CORRUPT, None),
                _borrow_embedder(embedder),
                _borrow_writer(writer),
                _borrow_resetter(resetter),
            )

        error = raised.exception
        self.assertIs(error.stage, ProjectionFailureStage.REPLACEMENT)
        self.assertEqual(error.affected_document, DocumentKey("corpus", "beta"))
        self.assertIsNotNone(error.receipt)
        assert error.receipt is not None
        self.assertIs(error.receipt.outcome, ProjectionOutcome.PARTIAL)
        self.assertEqual(error.receipt.completed_documents, 1)
        self.assertIsNone(error.receipt.checkpoint)
        self.assertEqual(resetter.corpora, ["corpus"])

    def test_control_flow_exceptions_are_never_translated_or_retried(self) -> None:
        for failure in (KeyboardInterrupt("interrupt"), SystemExit("exit")):
            request, state = self._incremental_target((_document("alpha", text="x"),))
            events: list[str] = []
            embedder = _FakeEmbedder(
                request.projection.embedding,
                events,
                embed_failure_at=0,
                embed_failure=failure,
            )
            writer = _FakeWriter(events)
            with self.subTest(failure=type(failure).__name__):
                with self.assertRaises(type(failure)) as raised:
                    project_documents(
                        request,
                        state,
                        _borrow_embedder(embedder),
                        _borrow_writer(writer),
                    )
                self.assertIs(raised.exception, failure)
                self.assertEqual(len(embedder.embed_calls), 1)
                self.assertEqual(writer.replacements, [])

    def test_reset_control_flow_exception_is_not_translated_or_retried(self) -> None:
        request = _request((_document("alpha"),))
        failure = KeyboardInterrupt("reset interrupted")
        events: list[str] = []
        embedder = _FakeEmbedder(request.projection.embedding, events)
        writer = _FakeWriter(events)
        resetter = _FakeResetter(events, failure=failure)

        with self.assertRaises(KeyboardInterrupt) as raised:
            rebuild_projection(
                request,
                ProjectionStateSnapshot(ProjectionStateAvailability.MISSING, None),
                _borrow_embedder(embedder),
                _borrow_writer(writer),
                _borrow_resetter(resetter),
            )

        self.assertIs(raised.exception, failure)
        self.assertEqual(resetter.corpora, ["corpus"])
        self.assertEqual(writer.replacements, [])

    def test_workflows_never_enter_close_shutdown_or_suppress_resources(self) -> None:
        request, state = self._incremental_target((_document("alpha", text="x"),))
        embedder, writer, resetter, _ = _collaborators(request)

        project_documents(
            request,
            state,
            _borrow_embedder(embedder),
            _borrow_writer(writer),
        )
        rebuild_projection(
            request,
            ProjectionStateSnapshot(ProjectionStateAvailability.MISSING, None),
            _borrow_embedder(embedder),
            _borrow_writer(writer),
            _borrow_resetter(resetter),
        )

        self.assertEqual(embedder.lifecycle_calls, [])
        self.assertEqual(writer.lifecycle_calls, [])
        self.assertEqual(resetter.lifecycle_calls, [])


if __name__ == "__main__":
    unittest.main()
