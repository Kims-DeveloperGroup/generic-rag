"""Contract tests for immutable projection values and truthful state."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields
from typing import cast

from generic_rag.contracts import (
    ChunkingPolicy,
    Document,
    DocumentIdentity,
    DocumentKey,
    EmbeddingIdentity,
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
)
from generic_rag.errors import ContractValidationError


class _StringSubclass(str):
    pass


class _IntegerSubclass(int):
    pass


def _projection(
    *,
    schema_id: str = "schema",
    model_id: str = "model",
    dimensions: int = 2,
) -> ProjectionIdentity:
    return ProjectionIdentity(
        schema_id=schema_id,
        embedding=EmbeddingIdentity(model_id, dimensions),
    )


def _checkpoint(
    *,
    corpus_id: str = "corpus",
    projection: ProjectionIdentity | None = None,
    token: str = "checkpoint",
) -> ProjectionCheckpoint:
    return ProjectionCheckpoint(
        corpus_id=corpus_id,
        projection=projection or _projection(),
        token=token,
    )


def _document(
    document_id: str = "document",
    *,
    corpus_id: str = "corpus",
    revision_id: str = "revision",
    text: str = "text",
) -> Document:
    return Document(
        DocumentIdentity(DocumentKey(corpus_id, document_id), revision_id),
        text,
    )


def _entry(
    document_id: str = "document",
    *,
    corpus_id: str = "corpus",
    revision_id: str = "revision",
    digest_digit: str = "0",
    fragment_count: int = 1,
) -> ProjectionManifestEntry:
    return ProjectionManifestEntry(
        _document(
            document_id,
            corpus_id=corpus_id,
            revision_id=revision_id,
        ).identity,
        f"sha256:{digest_digit * 64}",
        fragment_count,
    )


def _manifest(
    *,
    corpus_id: str = "corpus",
    projection: ProjectionIdentity | None = None,
    chunking: ChunkingPolicy | None = None,
    entries: tuple[ProjectionManifestEntry, ...] | None = None,
    token: str = "checkpoint",
) -> ProjectionManifest:
    resolved_projection = projection or _projection()
    return ProjectionManifest(
        corpus_id,
        resolved_projection,
        chunking or ChunkingPolicy(4, 1),
        (_entry(corpus_id=corpus_id),) if entries is None else entries,
        _checkpoint(
            corpus_id=corpus_id,
            projection=resolved_projection,
            token=token,
        ),
    )


class ProjectionContractTests(unittest.TestCase):
    def test_projection_fields_are_exact_frozen_and_slotted(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(ProjectionIdentity)),
            ("schema_id", "embedding"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(ProjectionCheckpoint)),
            ("corpus_id", "projection", "token"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(ProjectionReceipt)),
            (
                "corpus_id",
                "projection",
                "outcome",
                "attempted_documents",
                "completed_documents",
                "checkpoint",
            ),
        )

        projection = _projection()
        checkpoint = _checkpoint(projection=projection)
        receipt = ProjectionReceipt(
            "corpus",
            projection,
            ProjectionOutcome.COMPLETED,
            0,
            0,
            checkpoint,
        )
        for instance, field_name in (
            (projection, "schema_id"),
            (checkpoint, "token"),
            (receipt, "completed_documents"),
        ):
            with self.subTest(instance=type(instance).__name__):
                self.assertFalse(hasattr(instance, "__dict__"))
                with self.assertRaises(FrozenInstanceError):
                    setattr(instance, field_name, object())
                self.assertIsInstance(hash(instance), int)

    def test_projection_outcomes_are_exact_closed_string_enums(self) -> None:
        self.assertEqual(
            tuple((member.name, member.value) for member in ProjectionOutcome),
            (
                ("COMPLETED", "completed"),
                ("UNCHANGED", "unchanged"),
                ("PARTIAL", "partial"),
                ("FAILED", "failed"),
            ),
        )
        self.assertEqual(ProjectionOutcome("completed"), ProjectionOutcome.COMPLETED)
        self.assertIsInstance(ProjectionOutcome.COMPLETED, str)

        for invalid in ("COMPLETED", "", "unknown", None, 1, object()):
            with self.subTest(value=invalid):
                with self.assertRaises(ContractValidationError):
                    ProjectionOutcome(cast(str, invalid))

    def test_projection_identity_preserves_opaque_strings(self) -> None:
        schema_id = "  Schema/../e\u0301  "
        model_id = " Model:ABC "
        projection = _projection(schema_id=schema_id, model_id=model_id)

        self.assertEqual(projection.schema_id, schema_id)
        self.assertEqual(projection.embedding.model_id, model_id)

    def test_projection_identity_rejects_invalid_fields(self) -> None:
        invalid_strings: tuple[object, ...] = (
            "",
            " \t",
            None,
            b"schema",
            _StringSubclass("schema"),
        )
        for invalid in invalid_strings:
            with self.subTest(value=invalid):
                with self.assertRaises(ContractValidationError):
                    ProjectionIdentity(
                        cast(str, invalid),
                        EmbeddingIdentity("model", 2),
                    )

        with self.assertRaises(ContractValidationError):
            ProjectionIdentity("schema", cast(EmbeddingIdentity, object()))

    def test_checkpoint_requires_exact_opaque_identity_values(self) -> None:
        corpus_id = " Corpus/../A "
        token = " Token/../001 "
        projection = _projection()
        checkpoint = _checkpoint(
            corpus_id=corpus_id,
            projection=projection,
            token=token,
        )

        self.assertEqual(checkpoint.corpus_id, corpus_id)
        self.assertEqual(checkpoint.token, token)
        self.assertIs(checkpoint.projection, projection)

        for invalid in ("", " \n", None, _StringSubclass("value")):
            with self.subTest(field="corpus_id", value=invalid):
                with self.assertRaises(ContractValidationError):
                    ProjectionCheckpoint(
                        cast(str, invalid),
                        projection,
                        "token",
                    )
            with self.subTest(field="token", value=invalid):
                with self.assertRaises(ContractValidationError):
                    ProjectionCheckpoint(
                        "corpus",
                        projection,
                        cast(str, invalid),
                    )
        with self.assertRaises(ContractValidationError):
            ProjectionCheckpoint(
                "corpus",
                cast(ProjectionIdentity, object()),
                "token",
            )

    def test_completed_and_unchanged_receipts_require_truthful_checkpoints(
        self,
    ) -> None:
        projection = _projection()
        checkpoint = _checkpoint(projection=projection)

        for outcome in (
            ProjectionOutcome.COMPLETED,
            ProjectionOutcome.UNCHANGED,
        ):
            with self.subTest(outcome=outcome):
                empty = ProjectionReceipt(
                    "corpus",
                    projection,
                    outcome,
                    0,
                    0,
                    checkpoint,
                )
                complete = ProjectionReceipt(
                    "corpus",
                    projection,
                    outcome,
                    3,
                    3,
                    checkpoint,
                )
                self.assertEqual(empty.completed_documents, 0)
                self.assertEqual(complete.completed_documents, 3)
                self.assertIs(complete.checkpoint, checkpoint)

                with self.assertRaises(ContractValidationError):
                    ProjectionReceipt(
                        "corpus",
                        projection,
                        outcome,
                        3,
                        2,
                        checkpoint,
                    )
                with self.assertRaises(ContractValidationError):
                    ProjectionReceipt(
                        "corpus",
                        projection,
                        outcome,
                        3,
                        3,
                        None,
                    )

    def test_partial_and_failed_receipts_cannot_claim_completed_work(self) -> None:
        projection = _projection()
        partial = ProjectionReceipt(
            "corpus",
            projection,
            ProjectionOutcome.PARTIAL,
            3,
            1,
            None,
        )
        failed = ProjectionReceipt(
            "corpus",
            projection,
            ProjectionOutcome.FAILED,
            2,
            0,
            None,
        )

        self.assertEqual(partial.completed_documents, 1)
        self.assertIsNone(partial.checkpoint)
        self.assertEqual(failed.completed_documents, 0)
        self.assertIsNone(failed.checkpoint)

        invalid_partial_counts = ((0, 0), (1, 0), (1, 1), (2, 2))
        for attempted, completed in invalid_partial_counts:
            with self.subTest(attempted=attempted, completed=completed):
                with self.assertRaises(ContractValidationError):
                    ProjectionReceipt(
                        "corpus",
                        projection,
                        ProjectionOutcome.PARTIAL,
                        attempted,
                        completed,
                        None,
                    )

        invalid_failed_counts = ((0, 0), (1, 1), (2, 1))
        for attempted, completed in invalid_failed_counts:
            with self.subTest(attempted=attempted, completed=completed):
                with self.assertRaises(ContractValidationError):
                    ProjectionReceipt(
                        "corpus",
                        projection,
                        ProjectionOutcome.FAILED,
                        attempted,
                        completed,
                        None,
                    )

        for outcome, attempted, completed in (
            (ProjectionOutcome.PARTIAL, 2, 1),
            (ProjectionOutcome.FAILED, 2, 0),
        ):
            with self.subTest(outcome=outcome):
                with self.assertRaises(ContractValidationError):
                    ProjectionReceipt(
                        "corpus",
                        projection,
                        outcome,
                        attempted,
                        completed,
                        _checkpoint(projection=projection),
                    )

    def test_receipt_checkpoint_must_match_corpus_and_projection(self) -> None:
        projection = _projection()
        mismatched_projection = _projection(schema_id="other-schema")

        for checkpoint in (
            _checkpoint(corpus_id="other-corpus", projection=projection),
            _checkpoint(corpus_id="corpus", projection=mismatched_projection),
        ):
            with self.subTest(checkpoint=checkpoint):
                with self.assertRaises(ContractValidationError):
                    ProjectionReceipt(
                        "corpus",
                        projection,
                        ProjectionOutcome.COMPLETED,
                        1,
                        1,
                        checkpoint,
                    )

    def test_receipt_rejects_inexact_nested_and_discrete_types(self) -> None:
        projection = _projection()
        checkpoint = _checkpoint(projection=projection)

        with self.assertRaises(ContractValidationError):
            ProjectionReceipt(
                "corpus",
                cast(ProjectionIdentity, object()),
                ProjectionOutcome.COMPLETED,
                1,
                1,
                checkpoint,
            )
        with self.assertRaises(ContractValidationError):
            ProjectionReceipt(
                "corpus",
                projection,
                cast(ProjectionOutcome, "completed"),
                1,
                1,
                checkpoint,
            )
        with self.assertRaises(ContractValidationError):
            ProjectionReceipt(
                "corpus",
                projection,
                ProjectionOutcome.COMPLETED,
                1,
                1,
                cast(ProjectionCheckpoint, object()),
            )

        for invalid in (-1, True, 1.0, _IntegerSubclass(1)):
            with self.subTest(field="attempted_documents", value=invalid):
                with self.assertRaises(ContractValidationError):
                    ProjectionReceipt(
                        "corpus",
                        projection,
                        ProjectionOutcome.COMPLETED,
                        cast(int, invalid),
                        1,
                        checkpoint,
                    )
            with self.subTest(field="completed_documents", value=invalid):
                with self.assertRaises(ContractValidationError):
                    ProjectionReceipt(
                        "corpus",
                        projection,
                        ProjectionOutcome.COMPLETED,
                        1,
                        cast(int, invalid),
                        checkpoint,
                    )

        with self.assertRaises(ContractValidationError):
            ProjectionReceipt(
                "corpus",
                projection,
                ProjectionOutcome.COMPLETED,
                1,
                2,
                checkpoint,
            )


class ProjectionWorkflowContractTests(unittest.TestCase):
    def test_new_projection_values_have_exact_frozen_slotted_fields(self) -> None:
        expected_fields = {
            ChunkingPolicy: ("max_fragment_codepoints", "overlap_codepoints"),
            ProjectionLimits: (
                "max_documents",
                "max_document_codepoints",
                "max_embedding_batch_size",
            ),
            ProjectionRequest: (
                "corpus_id",
                "projection",
                "chunking",
                "limits",
                "documents",
            ),
            ProjectionManifestEntry: (
                "document",
                "source_digest",
                "fragment_count",
            ),
            ProjectionManifest: (
                "corpus_id",
                "projection",
                "chunking",
                "entries",
                "checkpoint",
            ),
            ProjectionStateSnapshot: ("availability", "manifest"),
            ProjectionResult: ("status_before", "receipt", "manifest"),
        }
        projection = _projection()
        manifest = _manifest(projection=projection)
        instances = (
            ChunkingPolicy(4, 1),
            ProjectionLimits(2, 20, 2),
            ProjectionRequest(
                "corpus",
                projection,
                ChunkingPolicy(4, 1),
                ProjectionLimits(2, 20, 2),
                (_document(),),
            ),
            manifest.entries[0],
            manifest,
            ProjectionStateSnapshot(ProjectionStateAvailability.PRESENT, manifest),
            ProjectionResult(
                ProjectionStateStatus.STALE,
                ProjectionReceipt(
                    "corpus",
                    projection,
                    ProjectionOutcome.COMPLETED,
                    1,
                    1,
                    manifest.checkpoint,
                ),
                manifest,
            ),
        )

        for instance in instances:
            with self.subTest(value=type(instance).__name__):
                self.assertEqual(
                    tuple(field.name for field in fields(type(instance))),
                    expected_fields[type(instance)],
                )
                self.assertFalse(hasattr(instance, "__dict__"))
                with self.assertRaises(FrozenInstanceError):
                    setattr(instance, expected_fields[type(instance)][0], object())
                self.assertIsInstance(hash(instance), int)

    def test_chunking_policy_uses_positive_codepoint_size_and_bounded_overlap(
        self,
    ) -> None:
        self.assertEqual(ChunkingPolicy(4, 0), ChunkingPolicy(4, 0))
        self.assertEqual(ChunkingPolicy(4, 3).overlap_codepoints, 3)

        for maximum, overlap in (
            (0, 0),
            (-1, 0),
            (4, -1),
            (4, 4),
            (4, 5),
            (True, 0),
            (4, True),
            (4.0, 0),
            (4, 1.0),
            (_IntegerSubclass(4), 0),
        ):
            with self.subTest(maximum=maximum, overlap=overlap):
                with self.assertRaises(ContractValidationError):
                    ChunkingPolicy(cast(int, maximum), cast(int, overlap))

    def test_projection_limits_require_three_positive_exact_integers(self) -> None:
        self.assertEqual(ProjectionLimits(1, 1, 1), ProjectionLimits(1, 1, 1))

        for field_index in range(3):
            for invalid in (0, -1, True, 1.0, _IntegerSubclass(1)):
                values: list[object] = [2, 20, 3]
                values[field_index] = invalid
                with self.subTest(field=field_index, value=invalid):
                    with self.assertRaises(ContractValidationError):
                        ProjectionLimits(
                            cast(int, values[0]),
                            cast(int, values[1]),
                            cast(int, values[2]),
                        )

    def test_request_preserves_opaque_values_and_canonicalizes_document_order(
        self,
    ) -> None:
        documents = (
            _document("zeta", text=""),
            _document("alpha", revision_id=" revision/../2 ", text="e\u0301"),
        )
        request = ProjectionRequest(
            "corpus",
            _projection(),
            ChunkingPolicy(2, 1),
            ProjectionLimits(2, 2, 1),
            documents,
        )

        self.assertEqual(
            tuple(document.identity.key.document_id for document in request.documents),
            ("alpha", "zeta"),
        )
        self.assertEqual(request.documents[0].identity.revision_id, " revision/../2 ")
        self.assertEqual(request.documents[0].text, "e\u0301")
        self.assertEqual(
            ProjectionRequest(
                "corpus",
                _projection(),
                ChunkingPolicy(2, 1),
                ProjectionLimits(2, 2, 1),
                (),
            ).documents,
            (),
        )

    def test_request_rejects_inexact_fields_and_document_bound_violations(
        self,
    ) -> None:
        projection = _projection()
        chunking = ChunkingPolicy(4, 1)
        limits = ProjectionLimits(1, 4, 2)
        valid = _document(text="four")
        invalid_requests = (
            ("", projection, chunking, limits, (valid,)),
            (_StringSubclass("corpus"), projection, chunking, limits, (valid,)),
            ("corpus", object(), chunking, limits, (valid,)),
            ("corpus", projection, object(), limits, (valid,)),
            ("corpus", projection, chunking, object(), (valid,)),
            ("corpus", projection, chunking, limits, [valid]),
            ("corpus", projection, chunking, limits, (object(),)),
            ("corpus", projection, chunking, limits, (valid, _document("two"))),
            ("corpus", projection, chunking, limits, (_document(text="12345"),)),
            (
                "corpus",
                projection,
                chunking,
                limits,
                (_document(corpus_id="other"),),
            ),
            (
                "corpus",
                projection,
                chunking,
                ProjectionLimits(2, 4, 2),
                (_document(revision_id="r1"), _document(revision_id="r2")),
            ),
        )
        for values in invalid_requests:
            with self.subTest(values=values):
                with self.assertRaises(ContractValidationError):
                    ProjectionRequest(
                        values[0],
                        cast(ProjectionIdentity, values[1]),
                        cast(ChunkingPolicy, values[2]),
                        cast(ProjectionLimits, values[3]),
                        cast(tuple[Document, ...], values[4]),
                    )

    def test_manifest_entry_requires_lowercase_sha256_and_nonnegative_count(
        self,
    ) -> None:
        entry = _entry(fragment_count=0)
        self.assertEqual(entry.source_digest, "sha256:" + "0" * 64)
        self.assertEqual(entry.fragment_count, 0)

        for digest in (
            "",
            "0" * 64,
            "sha256:" + "A" * 64,
            "sha256:" + "0" * 63,
            "sha256:" + "0" * 65,
            _StringSubclass("sha256:" + "0" * 64),
        ):
            with self.subTest(digest=digest):
                with self.assertRaises(ContractValidationError):
                    ProjectionManifestEntry(
                        _document().identity,
                        digest,
                        1,
                    )
        for count in (-1, True, 1.0, _IntegerSubclass(1)):
            with self.subTest(count=count):
                with self.assertRaises(ContractValidationError):
                    ProjectionManifestEntry(
                        _document().identity,
                        "sha256:" + "0" * 64,
                        cast(int, count),
                    )
        with self.assertRaises(ContractValidationError):
            ProjectionManifestEntry(
                cast(DocumentIdentity, object()),
                "sha256:" + "0" * 64,
                1,
            )

    def test_manifest_canonicalizes_entries_and_preserves_checkpoint(self) -> None:
        projection = _projection()
        checkpoint = _checkpoint(projection=projection, token=" token/../x ")
        zeta = _entry("zeta", digest_digit="1")
        alpha = _entry("alpha", digest_digit="2")
        manifest = ProjectionManifest(
            "corpus",
            projection,
            ChunkingPolicy(4, 1),
            (zeta, alpha),
            checkpoint,
        )

        self.assertEqual(manifest.entries, (alpha, zeta))
        self.assertIs(manifest.checkpoint, checkpoint)
        self.assertEqual(manifest.checkpoint.token, " token/../x ")

    def test_manifest_rejects_mismatched_inexact_or_duplicate_content(self) -> None:
        projection = _projection()
        chunking = ChunkingPolicy(4, 1)
        checkpoint = _checkpoint(projection=projection)
        entry = _entry()
        invalid_values = (
            ("", projection, chunking, (entry,), checkpoint),
            ("corpus", object(), chunking, (entry,), checkpoint),
            ("corpus", projection, object(), (entry,), checkpoint),
            ("corpus", projection, chunking, [entry], checkpoint),
            ("corpus", projection, chunking, (object(),), checkpoint),
            ("corpus", projection, chunking, (entry,), object()),
            (
                "corpus",
                projection,
                chunking,
                (_entry(corpus_id="other"),),
                checkpoint,
            ),
            ("corpus", projection, chunking, (entry, entry), checkpoint),
            (
                "corpus",
                projection,
                chunking,
                (entry,),
                _checkpoint(corpus_id="other", projection=projection),
            ),
            (
                "corpus",
                projection,
                chunking,
                (entry,),
                _checkpoint(projection=_projection(schema_id="other")),
            ),
        )
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ContractValidationError):
                    ProjectionManifest(
                        values[0],
                        cast(ProjectionIdentity, values[1]),
                        cast(ChunkingPolicy, values[2]),
                        cast(tuple[ProjectionManifestEntry, ...], values[3]),
                        cast(ProjectionCheckpoint, values[4]),
                    )

    def test_state_availability_is_an_exact_closed_string_enum(self) -> None:
        self.assertEqual(
            tuple(
                (member.name, member.value) for member in ProjectionStateAvailability
            ),
            (("MISSING", "missing"), ("PRESENT", "present"), ("CORRUPT", "corrupt")),
        )
        for invalid in ("MISSING", "", "unknown", None, 1, object()):
            with self.subTest(value=invalid):
                with self.assertRaises(ContractValidationError):
                    ProjectionStateAvailability(cast(str, invalid))

    def test_state_snapshot_requires_manifest_only_when_present(self) -> None:
        manifest = _manifest()
        self.assertIs(
            ProjectionStateSnapshot(
                ProjectionStateAvailability.PRESENT,
                manifest,
            ).manifest,
            manifest,
        )
        for availability in (
            ProjectionStateAvailability.MISSING,
            ProjectionStateAvailability.CORRUPT,
        ):
            self.assertIsNone(ProjectionStateSnapshot(availability, None).manifest)
            with self.assertRaises(ContractValidationError):
                ProjectionStateSnapshot(availability, manifest)
        with self.assertRaises(ContractValidationError):
            ProjectionStateSnapshot(ProjectionStateAvailability.PRESENT, None)
        with self.assertRaises(ContractValidationError):
            ProjectionStateSnapshot(
                cast(ProjectionStateAvailability, "present"),
                manifest,
            )

    def test_state_status_is_an_exact_closed_string_enum(self) -> None:
        self.assertEqual(
            tuple((member.name, member.value) for member in ProjectionStateStatus),
            (
                ("MISSING", "missing"),
                ("CURRENT", "current"),
                ("STALE", "stale"),
                ("CORRUPT", "corrupt"),
                ("SCHEMA_MISMATCH", "schema_mismatch"),
                ("EMBEDDING_MISMATCH", "embedding_mismatch"),
            ),
        )
        for invalid in ("CURRENT", "", "unknown", None, 1, object()):
            with self.subTest(value=invalid):
                with self.assertRaises(ContractValidationError):
                    ProjectionStateStatus(cast(str, invalid))

    def test_result_accepts_matching_completed_and_truthful_unchanged_values(
        self,
    ) -> None:
        manifest = _manifest()
        completed = ProjectionResult(
            ProjectionStateStatus.STALE,
            ProjectionReceipt(
                "corpus",
                manifest.projection,
                ProjectionOutcome.COMPLETED,
                2,
                2,
                manifest.checkpoint,
            ),
            manifest,
        )
        unchanged = ProjectionResult(
            ProjectionStateStatus.CURRENT,
            ProjectionReceipt(
                "corpus",
                manifest.projection,
                ProjectionOutcome.UNCHANGED,
                0,
                0,
                manifest.checkpoint,
            ),
            manifest,
        )

        self.assertEqual(completed.status_before, ProjectionStateStatus.STALE)
        self.assertEqual(unchanged.receipt.attempted_documents, 0)

    def test_result_rejects_inexact_mismatched_or_untruthful_values(self) -> None:
        manifest = _manifest()
        completed = ProjectionReceipt(
            "corpus",
            manifest.projection,
            ProjectionOutcome.COMPLETED,
            1,
            1,
            manifest.checkpoint,
        )
        invalid_values = (
            ("stale", completed, manifest),
            (ProjectionStateStatus.STALE, object(), manifest),
            (ProjectionStateStatus.STALE, completed, object()),
            (
                ProjectionStateStatus.STALE,
                ProjectionReceipt(
                    "other",
                    manifest.projection,
                    ProjectionOutcome.COMPLETED,
                    1,
                    1,
                    _checkpoint(corpus_id="other", projection=manifest.projection),
                ),
                manifest,
            ),
            (
                ProjectionStateStatus.STALE,
                ProjectionReceipt(
                    "corpus",
                    manifest.projection,
                    ProjectionOutcome.PARTIAL,
                    2,
                    1,
                    None,
                ),
                manifest,
            ),
            (
                ProjectionStateStatus.STALE,
                ProjectionReceipt(
                    "corpus",
                    manifest.projection,
                    ProjectionOutcome.FAILED,
                    1,
                    0,
                    None,
                ),
                manifest,
            ),
            (
                ProjectionStateStatus.STALE,
                ProjectionReceipt(
                    "corpus",
                    _projection(schema_id="other"),
                    ProjectionOutcome.COMPLETED,
                    1,
                    1,
                    _checkpoint(projection=_projection(schema_id="other")),
                ),
                manifest,
            ),
            (
                ProjectionStateStatus.STALE,
                ProjectionReceipt(
                    "corpus",
                    manifest.projection,
                    ProjectionOutcome.COMPLETED,
                    1,
                    1,
                    _checkpoint(projection=manifest.projection, token="other"),
                ),
                manifest,
            ),
            (
                ProjectionStateStatus.STALE,
                ProjectionReceipt(
                    "corpus",
                    manifest.projection,
                    ProjectionOutcome.UNCHANGED,
                    0,
                    0,
                    manifest.checkpoint,
                ),
                manifest,
            ),
            (
                ProjectionStateStatus.CURRENT,
                ProjectionReceipt(
                    "corpus",
                    manifest.projection,
                    ProjectionOutcome.UNCHANGED,
                    1,
                    1,
                    manifest.checkpoint,
                ),
                manifest,
            ),
        )
        for status, receipt, value_manifest in invalid_values:
            with self.subTest(status=status, receipt=receipt):
                with self.assertRaises(ContractValidationError):
                    ProjectionResult(
                        cast(ProjectionStateStatus, status),
                        cast(ProjectionReceipt, receipt),
                        cast(ProjectionManifest, value_manifest),
                    )


if __name__ == "__main__":
    unittest.main()
