"""Contract tests for projection identities, checkpoints, and receipts."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields
from typing import cast

from generic_rag.contracts import (
    EmbeddingIdentity,
    ProjectionCheckpoint,
    ProjectionIdentity,
    ProjectionOutcome,
    ProjectionReceipt,
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


if __name__ == "__main__":
    unittest.main()
