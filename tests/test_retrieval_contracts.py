"""Contract tests for bounded score-free retrieval values."""

from __future__ import annotations

import inspect
import unittest
from dataclasses import FrozenInstanceError, fields
from typing import cast

from generic_rag.contracts import (
    DocumentIdentity,
    DocumentKey,
    Fragment,
    FragmentIdentity,
    RetrievalHit,
    RetrievalLimits,
    RetrievalOutcome,
    RetrievalQuery,
    RetrievalResult,
)
from generic_rag.errors import ContractValidationError


class _StringSubclass(str):
    pass


class _IntegerSubclass(int):
    pass


class _TupleSubclass(tuple[object, ...]):
    pass


def _fragment(
    fragment_id: str,
    *,
    corpus_id: str = "corpus",
    document_id: str = "document",
    revision_id: str = "revision",
    start: int = 0,
    attributes: tuple[tuple[str, str], ...] = (),
) -> Fragment:
    document = DocumentIdentity(
        DocumentKey(corpus_id, document_id),
        revision_id,
    )
    identity = FragmentIdentity(
        document=document,
        fragment_id=fragment_id,
        start=start,
        end=start + 1,
    )
    return Fragment(identity, "x", attributes)


def _query(
    *,
    corpus_id: str = "corpus",
    text: str = "query",
    hit_limit: int = 3,
    candidate_limit: int = 5,
) -> RetrievalQuery:
    return RetrievalQuery(
        corpus_id=corpus_id,
        text=text,
        hit_limit=hit_limit,
        candidate_limit=candidate_limit,
    )


class RetrievalContractTests(unittest.TestCase):
    def test_retrieval_fields_are_exact_frozen_and_slotted(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(RetrievalLimits)),
            ("max_query_codepoints",),
        )
        self.assertEqual(
            tuple(field.name for field in fields(RetrievalQuery)),
            ("corpus_id", "text", "hit_limit", "candidate_limit"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(RetrievalHit)),
            ("fragment", "rank"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(RetrievalResult)),
            ("query", "outcome", "hits", "truncated"),
        )

        limits = RetrievalLimits(100)
        query = _query()
        hit = RetrievalHit(_fragment("fragment"), 1)
        result = RetrievalResult(
            query,
            RetrievalOutcome.COMPLETE,
            (hit,),
            False,
        )
        for instance, field_name in (
            (limits, "max_query_codepoints"),
            (query, "text"),
            (hit, "rank"),
            (result, "truncated"),
        ):
            with self.subTest(instance=type(instance).__name__):
                self.assertFalse(hasattr(instance, "__dict__"))
                with self.assertRaises(FrozenInstanceError):
                    setattr(instance, field_name, object())
                self.assertIsInstance(hash(instance), int)

    def test_retrieval_limits_are_independent_positive_exact_integers(self) -> None:
        limits = RetrievalLimits(max_query_codepoints=7)

        self.assertEqual(limits.max_query_codepoints, 7)
        self.assertNotIn("hit_limit", inspect.signature(RetrievalLimits).parameters)
        self.assertNotIn(
            "candidate_limit",
            inspect.signature(RetrievalLimits).parameters,
        )
        for invalid in (0, -1, True, 1.0, _IntegerSubclass(1)):
            with self.subTest(value=invalid):
                with self.assertRaises(ContractValidationError):
                    RetrievalLimits(cast(int, invalid))

    def test_retrieval_outcomes_are_exact_closed_string_enums(self) -> None:
        self.assertEqual(
            tuple((member.name, member.value) for member in RetrievalOutcome),
            (
                ("COMPLETE", "complete"),
                ("PARTIAL", "partial"),
                ("UNAVAILABLE", "unavailable"),
                ("STALE", "stale"),
                ("FAILED", "failed"),
            ),
        )
        self.assertEqual(RetrievalOutcome("complete"), RetrievalOutcome.COMPLETE)
        self.assertIsInstance(RetrievalOutcome.COMPLETE, str)

        for invalid in ("COMPLETE", "", "unknown", None, 1, object()):
            with self.subTest(value=invalid):
                with self.assertRaises(ContractValidationError):
                    RetrievalOutcome(cast(str, invalid))

    def test_query_preserves_opaque_corpus_and_nonblank_text(self) -> None:
        corpus_id = " Corpus/../e\u0301 "
        text = "  Query/../A  "
        query = _query(corpus_id=corpus_id, text=text)

        self.assertEqual(query.corpus_id, corpus_id)
        self.assertEqual(query.text, text)

        for invalid in ("", " \t\n", None, b"value", _StringSubclass("value")):
            with self.subTest(field="corpus_id", value=invalid):
                with self.assertRaises(ContractValidationError):
                    _query(corpus_id=cast(str, invalid))
            with self.subTest(field="text", value=invalid):
                with self.assertRaises(ContractValidationError):
                    _query(text=cast(str, invalid))

    def test_query_limits_are_positive_exact_integers_and_bounded(self) -> None:
        query = _query(hit_limit=2, candidate_limit=2)
        self.assertEqual((query.hit_limit, query.candidate_limit), (2, 2))

        for invalid in (0, -1, True, 1.0, _IntegerSubclass(1)):
            with self.subTest(field="hit_limit", value=invalid):
                with self.assertRaises(ContractValidationError):
                    _query(hit_limit=cast(int, invalid))
            with self.subTest(field="candidate_limit", value=invalid):
                with self.assertRaises(ContractValidationError):
                    _query(candidate_limit=cast(int, invalid))

        with self.assertRaises(ContractValidationError):
            _query(hit_limit=3, candidate_limit=2)

    def test_hit_is_positive_ranked_exact_fragment_without_score(self) -> None:
        fragment = _fragment("fragment")
        hit = RetrievalHit(fragment, 1)

        self.assertIs(hit.fragment, fragment)
        self.assertEqual(hit.rank, 1)
        self.assertNotIn("score", inspect.signature(RetrievalHit).parameters)
        self.assertFalse(hasattr(hit, "score"))

        with self.assertRaises(ContractValidationError):
            RetrievalHit(cast(Fragment, object()), 1)
        for invalid in (0, -1, True, 1.0, _IntegerSubclass(1)):
            with self.subTest(rank=invalid):
                with self.assertRaises(ContractValidationError):
                    RetrievalHit(fragment, cast(int, invalid))

    def test_result_accepts_contiguous_unique_bounded_matching_hits(self) -> None:
        query = _query(hit_limit=3, candidate_limit=5)
        hits = (
            RetrievalHit(_fragment("first"), 1),
            RetrievalHit(_fragment("second", document_id="other"), 2),
        )
        result = RetrievalResult(
            query,
            RetrievalOutcome.COMPLETE,
            hits,
            False,
        )

        self.assertEqual(result.hits, hits)
        self.assertFalse(result.truncated)

        explicitly_truncated = RetrievalResult(
            query,
            RetrievalOutcome.COMPLETE,
            hits,
            True,
        )
        self.assertTrue(explicitly_truncated.truncated)

    def test_result_rejects_noncontiguous_or_duplicate_hits(self) -> None:
        query = _query()
        first = _fragment("first")
        second = _fragment("second")

        invalid_rank_sequences = (
            (RetrievalHit(first, 2),),
            (RetrievalHit(first, 1), RetrievalHit(second, 3)),
            (RetrievalHit(first, 1), RetrievalHit(second, 1)),
        )
        for hits in invalid_rank_sequences:
            with self.subTest(ranks=tuple(hit.rank for hit in hits)):
                with self.assertRaises(ContractValidationError):
                    RetrievalResult(
                        query,
                        RetrievalOutcome.COMPLETE,
                        hits,
                        False,
                    )

        duplicate_identity = _fragment(
            "duplicate",
            attributes=(("variant", "one"),),
        )
        duplicate_identity_other_value = _fragment(
            "duplicate",
            attributes=(("variant", "two"),),
        )
        with self.assertRaises(ContractValidationError):
            RetrievalResult(
                query,
                RetrievalOutcome.COMPLETE,
                (
                    RetrievalHit(duplicate_identity, 1),
                    RetrievalHit(duplicate_identity_other_value, 2),
                ),
                False,
            )

    def test_result_rejects_wrong_corpus_and_excess_hits(self) -> None:
        with self.assertRaises(ContractValidationError):
            RetrievalResult(
                _query(corpus_id="corpus"),
                RetrievalOutcome.COMPLETE,
                (RetrievalHit(_fragment("hit", corpus_id="other"), 1),),
                False,
            )

        with self.assertRaises(ContractValidationError):
            RetrievalResult(
                _query(hit_limit=1),
                RetrievalOutcome.COMPLETE,
                (
                    RetrievalHit(_fragment("first"), 1),
                    RetrievalHit(_fragment("second"), 2),
                ),
                False,
            )

    def test_terminal_outcomes_cannot_claim_hits_or_truncation(self) -> None:
        query = _query()
        hit = RetrievalHit(_fragment("fragment"), 1)

        for outcome in (
            RetrievalOutcome.UNAVAILABLE,
            RetrievalOutcome.STALE,
            RetrievalOutcome.FAILED,
        ):
            with self.subTest(outcome=outcome):
                result = RetrievalResult(query, outcome, (), False)
                self.assertEqual(result.hits, ())
                self.assertFalse(result.truncated)

                with self.assertRaises(ContractValidationError):
                    RetrievalResult(query, outcome, (hit,), False)
                with self.assertRaises(ContractValidationError):
                    RetrievalResult(query, outcome, (), True)

    def test_partial_results_require_at_least_one_valid_hit(self) -> None:
        query = _query()

        with self.assertRaises(ContractValidationError):
            RetrievalResult(query, RetrievalOutcome.PARTIAL, (), False)

        hit = RetrievalHit(_fragment("fragment"), 1)
        for truncated in (False, True):
            with self.subTest(truncated=truncated):
                result = RetrievalResult(
                    query,
                    RetrievalOutcome.PARTIAL,
                    (hit,),
                    truncated,
                )
                self.assertEqual(result.hits, (hit,))

    def test_result_requires_exact_nested_container_and_bool_types(self) -> None:
        query = _query()
        hit = RetrievalHit(_fragment("fragment"), 1)

        with self.assertRaises(ContractValidationError):
            RetrievalResult(
                cast(RetrievalQuery, object()),
                RetrievalOutcome.COMPLETE,
                (),
                False,
            )
        with self.assertRaises(ContractValidationError):
            RetrievalResult(
                query,
                cast(RetrievalOutcome, "complete"),
                (),
                False,
            )
        for invalid_hits in (
            [hit],
            iter((hit,)),
            _TupleSubclass((hit,)),
            (object(),),
        ):
            with self.subTest(hits=type(invalid_hits).__name__):
                with self.assertRaises(ContractValidationError):
                    RetrievalResult(
                        query,
                        RetrievalOutcome.COMPLETE,
                        cast(tuple[RetrievalHit, ...], invalid_hits),
                        False,
                    )
        for invalid_truncated in (0, 1, None, "false"):
            with self.subTest(truncated=invalid_truncated):
                with self.assertRaises(ContractValidationError):
                    RetrievalResult(
                        query,
                        RetrievalOutcome.COMPLETE,
                        (),
                        cast(bool, invalid_truncated),
                    )


if __name__ == "__main__":
    unittest.main()
