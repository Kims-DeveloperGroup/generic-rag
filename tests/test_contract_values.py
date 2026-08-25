"""Contract tests for immutable document, fragment, and embedding values."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields
from typing import cast

from generic_rag.contracts import (
    Document,
    DocumentIdentity,
    DocumentKey,
    EmbeddingIdentity,
    EmbeddingVector,
    Fragment,
    FragmentIdentity,
    VectorRecord,
)
from generic_rag.errors import ContractValidationError


class _StringSubclass(str):
    pass


class _IntegerSubclass(int):
    pass


class _FloatSubclass(float):
    pass


class _TupleSubclass(tuple[object, ...]):
    pass


def _document_identity(
    *,
    corpus_id: str = "corpus",
    document_id: str = "document",
    revision_id: str = "revision",
) -> DocumentIdentity:
    return DocumentIdentity(
        key=DocumentKey(corpus_id=corpus_id, document_id=document_id),
        revision_id=revision_id,
    )


def _fragment_identity(
    *,
    corpus_id: str = "corpus",
    document_id: str = "document",
    revision_id: str = "revision",
    fragment_id: str = "fragment",
    start: int = 0,
    end: int = 2,
) -> FragmentIdentity:
    return FragmentIdentity(
        document=_document_identity(
            corpus_id=corpus_id,
            document_id=document_id,
            revision_id=revision_id,
        ),
        fragment_id=fragment_id,
        start=start,
        end=end,
    )


class ContractValueTests(unittest.TestCase):
    def assert_frozen_and_slotted(
        self,
        instance: object,
        field_name: str,
    ) -> None:
        self.assertFalse(hasattr(instance, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            setattr(instance, field_name, object())

    def test_exact_field_layouts(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(DocumentKey)),
            ("corpus_id", "document_id"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(DocumentIdentity)),
            ("key", "revision_id"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Document)),
            ("identity", "text", "attributes"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(FragmentIdentity)),
            ("document", "fragment_id", "start", "end"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Fragment)),
            ("identity", "text", "attributes"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(EmbeddingIdentity)),
            ("model_id", "dimensions"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(EmbeddingVector)),
            ("values",),
        )
        self.assertEqual(
            tuple(field.name for field in fields(VectorRecord)),
            ("fragment", "embedding"),
        )

    def test_values_are_frozen_slotted_hashable_value_objects(self) -> None:
        document_key = DocumentKey("corpus", "document")
        document_identity = DocumentIdentity(document_key, "revision")
        document = Document(document_identity, "")
        fragment_identity = FragmentIdentity(
            document_identity,
            "fragment",
            0,
            1,
        )
        fragment = Fragment(fragment_identity, "x")
        embedding_identity = EmbeddingIdentity("model", 1)
        embedding = EmbeddingVector((1,))
        record = VectorRecord(fragment, embedding)
        cases = (
            (document_key, "corpus_id"),
            (document_identity, "revision_id"),
            (document, "text"),
            (fragment_identity, "start"),
            (fragment, "text"),
            (embedding_identity, "dimensions"),
            (embedding, "values"),
            (record, "fragment"),
        )

        for instance, field_name in cases:
            with self.subTest(instance=type(instance).__name__):
                self.assert_frozen_and_slotted(instance, field_name)
                self.assertIsInstance(hash(instance), int)
                self.assertEqual(instance, instance)

    def test_opaque_identity_strings_are_preserved_exactly(self) -> None:
        corpus_id = "  Corpus/../e\u0301  "
        document_id = " Document:ABC "
        revision_id = " Rev/001 "
        fragment_id = " Fragment/../001 "
        key = DocumentKey(corpus_id, document_id)
        document = DocumentIdentity(key, revision_id)
        fragment = FragmentIdentity(document, fragment_id, 4, 5)

        self.assertEqual(key.corpus_id, corpus_id)
        self.assertEqual(key.document_id, document_id)
        self.assertEqual(document.revision_id, revision_id)
        self.assertEqual(fragment.fragment_id, fragment_id)

    def test_identity_strings_reject_empty_whitespace_and_inexact_types(self) -> None:
        invalid_values: tuple[object, ...] = (
            "",
            " \t\n",
            None,
            b"value",
            1,
            _StringSubclass("value"),
        )

        for invalid in invalid_values:
            with self.subTest(field="corpus_id", value=invalid):
                with self.assertRaises(ContractValidationError):
                    DocumentKey(cast(str, invalid), "document")
            with self.subTest(field="document_id", value=invalid):
                with self.assertRaises(ContractValidationError):
                    DocumentKey("corpus", cast(str, invalid))
            with self.subTest(field="revision_id", value=invalid):
                with self.assertRaises(ContractValidationError):
                    DocumentIdentity(
                        DocumentKey("corpus", "document"),
                        cast(str, invalid),
                    )
            with self.subTest(field="fragment_id", value=invalid):
                with self.assertRaises(ContractValidationError):
                    FragmentIdentity(
                        _document_identity(),
                        cast(str, invalid),
                        0,
                        1,
                    )
            with self.subTest(field="model_id", value=invalid):
                with self.assertRaises(ContractValidationError):
                    EmbeddingIdentity(cast(str, invalid), 1)

    def test_nested_value_fields_require_the_exact_declared_classes(self) -> None:
        document_identity = _document_identity()
        fragment_identity = _fragment_identity()
        fragment = Fragment(fragment_identity, "ab")
        embedding = EmbeddingVector((1.0,))

        with self.assertRaises(ContractValidationError):
            DocumentIdentity(cast(DocumentKey, object()), "revision")
        with self.assertRaises(ContractValidationError):
            Document(cast(DocumentIdentity, object()), "text")
        with self.assertRaises(ContractValidationError):
            FragmentIdentity(
                cast(DocumentIdentity, object()),
                "fragment",
                0,
                1,
            )
        with self.assertRaises(ContractValidationError):
            Fragment(cast(FragmentIdentity, object()), "x")
        with self.assertRaises(ContractValidationError):
            VectorRecord(cast(Fragment, object()), embedding)
        with self.assertRaises(ContractValidationError):
            VectorRecord(fragment, cast(EmbeddingVector, object()))

        class DocumentKeySubclass(DocumentKey):
            pass

        subclass_key = DocumentKeySubclass("corpus", "document")
        with self.assertRaises(ContractValidationError):
            DocumentIdentity(subclass_key, "revision")
        self.assertEqual(document_identity.key.corpus_id, "corpus")

    def test_document_text_is_exact_and_may_be_empty(self) -> None:
        identity = _document_identity()

        self.assertEqual(Document(identity, "").text, "")
        text = " \u0000 e\u0301 \n"
        self.assertEqual(Document(identity, text).text, text)
        for invalid in (None, b"text", _StringSubclass("text")):
            with self.subTest(value=invalid):
                with self.assertRaises(ContractValidationError):
                    Document(identity, cast(str, invalid))

    def test_attributes_preserve_empty_values_duplicates_and_order(self) -> None:
        attributes = (
            ("", ""),
            ("key", "first"),
            ("key", "first"),
            ("key", "second"),
        )
        document = Document(_document_identity(), "text", attributes)
        fragment = Fragment(_fragment_identity(), "ab", attributes)

        self.assertEqual(document.attributes, attributes)
        self.assertEqual(fragment.attributes, attributes)

    def test_attributes_reject_inexact_or_malformed_containers(self) -> None:
        invalid_attributes: tuple[object, ...] = (
            [("key", "value")],
            {"key": "value"},
            iter((("key", "value"),)),
            _TupleSubclass((("key", "value"),)),
            (["key", "value"],),
            (("key",),),
            (("key", "value", "extra"),),
            ((1, "value"),),
            (("key", 1),),
            ((_StringSubclass("key"), "value"),),
            (("key", _StringSubclass("value")),),
        )

        for invalid in invalid_attributes:
            with self.subTest(value=invalid):
                with self.assertRaises(ContractValidationError):
                    Document(
                        _document_identity(),
                        "text",
                        cast(tuple[tuple[str, str], ...], invalid),
                    )
                with self.assertRaises(ContractValidationError):
                    Fragment(
                        _fragment_identity(),
                        "ab",
                        cast(tuple[tuple[str, str], ...], invalid),
                    )

    def test_fragment_ranges_use_exact_half_open_code_point_offsets(self) -> None:
        emoji_identity = _fragment_identity(start=10, end=11)
        combining_identity = _fragment_identity(start=20, end=22)

        self.assertEqual(Fragment(emoji_identity, "\U0001f600").text, "\U0001f600")
        combining = "e\u0301"
        self.assertEqual(len(combining), 2)
        self.assertEqual(Fragment(combining_identity, combining).text, combining)

        invalid_ranges = (
            (-1, 1),
            (0, 0),
            (2, 1),
        )
        for start, end in invalid_ranges:
            with self.subTest(start=start, end=end):
                with self.assertRaises(ContractValidationError):
                    _fragment_identity(start=start, end=end)

        for invalid in (True, 1.0, _IntegerSubclass(1)):
            with self.subTest(field="start", value=invalid):
                with self.assertRaises(ContractValidationError):
                    _fragment_identity(start=cast(int, invalid), end=2)
            with self.subTest(field="end", value=invalid):
                with self.assertRaises(ContractValidationError):
                    _fragment_identity(start=0, end=cast(int, invalid))

    def test_fragment_text_is_nonempty_and_matches_the_range_length(self) -> None:
        identity = _fragment_identity(start=4, end=6)

        self.assertEqual(Fragment(identity, "xy").text, "xy")
        for invalid in ("", "x", "xyz"):
            with self.subTest(value=invalid):
                with self.assertRaises(ContractValidationError):
                    Fragment(identity, invalid)
        for invalid_text_type in (None, b"xy", _StringSubclass("xy")):
            with self.subTest(value=invalid_text_type):
                with self.assertRaises(ContractValidationError):
                    Fragment(identity, cast(str, invalid_text_type))

    def test_embedding_dimensions_are_positive_exact_integers(self) -> None:
        self.assertEqual(EmbeddingIdentity("model", 3).dimensions, 3)

        for invalid in (0, -1, True, 1.0, _IntegerSubclass(1)):
            with self.subTest(value=invalid):
                with self.assertRaises(ContractValidationError):
                    EmbeddingIdentity("model", cast(int, invalid))

    def test_embedding_vectors_are_nonempty_exact_tuples_of_finite_numbers(
        self,
    ) -> None:
        vector = EmbeddingVector((1, -0.0, 2.5))

        self.assertEqual(vector.values, (1.0, -0.0, 2.5))
        self.assertTrue(all(type(value) is float for value in vector.values))

        invalid_containers: tuple[object, ...] = (
            [],
            iter((1.0,)),
            _TupleSubclass((1.0,)),
            (),
        )
        for invalid in invalid_containers:
            with self.subTest(value=invalid):
                with self.assertRaises(ContractValidationError):
                    EmbeddingVector(cast(tuple[float, ...], invalid))

        invalid_coordinates: tuple[object, ...] = (
            True,
            None,
            "1",
            _IntegerSubclass(1),
            _FloatSubclass(1.0),
            float("nan"),
            float("inf"),
            float("-inf"),
            10**10000,
        )
        for invalid in invalid_coordinates:
            with self.subTest(value=type(invalid).__name__):
                with self.assertRaises(ContractValidationError):
                    EmbeddingVector(cast(tuple[float, ...], (cast(float, invalid),)))


if __name__ == "__main__":
    unittest.main()
