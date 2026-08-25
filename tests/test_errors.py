"""Contract tests for the public generic RAG exception hierarchy."""

from __future__ import annotations

import unittest

import generic_rag.errors as errors
from generic_rag.errors import (
    CollaborationError,
    ContractValidationError,
    GenericRagError,
    StateCompatibilityError,
)


class ErrorContractTests(unittest.TestCase):
    def test_exports_are_exact_and_owned_by_the_module(self) -> None:
        expected = (
            "GenericRagError",
            "ContractValidationError",
            "CollaborationError",
            "StateCompatibilityError",
        )

        self.assertEqual(errors.__all__, expected)
        for name in expected:
            exported = getattr(errors, name)
            self.assertEqual(exported.__module__, "generic_rag.errors")

    def test_specialized_errors_share_the_public_base(self) -> None:
        specialized = (
            ContractValidationError,
            CollaborationError,
            StateCompatibilityError,
        )

        for error_type in specialized:
            with self.subTest(error_type=error_type.__name__):
                self.assertTrue(issubclass(error_type, GenericRagError))
                self.assertIsNot(error_type, GenericRagError)

    def test_specialized_errors_remain_distinct_categories(self) -> None:
        error_types = {
            ContractValidationError,
            CollaborationError,
            StateCompatibilityError,
        }

        self.assertEqual(len(error_types), 3)


if __name__ == "__main__":
    unittest.main()
