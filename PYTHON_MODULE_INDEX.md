# Python Module Index

## Inventory

- Declared source root: `src`
- Packaging source of truth: `pyproject.toml`
- Importable production units: 4
- Indexed production units: 4
- Source/index parity: 4/4
- Package data: `src/generic_rag/py.typed`
- Locked verification owner: `.github/workflows/ci.yml` (supporting workflow,
  not an importable unit)

## `generic_rag`

- Source: `src/generic_rag/__init__.py`
- Responsibility: establish the side-effect-free package import namespace.
- Supported public imports: `import generic_rag`.
- Re-exports: none; `__all__ = ()` is intentional until a meaningful high-level
  facade is implemented.
- Direct internal dependencies: none.
- Owned state or external resources: none.
- Material side effects: none.
- Verification: `tests/test_package_boundaries.py`,
  `tests/support/clean_import_probe.py`, `tests/support/verify_artifacts.py`, and
  the locked CI build, clean-install, import, compile, and artifact checks.
- Documentation: `README.md` and `docs/api.md`.
- Package data: declares and ships `py.typed`.

## `generic_rag.errors`

- Source: `src/generic_rag/errors.py`
- Responsibility: define the typed public failure categories exposed by generic
  RAG contracts and workflows.
- Supported public imports: `GenericRagError`, `ContractValidationError`,
  `CollaborationError`, and `StateCompatibilityError` from
  `generic_rag.errors`.
- Re-exports: exactly the names in the module's `__all__`; none from the package
  root.
- Direct internal dependencies: none.
- Owned state or external resources: none.
- Material side effects: none.
- Verification: `tests/test_errors.py`, `tests/test_package_boundaries.py`,
  `tests/support/clean_import_probe.py`, and the locked CI import and boundary
  checks.
- Documentation: `docs/api.md`.

## `generic_rag.contracts`

- Source: `src/generic_rag/contracts.py`
- Responsibility: define immutable validated values shared by generic RAG
  workflows and provider ports.
- Supported public imports: `DocumentKey`, `DocumentIdentity`, `Document`,
  `FragmentIdentity`, `Fragment`, `EmbeddingIdentity`, `EmbeddingVector`,
  `VectorRecord`, `ProjectionIdentity`, `ProjectionCheckpoint`,
  `ProjectionOutcome`, `ProjectionReceipt`, `RetrievalQuery`,
  `RetrievalOutcome`, `RetrievalHit`, and `RetrievalResult` from
  `generic_rag.contracts`.
- Re-exports: exactly the names in the module's `__all__`; none from the package
  root.
- Direct internal dependencies: `generic_rag.errors`.
- Owned state or external resources: none; instances own only immutable caller
  values.
- Material side effects: none.
- Verification: `tests/test_contract_values.py`,
  `tests/test_projection_contracts.py`, `tests/test_retrieval_contracts.py`,
  `tests/test_package_boundaries.py`, `tests/support/clean_import_probe.py`,
  `tests/support/verify_artifacts.py`, and the locked CI import, boundary, and
  artifact checks.
- Documentation: `docs/api.md` and `docs/security-and-privacy.md`.

## `generic_rag.ports`

- Source: `src/generic_rag/ports.py`
- Responsibility: define synchronous injected collaborator interfaces and
  explicit caller-owned borrowing semantics.
- Supported public imports: `Borrowed`, `Embedder`, `VectorIndexWriter`,
  `VectorIndexReader`, and `LexicalRetriever` from `generic_rag.ports`.
- Re-exports: exactly the names in the module's `__all__`; none from the package
  root.
- Direct internal dependencies: `generic_rag.contracts`.
- Owned state or external resources: `Borrowed` retains a reference but never
  owns, acquires, releases, closes, or shuts down the resource.
- Material side effects: none.
- Verification: `tests/test_ports.py`, `tests/test_package_boundaries.py`,
  `tests/support/clean_import_probe.py`, and the locked CI import and boundary
  checks.
- Documentation: `docs/api.md`, `docs/lifecycle.md`, and
  `docs/security-and-privacy.md`.
