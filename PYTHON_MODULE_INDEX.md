# Python Module Index

## Inventory

- Declared source root: `src`
- Packaging source of truth: `pyproject.toml`
- Importable production units: 7
- Indexed production units: 7
- Source/index parity: 7/7
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
  `tests/test_projection.py`, `tests/support/clean_import_probe.py`, and the
  locked CI import and boundary checks.
- Documentation: `docs/api.md` and `docs/projection.md`.

## `generic_rag.contracts`

- Source: `src/generic_rag/contracts.py`
- Responsibility: define immutable validated values shared by generic RAG
  workflows and provider ports.
- Supported public imports: `DocumentKey`, `DocumentIdentity`, `Document`,
  `FragmentIdentity`, `Fragment`, `EmbeddingIdentity`, `EmbeddingVector`,
  `VectorRecord`, `ProjectionIdentity`, `ProjectionCheckpoint`,
  `ProjectionOutcome`, `ProjectionReceipt`, `ChunkingPolicy`,
  `ProjectionLimits`, `ProjectionRequest`, `ProjectionManifestEntry`,
  `ProjectionManifest`, `ProjectionStateAvailability`,
  `ProjectionStateSnapshot`, `ProjectionStateStatus`, `ProjectionResult`,
  `RetrievalLimits`, `RetrievalQuery`, `RetrievalOutcome`, `RetrievalHit`, and
  `RetrievalResult` from `generic_rag.contracts`.
- Re-exports: exactly the names in the module's `__all__`; none from the package
  root.
- Direct internal dependencies: `generic_rag.errors`.
- Owned state or external resources: none; instances own only immutable caller
  values.
- Material side effects: none.
- Verification: `tests/test_contract_values.py`,
  `tests/test_projection_contracts.py`, `tests/test_projection.py`,
  `tests/test_retrieval_contracts.py`, `tests/test_package_boundaries.py`,
  `tests/support/clean_import_probe.py`, `tests/support/verify_artifacts.py`, and
  the locked CI import, boundary, and artifact checks.
- Documentation: `docs/api.md`, `docs/projection.md`, `docs/retrieval.md`, and
  `docs/security-and-privacy.md`.

## `generic_rag.ports`

- Source: `src/generic_rag/ports.py`
- Responsibility: define synchronous injected collaborator interfaces and
  explicit caller-owned borrowing semantics.
- Supported public imports: `Borrowed`, `Embedder`, `VectorIndexWriter`,
  `VectorIndexResetter`, `VectorIndexReader`, and `LexicalRetriever` from
  `generic_rag.ports`.
- Re-exports: exactly the names in the module's `__all__`; none from the package
  root.
- Direct internal dependencies: `generic_rag.contracts`.
- Owned state or external resources: `Borrowed` retains a reference but never
  owns, acquires, releases, closes, or shuts down the resource.
- Material side effects: none.
- Verification: `tests/test_ports.py`, `tests/test_package_boundaries.py`,
  `tests/test_projection.py`, `tests/support/clean_import_probe.py`, and the
  locked CI import and boundary checks.
- Documentation: `docs/api.md`, `docs/lifecycle.md`, `docs/projection.md`,
  `docs/retrieval.md`, and `docs/security-and-privacy.md`.

## `generic_rag.projection`

- Source: `src/generic_rag/projection.py`
- Responsibility: deterministically plan and synchronously execute bounded,
  revision-aware document projection against caller-supplied state.
- Supported public imports: `ProjectionFailureStage`, `ProjectionStateError`,
  `ProjectionOperationError`, `project_documents`, and `rebuild_projection`
  from `generic_rag.projection`.
- Re-exports: exactly the names in the module's `__all__`; none from the package
  root.
- Direct internal dependencies: `generic_rag.contracts`, `generic_rag.errors`,
  `generic_rag.ports`, and `generic_rag.projection_integrity`.
- Integrity ownership: delegates deterministic source-digest, fragment-ID, and
  checkpoint-token derivation and checkpoint validation to
  `generic_rag.projection_integrity`.
- Owned state or external resources: none; planning state is immutable and
  local to each call, while every embedder, writer, and resetter remains
  caller-owned through `Borrowed`.
- Material side effects: none at import time. At explicit workflow call time it
  may invoke the borrowed embedder and vector writer, and full rebuild may
  invoke the borrowed corpus resetter; it performs no persistence, network,
  retry, acquisition, release, or lifecycle action itself.
- Verification: `tests/test_projection.py`, `tests/test_package_boundaries.py`,
  `tests/support/clean_import_probe.py`, `tests/support/verify_artifacts.py`, and
  the locked CI test, lint, type, build, clean-install, import, and artifact
  checks.
- Documentation: `docs/projection.md`, `docs/api.md`, `docs/lifecycle.md`, and
  `docs/security-and-privacy.md`.

## `generic_rag.projection_integrity`

- Source: `src/generic_rag/projection_integrity.py`
- Responsibility: own deterministic v1 derivation and validation algorithms
  for projection integrity values.
- Supported public imports: `derive_source_digest`, `derive_fragment_id`,
  `derive_projection_checkpoint_token`, and
  `has_valid_projection_checkpoint` from `generic_rag.projection_integrity`.
- Re-exports: exactly the names in the module's `__all__`; none from the package
  root.
- Direct internal dependencies: `generic_rag.contracts` and
  `generic_rag.errors`.
- Owned state or external resources: none; each call derives or checks an
  immutable value from exact caller-supplied contracts.
- Material side effects: none at import or call time.
- Verification: `tests/test_projection_integrity.py`,
  `tests/test_projection.py`, `tests/test_retrieval.py`,
  `tests/test_package_boundaries.py`, `tests/support/clean_import_probe.py`,
  `tests/support/verify_artifacts.py`, and the locked CI test, lint, strict
  type, compile, build, clean-install, import, and artifact checks.
- Documentation: `docs/api.md`, `docs/projection.md`, `docs/retrieval.md`, and
  `docs/security-and-privacy.md`.

## `generic_rag.retrieval`

- Source: `src/generic_rag/retrieval.py`
- Responsibility: deterministically compose bounded semantic and hybrid
  retrieval results from published caller-owned projection state.
- Supported public imports: `retrieve_semantic` and `retrieve_hybrid` from
  `generic_rag.retrieval`.
- Re-exports: exactly the names in the module's `__all__`; none from the package
  root.
- Direct internal dependencies: `generic_rag.contracts`, `generic_rag.errors`,
  `generic_rag.ports`, and `generic_rag.projection_integrity`.
- Owned state or external resources: none; ranking state is local to each call,
  while every embedder, vector reader, and lexical retriever remains
  caller-owned through `Borrowed`.
- Material side effects: none at import time. At explicit workflow call time it
  may read one borrowed embedder identity, embed one query, search one borrowed
  vector reader, and for hybrid retrieval search one borrowed lexical
  retriever; it performs no persistence, network, retry, acquisition, release,
  logging, authorization, citation, or lifecycle action itself.
- Retrieval semantics: validates and deduplicates exact fragment identities,
  filters candidates against the full current published document revision,
  and preserves semantic provider order or fuses original semantic and lexical
  ranks deterministically without comparing raw scores. Returned fragments
  remain non-authoritative; this module does not authorize a query or source,
  validate authoritative source text, or create citations.
- Verification: `tests/test_retrieval.py`,
  `tests/test_retrieval_contracts.py`, `tests/test_ports.py`,
  `tests/test_package_boundaries.py`, `tests/support/clean_import_probe.py`,
  `tests/support/verify_artifacts.py`, and the locked CI test, lint, strict
  type, compile, build, clean-install, import, and artifact checks.
- Documentation: `docs/retrieval.md`, `docs/api.md`, `docs/lifecycle.md`,
  `docs/security-and-privacy.md`, and `README.md`.
