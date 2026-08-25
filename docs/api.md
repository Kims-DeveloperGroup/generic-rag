# Public API

Version 0.1.0 exposes immutable values, typed error categories, synchronous
collaborator protocols, deterministic bounded document projection, and
score-free semantic and hybrid retrieval. See the [projection guide](projection.md),
[retrieval guide](retrieval.md), [resource lifecycle](lifecycle.md), and
[security and privacy boundary](security-and-privacy.md) for the surrounding
usage contract.

## Import boundary

The supported package-root operation is `import generic_rag`. The root has an
intentionally empty export list, `generic_rag.__all__ == ()`, and re-exports no
public symbols. Import names from their owning modules instead.

`generic_rag.errors` exports exactly:

- `GenericRagError`
- `ContractValidationError`
- `CollaborationError`
- `StateCompatibilityError`

`generic_rag.contracts` exports exactly:

- `DocumentKey`
- `DocumentIdentity`
- `Document`
- `FragmentIdentity`
- `Fragment`
- `EmbeddingIdentity`
- `EmbeddingVector`
- `VectorRecord`
- `ProjectionIdentity`
- `ProjectionCheckpoint`
- `ProjectionOutcome`
- `ProjectionReceipt`
- `ChunkingPolicy`
- `ProjectionLimits`
- `ProjectionRequest`
- `ProjectionManifestEntry`
- `ProjectionManifest`
- `ProjectionStateAvailability`
- `ProjectionStateSnapshot`
- `ProjectionStateStatus`
- `ProjectionResult`
- `RetrievalLimits`
- `RetrievalQuery`
- `RetrievalOutcome`
- `RetrievalHit`
- `RetrievalResult`

`generic_rag.ports` exports exactly:

- `Borrowed`
- `Embedder`
- `VectorIndexWriter`
- `VectorIndexResetter`
- `VectorIndexReader`
- `LexicalRetriever`

`generic_rag.projection` exports exactly:

- `ProjectionFailureStage`
- `ProjectionStateError`
- `ProjectionOperationError`
- `project_documents`
- `rebuild_projection`

`generic_rag.projection_integrity` exports exactly:

- `derive_source_digest`
- `derive_fragment_id`
- `derive_projection_checkpoint_token`
- `has_valid_projection_checkpoint`

`generic_rag.retrieval` exports exactly:

- `retrieve_semantic`
- `retrieve_hybrid`

The package does not support importing any of these names from the package
root.

## Shared value rules

Public contract dataclasses are frozen, slotted, hashable value objects. They
require exact concrete field types, including exact nested contract classes;
subclasses do not satisfy these runtime validators. In particular:

- strings must be exact `str` values;
- discrete integers must be exact `int` values, so `bool`, floats, and integer
  subclasses are rejected;
- tuple fields must be exact tuples, not lists, iterators, or tuple subclasses;
  and
- field invariant violations detected during construction raise
  `ContractValidationError`.

Opaque identity strings must be nonempty and not whitespace-only. They are
otherwise stored exactly as supplied: the package does not trim, normalize,
canonicalize, parse, or resolve them.

Attributes use `tuple[tuple[str, str], ...]`. Every pair must be an exact
two-element tuple containing exact strings. Empty keys and values are allowed,
and pair order and duplicates are preserved.

## Errors

The base error relationships used by projection are:

```text
GenericRagError
├── ContractValidationError
├── CollaborationError
│   └── ProjectionOperationError
└── StateCompatibilityError
    └── ProjectionStateError
```

- `ContractValidationError` reports a violated public value invariant or an
  invalid top-level workflow input.
- `ProjectionStateError(status: ProjectionStateStatus)` reports state that an
  incremental operation cannot use. Its exact `status` field gives the reason;
  its message contains no state identifier.
- `ProjectionOperationError(stage, affected_document, receipt)` translates an
  ordinary collaborator failure or invalid collaborator return. Its message is
  content-free. `affected_document` is a stable key for document-specific
  failures and otherwise `None`; `receipt` is truthful `FAILED` or `PARTIAL`
  progress, or `None` when there were zero document attempts.

The closed string enum `ProjectionFailureStage` has exact values:

| Member | String value |
| --- | --- |
| `EMBEDDER_IDENTITY` | `"embedder_identity"` |
| `EMBEDDING` | `"embedding"` |
| `REPLACEMENT` | `"replacement"` |
| `DELETION` | `"deletion"` |
| `RESET` | `"reset"` |

An ordinary collaborator exception is chained as the operation error's cause.
An invalid identity, malformed vector result, or non-`None` command result has
no internal cause. `KeyboardInterrupt` and `SystemExit` pass through unchanged.

Retrieval defines no workflow-specific public exception. Invalid top-level
workflow inputs, including a query that exceeds `RetrievalLimits`, raise
`ContractValidationError` before collaborator effects. After a collaborator
boundary is entered, ordinary `Exception` failures and malformed collaborator
returns contribute a content-free failure state. A failed branch contributes no
fragment or exception text; a `PARTIAL` result can contain independently
validated hits from another branch. No exception cause crosses the result
boundary. `BaseException` subclasses pass through unchanged.

## Documents and fragments

| Type | Fields | Construction rules |
| --- | --- | --- |
| `DocumentKey` | `corpus_id: str`, `document_id: str` | Both values are nonblank opaque identities. |
| `DocumentIdentity` | `key: DocumentKey`, `revision_id: str` | `key` is the exact class; the revision is a nonblank opaque identity. |
| `Document` | `identity: DocumentIdentity`, `text: str`, `attributes: tuple[tuple[str, str], ...] = ()` | `text` is stored exactly and may be empty; attributes follow the shared tuple rules. |
| `FragmentIdentity` | `document: DocumentIdentity`, `fragment_id: str`, `start: int`, `end: int` | The ID is nonblank; offsets are exact nonnegative integers satisfying `start < end`. |
| `Fragment` | `identity: FragmentIdentity`, `text: str`, `attributes: tuple[tuple[str, str], ...] = ()` | `text` is nonempty and `len(text) == end - start`; attributes follow the shared tuple rules. |

A fragment range is half-open, `[start, end)`, in Python Unicode code points.
It is not measured in bytes or user-perceived grapheme clusters. For example,
`"😀"` has one code point while `"e\u0301"` has two.

The contract checks that fragment text length equals the range width. A
standalone fragment cannot prove that its text equals the indicated source
slice; projection establishes that correspondence for fragments it derives.

## Embeddings and vector records

| Type | Fields | Construction rules |
| --- | --- | --- |
| `EmbeddingIdentity` | `model_id: str`, `dimensions: int` | The model ID is nonblank and opaque; dimensions is a positive exact integer. |
| `EmbeddingVector` | `values: tuple[float, ...]` | The tuple is exact and nonempty. Every coordinate is an exact `int` or `float`, excluding `bool`, and must convert to a finite float without overflow. Accepted coordinates are stored canonically as floats. |
| `VectorRecord` | `fragment: Fragment`, `embedding: EmbeddingVector` | Both fields require their exact contract classes. |

A standalone `EmbeddingVector` does not carry an `EmbeddingIdentity` and does
not itself compare length with declared dimensions. Projection validates the
embedder identity and every returned vector's exact dimensionality and
canonical finite-float representation.

## Projection request and manifest values

| Type | Fields | Construction rules |
| --- | --- | --- |
| `ProjectionIdentity` | `schema_id: str`, `embedding: EmbeddingIdentity` | The schema ID is nonblank and opaque; embedding requires its exact class. |
| `ChunkingPolicy` | `max_fragment_codepoints: int`, `overlap_codepoints: int` | Maximum is positive; overlap is nonnegative and smaller than maximum. |
| `ProjectionLimits` | `max_documents: int`, `max_document_codepoints: int`, `max_embedding_batch_size: int` | All three values are positive exact integers. |
| `ProjectionRequest` | `corpus_id: str`, `projection: ProjectionIdentity`, `chunking: ChunkingPolicy`, `limits: ProjectionLimits`, `documents: tuple[Document, ...]` | Documents must match the corpus, have unique stable keys, and remain within the count and per-document text caps. They are canonicalized by opaque `document_id`. |
| `ProjectionManifestEntry` | `document: DocumentIdentity`, `source_digest: str`, `fragment_count: int` | Digest must be lowercase `sha256:<64hex>` and fragment count is nonnegative. |
| `ProjectionCheckpoint` | `corpus_id: str`, `projection: ProjectionIdentity`, `token: str` | Corpus and token are nonblank; projection requires its exact class. |
| `ProjectionManifest` | `corpus_id: str`, `projection: ProjectionIdentity`, `chunking: ChunkingPolicy`, `entries: tuple[ProjectionManifestEntry, ...]`, `checkpoint: ProjectionCheckpoint` | Entries match the corpus, have unique stable keys, and are canonicalized by `document_id`; checkpoint corpus and projection match the manifest. |

The package produces manifests; the caller owns their persistence. Source
digests, fragment IDs, and checkpoint tokens are deterministic under explicit
v1 domains described in the [projection guide](projection.md#deterministic-projection-values).

The public integrity functions are synchronous and positional-only:

```python
def derive_source_digest(document: Document, /) -> str: ...

def derive_fragment_id(
    document: DocumentIdentity,
    start: int,
    end: int,
    /,
) -> str: ...

def derive_projection_checkpoint_token(
    corpus_id: str,
    projection: ProjectionIdentity,
    chunking: ChunkingPolicy,
    entries: tuple[ProjectionManifestEntry, ...],
    /,
) -> str: ...

def has_valid_projection_checkpoint(
    manifest: ProjectionManifest,
    /,
) -> bool: ...
```

They reproduce the same v1 integrity values used by projection and retrieval.
They validate exact public contract shapes and canonical manifest-entry order;
invalid inputs raise `ContractValidationError`. Checkpoint validation returns
whether the supplied token equals the derived token. It does not inspect a
provider index or establish authorization.

## Projection state and results

`ProjectionStateAvailability` is a closed string enum:

| Member | String value | Manifest rule |
| --- | --- | --- |
| `MISSING` | `"missing"` | Must be `None` |
| `PRESENT` | `"present"` | Must be an exact `ProjectionManifest` |
| `CORRUPT` | `"corrupt"` | Must be `None` |

`ProjectionStateSnapshot(availability, manifest)` stores that caller-supplied
state. Projection evaluates it to a closed `ProjectionStateStatus`:

| Member | String value | Meaning |
| --- | --- | --- |
| `MISSING` | `"missing"` | No state is available. |
| `CURRENT` | `"current"` | Valid state exactly matches the complete target. |
| `STALE` | `"stale"` | Valid compatible state requires mutations. |
| `CORRUPT` | `"corrupt"` | State is declared corrupt or fails integrity/consistency checks. |
| `SCHEMA_MISMATCH` | `"schema_mismatch"` | The schema ID differs. |
| `EMBEDDING_MISMATCH` | `"embedding_mismatch"` | The embedding identity differs. |

`ProjectionResult(status_before, receipt, manifest)` represents only complete
success. Its receipt and manifest have the same corpus, projection, and
checkpoint. An `UNCHANGED` result requires `CURRENT` state and zero attempted
and completed documents.

`ProjectionOutcome` is a closed string enum with these exact member values:

| Member | String value |
| --- | --- |
| `COMPLETED` | `"completed"` |
| `UNCHANGED` | `"unchanged"` |
| `PARTIAL` | `"partial"` |
| `FAILED` | `"failed"` |

`ProjectionReceipt` has fields `corpus_id`, `projection`, `outcome`,
`attempted_documents`, `completed_documents`, and `checkpoint`. It permits only
these truthful combinations:

| Outcome | Counts | Checkpoint |
| --- | --- | --- |
| `COMPLETED` | `completed_documents == attempted_documents`, including zero attempts | Required |
| `UNCHANGED` | `completed_documents == attempted_documents`, including zero attempts | Required |
| `PARTIAL` | `0 < completed_documents < attempted_documents` | Forbidden |
| `FAILED` | `attempted_documents > 0` and `completed_documents == 0` | Forbidden |

Any supplied checkpoint must have exactly the receipt's `corpus_id` and
`projection`. A successful workflow returns `ProjectionResult`; a failed
workflow exposes a failed or partial receipt only through
`ProjectionOperationError`.

## Projection workflows

Both public functions are synchronous and all parameters are positional-only:

```python
def project_documents(
    request: ProjectionRequest,
    state: ProjectionStateSnapshot,
    embedder: Borrowed[Embedder],
    writer: Borrowed[VectorIndexWriter],
    /,
) -> ProjectionResult: ...

def rebuild_projection(
    request: ProjectionRequest,
    state: ProjectionStateSnapshot,
    embedder: Borrowed[Embedder],
    writer: Borrowed[VectorIndexWriter],
    resetter: Borrowed[VectorIndexResetter],
    /,
) -> ProjectionResult: ...
```

`project_documents` returns without collaborator effects when state is
`CURRENT`, applies only the canonical delta when state is `STALE`, and raises
`ProjectionStateError` before effects for every other status.

`rebuild_projection` accepts every state status and always calls the resetter.
For a nonempty target it verifies embedder identity before reset, then replaces
every target document. For an empty target it resets without accessing the
embedder or writer. It is intentionally destructive and supplies no rollback
or retry. See the [projection guide](projection.md) for the full state and
failure matrices.

## Retrieval values

| Type | Fields | Construction rules |
| --- | --- | --- |
| `RetrievalLimits` | `max_query_codepoints: int` | The query-text cap is a positive exact integer. |
| `RetrievalQuery` | `corpus_id: str`, `text: str`, `hit_limit: int`, `candidate_limit: int` | Corpus and text are nonblank exact strings; limits are positive exact integers and `hit_limit <= candidate_limit`. Values are preserved exactly. |
| `RetrievalHit` | `fragment: Fragment`, `rank: int` | Fragment requires its exact class and rank is a positive exact integer. There is no score field. |
| `RetrievalResult` | `query: RetrievalQuery`, `outcome: RetrievalOutcome`, `hits: tuple[RetrievalHit, ...]`, `truncated: bool` | Nested values, the hit tuple, and the boolean require exact types. Hit count cannot exceed `query.hit_limit`. |

`RetrievalOutcome` is a closed string enum with exact values `"complete"`,
`"partial"`, `"unavailable"`, `"stale"`, and `"failed"`. Within every result,
ranks are contiguous from one, fragment identities are unique, and every
fragment belongs to the query corpus. `PARTIAL` requires at least one hit;
`UNAVAILABLE`, `STALE`, and `FAILED` require no hits and `truncated=False`.

The workflows set `truncated=True` exactly when validated, unique,
current-revision candidates exceed `query.hit_limit`. Hits and reader ports are
score-free; raw provider scores are neither represented nor promised
comparable. When constructing a `RetrievalResult` directly, callers remain
responsible for supplying a truthful `truncated` value because the value
contract cannot reconstruct discarded candidates.

## Collaborator ports

The protocols are synchronous, injected, structurally typed, and decorated
with `runtime_checkable`. Runtime protocol checks establish structural
presence, not the behavioral obligations below. Version 0.1.0 provides no
adapter, factory, or provider discovery.

| Port | Exact public operation | Semantic obligation |
| --- | --- | --- |
| `Embedder` | `identity: EmbeddingIdentity` | Identify the exact model used for produced vectors. |
| `Embedder` | `embed(texts: tuple[str, ...], /) -> tuple[EmbeddingVector, ...]` | Return one same-order vector per input text, each with `identity.dimensions` coordinates; empty input returns an empty tuple. |
| `VectorIndexWriter` | `replace_document(document: DocumentIdentity, records: tuple[VectorRecord, ...], /) -> None` | Replace all derived vectors for the document's stable key. Every record carries the supplied full identity; an empty record tuple is valid. |
| `VectorIndexWriter` | `delete_document(document: DocumentKey, /) -> None` | Delete every derived revision for the stable document key. |
| `VectorIndexResetter` | `reset_corpus(corpus_id: str, /) -> None` | Remove the complete derived vector projection for the corpus. |
| `VectorIndexReader` | `search(query: RetrievalQuery, embedding: EmbeddingVector, /) -> tuple[Fragment, ...]` | Return fragments from the requested corpus, in provider rank order, with at most `query.candidate_limit` entries. |
| `LexicalRetriever` | `search(query: RetrievalQuery, /) -> tuple[Fragment, ...]` | Return fragments from the requested corpus, in provider rank order, with at most `query.candidate_limit` entries. |

Projection enforces the embedder result rules and requires each writer or
resetter command to return exactly `None`. It cannot enforce external storage,
atomicity, authorization, concurrency, or lifecycle behavior.

Retrieval checks its embedder identity and output, provider tuple types and
candidate bounds, fragment integrity, corpus and published revisions, and
cross-provider identity consistency. The caller still owns provider selection,
authorization, persistence, concurrency, retries, and authoritative source
validation.

`Borrowed[T]` is the companion ownership marker, not a provider port. Its exact
behavior is documented in [resource lifecycle](lifecycle.md).

## Retrieval workflows

Both public functions are synchronous and all parameters are positional-only:

```python
def retrieve_semantic(
    query: RetrievalQuery,
    state: ProjectionStateSnapshot,
    limits: RetrievalLimits,
    embedder: Borrowed[Embedder],
    vector_reader: Borrowed[VectorIndexReader],
    /,
) -> RetrievalResult: ...

def retrieve_hybrid(
    query: RetrievalQuery,
    state: ProjectionStateSnapshot,
    limits: RetrievalLimits,
    embedder: Borrowed[Embedder],
    vector_reader: Borrowed[VectorIndexReader],
    lexical_retriever: Borrowed[LexicalRetriever],
    /,
) -> RetrievalResult: ...
```

Semantic retrieval embeds the query once, validates at most
`candidate_limit` vector candidates, preserves provider order through
current-revision filtering, and returns at most `hit_limit` hits. Hybrid
retrieval also obtains at most `candidate_limit` lexical candidates, preserves
each provider's original ranks, and fuses exact identities using deterministic
reciprocal rank fusion with offset 60. It uses provider ranks rather than raw
scores and applies opaque identity ordering to ties.

The workflows return `complete`, `partial`, `unavailable`, `stale`, or `failed`
according to published-state and collaborator results. They do not authorize,
cite, persist, log, retry, or manage collaborator resources. See the
[retrieval guide](retrieval.md) for candidate validation, exact outcome
handling, deterministic fusion, and the required user and agent host flow.
