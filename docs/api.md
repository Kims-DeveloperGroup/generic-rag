# Public API

Version 0.1.0 exposes immutable values, typed error categories, and synchronous
collaborator protocols. It does not expose projection or retrieval algorithms.
See the [project overview](../README.md), [resource lifecycle](lifecycle.md),
and [security and privacy boundary](security-and-privacy.md) for the surrounding
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
- `RetrievalQuery`
- `RetrievalOutcome`
- `RetrievalHit`
- `RetrievalResult`

`generic_rag.ports` exports exactly:

- `Borrowed`
- `Embedder`
- `VectorIndexWriter`
- `VectorIndexReader`
- `LexicalRetriever`

The package does not support importing public values from the package root.

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

`GenericRagError` is the base for the three public categories:

```text
GenericRagError
├── ContractValidationError
├── CollaborationError
└── StateCompatibilityError
```

- `ContractValidationError` reports a violated public value invariant.
- `CollaborationError` is reserved for a workflow that translates a
  collaborator operation failure.
- `StateCompatibilityError` is reserved for a workflow that detects derived
  state with an incompatible projection identity.

Version 0.1.0 has no projection or retrieval workflow that raises the latter
two categories. `Borrowed` also leaves provider exceptions unchanged.

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

The package checks that fragment text length equals the range width. Version
0.1.0 does not retain an authoritative `Document` beside a `Fragment`, so it
cannot verify that the text equals the indicated source slice. The caller, or
a future projection workflow, must establish that correspondence.

## Embeddings and vector records

| Type | Fields | Construction rules |
| --- | --- | --- |
| `EmbeddingIdentity` | `model_id: str`, `dimensions: int` | The model ID is nonblank and opaque; dimensions is a positive exact integer. |
| `EmbeddingVector` | `values: tuple[float, ...]` | The tuple is exact and nonempty. Every coordinate is an exact `int` or `float`, excluding `bool`, and must convert to a finite float without overflow. Oversized integers that cannot be represented as finite floats are rejected; accepted coordinates are stored canonically as floats. |
| `VectorRecord` | `fragment: Fragment`, `embedding: EmbeddingVector` | Both fields require their exact contract classes. |

A standalone `EmbeddingVector` does not carry an `EmbeddingIdentity`. Version
0.1.0 therefore does not compare the vector length with an identity's declared
`dimensions`; a provider and future orchestration must satisfy that semantic
relationship.

## Projection state

| Type | Fields | Construction rules |
| --- | --- | --- |
| `ProjectionIdentity` | `schema_id: str`, `embedding: EmbeddingIdentity` | The schema ID is nonblank and opaque; embedding requires its exact class. |
| `ProjectionCheckpoint` | `corpus_id: str`, `projection: ProjectionIdentity`, `token: str` | Corpus and token are nonblank opaque strings; projection requires its exact class. |
| `ProjectionReceipt` | `corpus_id: str`, `projection: ProjectionIdentity`, `outcome: ProjectionOutcome`, `attempted_documents: int`, `completed_documents: int`, `checkpoint: ProjectionCheckpoint \| None` | Nested values require their exact classes; counts are nonnegative exact integers and completed cannot exceed attempted. |

`ProjectionOutcome` is a closed string enum with these exact member values:

| Member | String value |
| --- | --- |
| `COMPLETED` | `"completed"` |
| `UNCHANGED` | `"unchanged"` |
| `PARTIAL` | `"partial"` |
| `FAILED` | `"failed"` |

Unknown enum values raise `ContractValidationError`. A receipt can represent
only the following truthful combinations:

| Outcome | Counts | Checkpoint |
| --- | --- | --- |
| `COMPLETED` | `completed_documents == attempted_documents`, including zero attempts | Required |
| `UNCHANGED` | `completed_documents == attempted_documents`, including zero attempts | Required |
| `PARTIAL` | `0 < completed_documents < attempted_documents` | Forbidden |
| `FAILED` | `attempted_documents > 0` and `completed_documents == 0` | Forbidden |

Any supplied checkpoint must have exactly the receipt's `corpus_id` and
`projection`.

## Retrieval values

| Type | Fields | Construction rules |
| --- | --- | --- |
| `RetrievalQuery` | `corpus_id: str`, `text: str`, `hit_limit: int`, `candidate_limit: int` | Corpus and text are nonblank exact strings; limits are positive exact integers and `hit_limit <= candidate_limit`. Values are preserved exactly. |
| `RetrievalHit` | `fragment: Fragment`, `rank: int` | Fragment requires its exact class and rank is a positive exact integer. There is no score field. |
| `RetrievalResult` | `query: RetrievalQuery`, `outcome: RetrievalOutcome`, `hits: tuple[RetrievalHit, ...]`, `truncated: bool` | Nested values, the hit tuple, and the boolean require exact types. Hit count cannot exceed `query.hit_limit`. |

`RetrievalOutcome` is a closed string enum with these exact member values:

| Member | String value |
| --- | --- |
| `COMPLETE` | `"complete"` |
| `PARTIAL` | `"partial"` |
| `UNAVAILABLE` | `"unavailable"` |
| `STALE` | `"stale"` |
| `FAILED` | `"failed"` |

Unknown enum values raise `ContractValidationError`. Within every result, ranks
must be contiguous starting at one, fragment identities must be unique, and
every fragment's corpus must match the query corpus. Fragment attributes do not
make two otherwise identical fragment identities distinct.

The outcome matrix is:

| Outcome | Hits | `truncated` |
| --- | --- | --- |
| `COMPLETE` | Zero through `query.hit_limit` | Either boolean |
| `PARTIAL` | One through `query.hit_limit` | Either boolean |
| `UNAVAILABLE` | None | `False` |
| `STALE` | None | `False` |
| `FAILED` | None | `False` |

`truncated=True` is the caller's explicit assertion that otherwise valid work
or results were cut by the query budget. It does not imply that
`len(hits) == query.hit_limit`; a bounded `COMPLETE` result may therefore still
be truncated. Hits and reader ports are score-free. Raw scores from different
providers are neither represented nor promised to be comparable.

## Collaborator ports

The protocols are synchronous, injected, structurally typed, and decorated
with `runtime_checkable`. Runtime protocol checks establish structural presence,
not the behavioral obligations below. Version 0.1.0 provides no implementation,
adapter, factory, provider discovery, or provider-behavior enforcement.

| Port | Exact public operation | Semantic obligation |
| --- | --- | --- |
| `Embedder` | `identity: EmbeddingIdentity` | Identify the exact model used for produced vectors. |
| `Embedder` | `embed(texts: tuple[str, ...], /) -> tuple[EmbeddingVector, ...]` | Return one same-order vector per input text, each with `identity.dimensions` coordinates; empty input returns an empty tuple. |
| `VectorIndexWriter` | `replace_document(document: DocumentIdentity, records: tuple[VectorRecord, ...], /) -> None` | Replace all derived vectors for the document's stable key. Every record carries the supplied full `DocumentIdentity`; an empty record tuple is valid. |
| `VectorIndexWriter` | `delete_document(document: DocumentKey, /) -> None` | Delete every derived revision for the stable document key. |
| `VectorIndexReader` | `search(query: RetrievalQuery, embedding: EmbeddingVector, /) -> tuple[Fragment, ...]` | Return fragments from the requested corpus, in provider rank order, with at most `query.candidate_limit` entries. |
| `LexicalRetriever` | `search(query: RetrievalQuery, /) -> tuple[Fragment, ...]` | Return fragments from the requested corpus, in provider rank order, with at most `query.candidate_limit` entries. |

`Borrowed[T]` is the companion ownership marker, not a provider port. Its exact
behavior is documented in [resource lifecycle](lifecycle.md).

## Planned workflows

Projection orchestration is planned for Issue #3. Retrieval and composition are
planned for Issue #4. Those future workflows are expected to accept protocol-
compatible collaborators explicitly, but their algorithms, APIs, compatibility
checks, exception translation, and outcome mapping are not implemented or
promised by version 0.1.0.
