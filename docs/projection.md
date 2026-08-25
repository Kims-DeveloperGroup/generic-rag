# Projection

Version 0.1.0 can turn a complete, caller-approved document set into a
deterministic vector projection. It supplies orchestration, contracts, and
failure reporting; the caller supplies and owns the embedder, vector index,
projection-state persistence, authorization policy, and synchronization.

Projection does not make content retrievable through this package. Retrieval,
result composition, and user or agent integration remain planned for [Issue
#4](https://github.com/Kims-DeveloperGroup/generic-rag/issues/4).

## Required adapters and state

Implement application-specific objects that structurally satisfy these public
ports:

- `Embedder` exposes an `EmbeddingIdentity` and returns one same-order vector
  of exactly that dimensionality for every input text.
- `VectorIndexWriter` replaces the complete record set for one stable document
  key or deletes every revision of that key. Both commands must return `None`.
- `VectorIndexResetter` removes every projected document for one corpus and
  returns `None`. It is required only by `rebuild_projection`.

The package does not provide adapters or a manifest store. Persist the
successful `ProjectionResult.manifest` in application-owned state, then load it
as a `ProjectionStateSnapshot` for the next operation. The stored manifest and
the vector index are one logical projection: publish them under application-
controlled synchronization so another operation cannot observe an unintended
combination.

All workflow parameters are positional-only. Each collaborator must be wrapped
in the exact `Borrowed` class; borrowing never acquires, closes, resets, or
otherwise owns the wrapped resource.

## Bootstrap, then update incrementally

The first projection has no valid present manifest, so bootstrap it through the
explicitly destructive rebuild path:

```python
from generic_rag.contracts import (
    ChunkingPolicy,
    Document,
    DocumentIdentity,
    DocumentKey,
    EmbeddingIdentity,
    EmbeddingVector,
    ProjectionIdentity,
    ProjectionLimits,
    ProjectionRequest,
    ProjectionStateAvailability,
    ProjectionStateSnapshot,
)
from generic_rag.ports import Borrowed
from generic_rag.projection import project_documents, rebuild_projection


class ApplicationEmbedder:
    identity = EmbeddingIdentity("embedding-model-v1", 2)

    def embed(self, texts, /):
        return tuple(EmbeddingVector((float(len(text)), 0.0)) for text in texts)


class ApplicationVectorIndex:
    def replace_document(self, document, records, /):
        # Replace all derived records for document.key in one adapter operation.
        return None

    def delete_document(self, document, /):
        # Delete every projected revision of this stable document key.
        return None

    def reset_corpus(self, corpus_id, /):
        # Delete the complete vector projection for this corpus.
        return None


def target(revision_id, text):
    return ProjectionRequest(
        "approved-corpus",
        ProjectionIdentity("schema-v1", ApplicationEmbedder.identity),
        ChunkingPolicy(800, 80),
        ProjectionLimits(100, 100_000, 32),
        (
            Document(
                DocumentIdentity(
                    DocumentKey("approved-corpus", "document-1"),
                    revision_id,
                ),
                text,
                (("source", "application-authorized"),),
            ),
        ),
    )


embedder = ApplicationEmbedder()
index = ApplicationVectorIndex()
initial_request = target("revision-1", "First approved source text")

initial_result = rebuild_projection(
    initial_request,
    ProjectionStateSnapshot(ProjectionStateAvailability.MISSING, None),
    Borrowed(embedder),
    Borrowed(index),
    Borrowed(index),
)
# Persist initial_result.manifest only after the call succeeds.

updated_request = target("revision-2", "Updated approved source text")
updated_result = project_documents(
    updated_request,
    ProjectionStateSnapshot(
        ProjectionStateAvailability.PRESENT,
        initial_result.manifest,
    ),
    Borrowed(embedder),
    Borrowed(index),
)
# Atomically publish updated_result.manifest as the new application-owned state.
```

The example uses one object for the writer and resetter ports, but separate
objects are equally valid. Production adapters must implement the stated
complete replacement, deletion, reset, persistence, and synchronization
semantics; the example deliberately omits storage.

## Incremental lifecycle

`project_documents(request, state, embedder, writer)` applies a complete target
to compatible state:

1. It validates the exact request, snapshot, and `Borrowed` wrapper types and
   prepares the complete target before calling a collaborator.
2. It evaluates the supplied manifest against the target. `current` returns an
   `UNCHANGED` result with zero attempts and no collaborator calls. Only
   `stale` proceeds; every other status raises `ProjectionStateError` before
   effects.
3. It sorts mutations by opaque `document_id`. Added documents, new revisions,
   and documents affected by a chunking change are replaced; absent target
   documents are deleted; unchanged entries are skipped. A source change under
   an unchanged full document identity is corrupt state, not an incremental
   replacement.
4. If any replacement is needed, it validates the embedder identity once,
   embeds nonempty ordered batches no larger than
   `max_embedding_batch_size`, and calls the writer once per document. A
   delete-only update does not access the embedder.
5. After every mutation succeeds, it returns `COMPLETED` with the complete
   target manifest and checkpoint. The caller may then publish that manifest.

An empty document produces no fragments and is still replaced with an explicit
empty record tuple. An empty target incrementally deletes every document in the
previous valid manifest.

## Destructive rebuild lifecycle

`rebuild_projection(request, state, embedder, writer, resetter)` is the explicit
bootstrap and recovery operation. It accepts every state status, including
`current`, but always resets the requested corpus and recreates the complete
target:

1. It validates inputs and computes `status_before` without effects.
2. For a nonempty target, it validates the embedder identity once *before* the
   reset. A mismatch or invalid identity therefore leaves reset and writer
   untouched.
3. It calls `reset_corpus`, then embeds and replaces every target document in
   canonical `document_id` order. An empty target calls only the resetter.
4. Complete success returns a `COMPLETED` result, even when the target has zero
   documents.

Reset is intentionally destructive. The package provides no transaction,
rollback, retry, or two-phase publication across the resetter, writer, and
caller-owned manifest store. If reset succeeds and a later replacement fails,
the vector index can contain an incomplete rebuild and no successful
checkpoint is issued. Keep the prior manifest out of service and run an
application-controlled recovery, normally another full rebuild.

## State matrix

`ProjectionStateAvailability` describes what the caller could load.
`ProjectionStateStatus` is the workflow's evaluation of that snapshot against
the requested complete target.

| Supplied state | Evaluated status | `project_documents` | `rebuild_projection` |
| --- | --- | --- | --- |
| `MISSING` with no manifest | `MISSING` | Raises before effects | Resets and builds target |
| `CORRUPT` with no manifest | `CORRUPT` | Raises before effects | Resets and builds target |
| Valid present manifest exactly matching target | `CURRENT` | Returns `UNCHANGED`; no effects | Resets and rebuilds target |
| Valid present manifest with compatible target differences | `STALE` | Applies incremental mutations | Resets and rebuilds target |
| Present manifest with invalid checkpoint, wrong corpus, or same-revision source/count inconsistency | `CORRUPT` | Raises before effects | Resets and rebuilds target |
| Present manifest with another schema ID | `SCHEMA_MISMATCH` | Raises before effects | Resets and rebuilds target |
| Present manifest with another embedding identity | `EMBEDDING_MISMATCH` | Raises before effects | Resets and rebuilds target |

A revision change is a normal stale update. For the same full document
identity, changing the source digest—or the fragment count under unchanged
chunking—is treated as corrupt state rather than an unannounced rewrite.

## Deterministic projection values

Documents and manifest entries are canonicalized by opaque `document_id`.
Fragments use half-open Python Unicode code-point ranges and copy the source's
ordered attributes. Each fragment contains at most
`max_fragment_codepoints`; consecutive fragments overlap by
`overlap_codepoints`. Attribute order and duplicates remain significant.

Source digests, fragment IDs, and checkpoint tokens use lowercase
`sha256:<64hex>` values. The current algorithms serialize tagged fields as
UTF-8 with `surrogatepass`, prefix each encoded field with its unsigned
eight-byte big-endian length, and hash them under these versioned domains:

| Value | v1 domain | Bound inputs |
| --- | --- | --- |
| Source digest | `generic-rag:projection-source:v1` | Exact document text and ordered attributes |
| Fragment ID | `generic-rag:fragment-id:v1` | Corpus, document, revision, and fragment range |
| Checkpoint token | `generic-rag:projection-checkpoint:v1` | Corpus, projection and embedding identities, chunking policy, and ordered manifest entries |

These values are reproducible for the same inputs and current v1 algorithm,
including across clean processes. The v1 domain names do not promise that a
future package version will retain the same algorithm or accept an old
manifest. Consumers that persist projection state should pin and review the
package version and use explicit rebuild for an incompatible upgrade.

Hashes are deterministic comparison and identity values, not encryption,
authorization, or a proof of source ownership. See [security and
privacy](security-and-privacy.md).

## Failures and receipts

Invalid public values raise `ContractValidationError`. Incremental state that
is not `current` or `stale` raises `ProjectionStateError`, whose `status` gives
the evaluated reason. Both cases are detected before collaborator effects.

An ordinary collaborator exception, or an invalid collaborator return, raises
`ProjectionOperationError` with:

- `stage`: `EMBEDDER_IDENTITY`, `EMBEDDING`, `REPLACEMENT`, `DELETION`, or
  `RESET`;
- `affected_document`: the stable key for a document-specific failure, else
  `None`; and
- `receipt`: `FAILED` when no planned document completed, `PARTIAL` after one
  or more but not all planned documents completed, or `None` when there were
  zero document attempts.

Failure receipts never contain a checkpoint. `attempted_documents` is the
total mutation or rebuild-document count; `completed_documents` counts only
fully completed document operations. The original ordinary exception is
chained as the cause. A structurally invalid identity, vector result, or
non-`None` writer/resetter return has no internal cause. `KeyboardInterrupt`
and `SystemExit` are neither translated nor retried.

The public error messages do not include document or vector content. Adapter
exception messages remain reachable through exception chaining, so adapters
and application logging must avoid disclosing sensitive values.

See the [API reference](api.md) for exact signatures and value invariants and
[resource lifecycle](lifecycle.md) for ownership details.
