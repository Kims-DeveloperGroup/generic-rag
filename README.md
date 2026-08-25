# generic-rag

`generic-rag` is a provider-neutral, runtime-dependency-free foundation for
retrieval-augmented generation (RAG). Version 0.1.0 requires Python 3.11 or
later and provides immutable contracts, typed error categories, synchronous
collaborator protocols, deterministic bounded projection orchestration,
semantic retrieval, rank-based hybrid retrieval, and explicit caller-owned
borrowing.

The package has no built-in adapter, provider, factory, persistence, network
client, configuration system, authentication, authorization, citation
mechanism, or CLI.

## Install from a checkout

The project is not documented as a published package yet. From a repository
checkout, install it with:

```console
python -m pip install .
```

The installed package has no runtime dependencies. Build and development tools
are separate locked dependency groups.

## Project approved documents

The host application remains responsible for authorization and policy checks.
After approving source content, construct a complete bounded target, wrap
application-owned adapters in `Borrowed`, and call a projection workflow:

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
from generic_rag.projection import rebuild_projection

class ExampleEmbedder:
    identity = EmbeddingIdentity("example-model", 2)

    def embed(self, texts, /):
        return tuple(EmbeddingVector((float(len(text)), 0.0)) for text in texts)


class ExampleWriter:
    def replace_document(self, document, records, /):
        # Replace the complete projection for this stable document key.
        return None

    def delete_document(self, document, /):
        return None


class ExampleResetter:
    def reset_corpus(self, corpus_id, /):
        # Remove every projected document for this corpus.
        return None


request = ProjectionRequest(
    "corpus-a",
    ProjectionIdentity("schema-v1", ExampleEmbedder.identity),
    ChunkingPolicy(max_fragment_codepoints=800, overlap_codepoints=80),
    ProjectionLimits(
        max_documents=100,
        max_document_codepoints=100_000,
        max_embedding_batch_size=32,
    ),
    (
        Document(
            DocumentIdentity(
                DocumentKey("corpus-a", "document-1"),
                "revision-3",
            ),
            "Approved source text",
            (("classification", "internal"),),
        ),
    ),
)

# Bootstrap and recovery are explicit and destructive: reset, then replace.
result = rebuild_projection(
    request,
    ProjectionStateSnapshot(ProjectionStateAvailability.MISSING, None),
    Borrowed(ExampleEmbedder()),
    Borrowed(ExampleWriter()),
    Borrowed(ExampleResetter()),
)

# Persist result.manifest in application-owned state only after success.
```

For normal updates, load that manifest into a present
`ProjectionStateSnapshot` and call `project_documents`; it changes only added,
updated, removed, or rechunked documents. Use `rebuild_projection` only when an
explicit corpus-wide reset is intended. See the [projection guide](docs/projection.md)
for the complete lifecycle, state matrix, adapter obligations, and failure
behavior.

Load the corresponding published manifest before retrieval, reauthorize each
query and source in the host, and call `retrieve_semantic` or
`retrieve_hybrid` with borrowed provider implementations. Returned fragment
text is non-authoritative: resolve each identity against the still-authorized
source revision and create host-owned citations before showing results to a
user or injecting bounded context into an agent. See the [retrieval
guide](docs/retrieval.md) for an executable independent-consumer example,
deterministic fusion behavior, outcome handling, and the complete host flow.

Public values must be imported from their owning modules:

- `generic_rag.contracts`
- `generic_rag.errors`
- `generic_rag.ports`
- `generic_rag.projection`
- `generic_rag.projection_integrity`
- `generic_rag.retrieval`

The package root intentionally has no re-exports: `generic_rag.__all__ == ()`.
See the [API reference](docs/api.md) for every supported name and invariant.

## Application RAG flow

The package itself has no concept of a user or agent. A consuming application
decides which sources a user may approve, which queries may be submitted, which
provider implementations receive data, and whether retrieved fragments are
shown to a user or supplied to a downstream tool or agent.

- Projection accepts caller-approved documents and explicitly injected
  collaborators. It derives deterministic fragments under a bounded chunking
  policy, embeds ordered fragment text, replaces or deletes complete document
  projections, and returns a manifest and truthful receipt for caller-owned
  persistence.
- Retrieval accepts the matching caller-loaded published state and injected
  semantic or lexical providers. It validates and filters current-revision
  candidates, deduplicates exact identities, and returns bounded score-free
  hits. Hybrid retrieval fuses provider ranks deterministically; it does not
  compare raw provider scores.
- The host reauthorizes every query, resolves each returned fragment identity
  against authoritative source data, verifies the exact source slice, and
  creates citations. It may then show cited results to a user or inject bounded
  cited context into an agent; the agent must retain those citations.

The package does not decide provider selection, authentication, authorization,
prompt or tool policy, retry, display, logging, or resource lifecycle. See the
[retrieval guide](docs/retrieval.md), [resource lifecycle](docs/lifecycle.md),
and [security and privacy](docs/security-and-privacy.md).

## Compatibility

Version 0.1.0 is pre-1.0. Consumers should pin a reviewed version and should not
assume compatibility across minor releases. For this release, direct imports
from the documented owning modules are the supported public paths; root-level
imports are not.

The distribution includes `py.typed`. The wheel contains exactly the seven
importable modules `generic_rag`, `generic_rag.errors`,
`generic_rag.contracts`, `generic_rag.ports`, `generic_rag.projection`,
`generic_rag.projection_integrity`, and `generic_rag.retrieval`, plus the typing
marker.

## Development verification

The project commits a universal lock generated with uv 0.12.5. Reproduce the
locked environment and primary checks with:

```console
uv lock --check
uv sync --locked --all-groups
uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -v
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
uv run --frozen mypy --strict src tests
uv run --frozen python -m compileall -q src tests
```

Build both distributions into a temporary output directory and inspect them:

```console
rag_dist_dir="$(mktemp -d)"
uv run --frozen python -m build --no-isolation --outdir "$rag_dist_dir"
uv run --frozen python tests/support/verify_artifacts.py "$rag_dist_dir"
```

CI is configured to run the tests on Python 3.11 and 3.14. On Python 3.11 it
also runs lint, format, strict type, compilation, artifact, source-rebuild,
clean-install, and isolated-import checks. The CI matrix is the authoritative
cross-version result.
