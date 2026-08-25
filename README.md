# generic-rag

`generic-rag` is a provider-neutral, runtime-dependency-free foundation for
retrieval-augmented generation (RAG). Version 0.1.0 requires Python 3.11 or
later and provides immutable contracts, typed error categories, synchronous
collaborator protocols, and explicit caller-owned borrowing.

No projection or retrieval algorithm is implemented in 0.1.0. The package has
no built-in adapter, provider, factory, persistence, network client,
configuration system, authentication, citation mechanism, or CLI.

## Install from a checkout

The project is not documented as a published package yet. From a repository
checkout, install it with:

```console
python -m pip install .
```

The installed package has no runtime dependencies. Build and development tools
are separate locked dependency groups.

## Use the contracts in application code

The host application remains responsible for authorization and policy checks.
After approving a source and query, application code can construct generic
values and accept an application-owned provider through a protocol:

```python
from generic_rag.contracts import (
    Document,
    DocumentIdentity,
    DocumentKey,
    EmbeddingVector,
    RetrievalQuery,
)
from generic_rag.ports import Borrowed, Embedder

# Construct these values only after application-specific authorization.
approved_document = Document(
    identity=DocumentIdentity(
        key=DocumentKey(corpus_id="corpus-a", document_id="document-1"),
        revision_id="revision-3",
    ),
    text="Approved source text",
    attributes=(("classification", "internal"),),
)
query = RetrievalQuery(
    corpus_id=approved_document.identity.key.corpus_id,
    text="What does the source say?",
    hit_limit=5,
    candidate_limit=20,
)


def application_embed_query(
    provider: Embedder,
    request: RetrievalQuery,
) -> EmbeddingVector:
    with Borrowed(provider) as embedder:
        vectors = embedder.embed((request.text,))
    if len(vectors) != 1:
        raise ValueError("the provider violated the Embedder contract")
    return vectors[0]
```

This is application orchestration, not a package retrieval workflow. Version
0.1.0 defines the boundary that provider implementations and later generic
workflows will use; it does not construct providers or call them on a user's
behalf.

Public values must be imported from their owning modules:

- `generic_rag.contracts`
- `generic_rag.errors`
- `generic_rag.ports`

The package root intentionally has no re-exports: `generic_rag.__all__ == ()`.
See the [API reference](docs/api.md) for every supported name and invariant.

## Planned RAG flow

The package itself has no concept of a user or agent. A consuming application
decides which sources a user may approve, which queries may be submitted, which
provider implementations receive data, and whether retrieved fragments are
shown to a user or supplied to a downstream tool or agent.

- [Issue #3](https://github.com/Kims-DeveloperGroup/generic-rag/issues/3) is
  planned to add generic projection orchestration. Its intended responsibility
  is to accept caller-approved documents and explicitly injected collaborators,
  derive fragments under a defined chunking policy, embed ordered fragment
  text, replace or delete complete document projections, and report truthful
  checkpoints and receipts. Its precise API and failure behavior are not part
  of 0.1.0.
- [Issue #4](https://github.com/Kims-DeveloperGroup/generic-rag/issues/4) is
  planned to add retrieval and composition. Its intended responsibility is to
  use an injected `Embedder` and `VectorIndexReader` for semantic candidates
  and an injected `LexicalRetriever` for lexical candidates, then define
  deduplication, fusion, limiting, and outcome behavior. Provider rank will be
  the input; raw provider scores are not represented or assumed comparable.

The caller/provider ownership model remains explicit throughout this plan. See
[resource lifecycle](docs/lifecycle.md) and
[security and privacy](docs/security-and-privacy.md).

## Compatibility

Version 0.1.0 is pre-1.0. Consumers should pin a reviewed version and should not
assume compatibility across minor releases. For this release, direct imports
from the documented owning modules are the supported public paths; root-level
imports are not.

The distribution includes `py.typed`. The wheel contains exactly the four
importable modules `generic_rag`, `generic_rag.errors`,
`generic_rag.contracts`, and `generic_rag.ports`, plus the typing marker.

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
clean-install, and isolated-import checks. A local Python 3.14.2 run currently
contains 53 passing tests; the CI matrix is the authoritative cross-version
result.
