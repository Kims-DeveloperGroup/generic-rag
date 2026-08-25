# Retrieval

`generic-rag` provides synchronous semantic and hybrid retrieval over a
host-published projection. The package composes caller-supplied collaborators;
it does not select, load, configure, retry, or close a provider.

The package root intentionally exports nothing. Import retrieval contracts from
`generic_rag.contracts`, borrowed collaborator protocols from
`generic_rag.ports`, integrity helpers from
`generic_rag.projection_integrity`, and workflows from
`generic_rag.retrieval`.

## Independent consumer example

This complete example constructs a valid published state, injects local fake
providers, invokes both workflows, and resolves returned fragment identities
against a still-authorized authoritative source before creating citations.

```python
from generic_rag.contracts import (
    ChunkingPolicy,
    Document,
    DocumentIdentity,
    DocumentKey,
    EmbeddingIdentity,
    EmbeddingVector,
    Fragment,
    FragmentIdentity,
    ProjectionCheckpoint,
    ProjectionIdentity,
    ProjectionManifest,
    ProjectionManifestEntry,
    ProjectionStateAvailability,
    ProjectionStateSnapshot,
    RetrievalLimits,
    RetrievalOutcome,
    RetrievalQuery,
)
from generic_rag.ports import Borrowed
from generic_rag.projection_integrity import (
    derive_fragment_id,
    derive_projection_checkpoint_token,
    derive_source_digest,
)
from generic_rag.retrieval import retrieve_hybrid, retrieve_semantic

corpus_id = "corpus-a"
source_document = Document(
    identity=DocumentIdentity(
        key=DocumentKey(corpus_id=corpus_id, document_id="document-a"),
        revision_id="revision-1",
    ),
    text="Approved source text.",
    attributes=(("classification", "public"),),
)
chunking = ChunkingPolicy(max_fragment_codepoints=128, overlap_codepoints=0)
embedding_identity = EmbeddingIdentity(model_id="example-embedding", dimensions=2)
projection_identity = ProjectionIdentity(
    schema_id="schema-v1",
    embedding=embedding_identity,
)
entries = (
    ProjectionManifestEntry(
        document=source_document.identity,
        source_digest=derive_source_digest(source_document),
        fragment_count=1,
    ),
)
checkpoint_token = derive_projection_checkpoint_token(
    corpus_id,
    projection_identity,
    chunking,
    entries,
)
published_manifest = ProjectionManifest(
    corpus_id=corpus_id,
    projection=projection_identity,
    chunking=chunking,
    entries=entries,
    checkpoint=ProjectionCheckpoint(
        corpus_id=corpus_id,
        projection=projection_identity,
        token=checkpoint_token,
    ),
)
published_state = ProjectionStateSnapshot(
    availability=ProjectionStateAvailability.PRESENT,
    manifest=published_manifest,
)

start, end = 0, len(source_document.text)
fragment = Fragment(
    identity=FragmentIdentity(
        document=source_document.identity,
        fragment_id=derive_fragment_id(source_document.identity, start, end),
        start=start,
        end=end,
    ),
    text=source_document.text[start:end],
    attributes=source_document.attributes,
)


class ExampleEmbedder:
    @property
    def identity(self):
        return embedding_identity

    def embed(self, texts, /):
        assert texts == ("Where is the approved source?",)
        return (EmbeddingVector(values=(1.0, 0.0)),)


class ExampleVectorReader:
    def search(self, query, embedding, /):
        assert embedding == EmbeddingVector(values=(1.0, 0.0))
        return (fragment,)


class ExampleLexicalRetriever:
    def search(self, query, /):
        return (fragment,)


query = RetrievalQuery(
    corpus_id=corpus_id,
    text="Where is the approved source?",
    hit_limit=2,
    candidate_limit=4,
)
limits = RetrievalLimits(max_query_codepoints=200)

# The host reauthorizes the corpus and documents for this query before calling
# retrieval. The published state must describe the projection being searched.
authorized_revisions = {source_document.identity: source_document}

semantic_result = retrieve_semantic(
    query,
    published_state,
    limits,
    Borrowed(ExampleEmbedder()),
    Borrowed(ExampleVectorReader()),
)
hybrid_result = retrieve_hybrid(
    query,
    published_state,
    limits,
    Borrowed(ExampleEmbedder()),
    Borrowed(ExampleVectorReader()),
    Borrowed(ExampleLexicalRetriever()),
)


def host_citations(result):
    if result.outcome not in (
        RetrievalOutcome.COMPLETE,
        RetrievalOutcome.PARTIAL,
    ):
        return ()

    citations = []
    for hit in result.hits:
        identity = hit.fragment.identity
        authoritative = authorized_revisions.get(identity.document)
        if authoritative is None:
            continue
        authoritative_text = authoritative.text[identity.start : identity.end]
        if authoritative_text != hit.fragment.text:
            continue
        citations.append(
            (
                identity.document.key.document_id,
                identity.document.revision_id,
                identity.start,
                identity.end,
            )
        )
    return tuple(citations)


assert semantic_result.outcome is RetrievalOutcome.COMPLETE
assert hybrid_result.outcome is RetrievalOutcome.COMPLETE
assert host_citations(semantic_result) == (("document-a", "revision-1", 0, 21),)
assert host_citations(hybrid_result) == (("document-a", "revision-1", 0, 21),)
```

`authorized_revisions` represents a host authorization decision made for this
query; it is not package state. A real host also verifies that the requesting
user or agent may use the corpus before invoking retrieval.

## Inputs and limits

Both workflows require:

- a `RetrievalQuery` with a nonblank corpus and query, a positive `hit_limit`,
  and a positive `candidate_limit` where `hit_limit <= candidate_limit`;
- the matching `ProjectionStateSnapshot` loaded by the host;
- a `RetrievalLimits` whose positive `max_query_codepoints` bounds the query
  before any collaborator call; and
- caller-owned collaborators wrapped in `Borrowed`.

The limits are caller-selected work budgets, not tenant quotas, rate limits,
authorization rules, or guarantees about a provider's resource use. Each
provider may return at most `candidate_limit` candidates. The package validates
that bound but does not configure the provider or restrict work hidden behind
its interface.

A missing state produces `unavailable`. A corrupt state or invalid checkpoint
produces `failed`. A state for another corpus produces `unavailable`. A valid
published state with no entries produces an empty `complete` result without
calling a provider.

Every provider return must be an exact tuple with no more than
`candidate_limit` entries. Each entry must be an exact `Fragment` for the query
corpus, stay within the manifest's fragment-width bound, and carry the
deterministic fragment ID for its document identity and range. Retrieval looks
up each stable `DocumentKey` in the manifest and keeps it only when the full
published `DocumentIdentity`, including revision, matches. Missing and old
revisions are filtered as stale. These checks still cannot prove that candidate
text equals its claimed authoritative source slice.

## Semantic retrieval

`retrieve_semantic` performs these steps in order:

1. Validate the top-level query, state, limits, and borrowed handles without
   collaborator effects.
2. Check that the borrowed embedder identity exactly matches the manifest's
   embedding identity. A mismatch produces `stale` without embedding or vector
   search.
3. Embed the exact one-element tuple `(query.text,)` and require exactly one
   finite vector with the published dimensions.
4. Ask the vector reader for up to `candidate_limit` fragments.
5. Validate, deduplicate, and filter the candidates against current manifest
   revisions, then return at most `hit_limit` hits in provider order.

Provider ranks used during validation and fusion are the candidates' original
one-based positions in the provider tuple. Filtering or exact-identity
deduplication does not close gaps in those source ranks. Semantic retrieval
preserves the remaining provider order, while public `RetrievalHit.rank` values
are final contiguous ranks from one. The first identical occurrence wins.
Conflicting payloads for one exact fragment identity fail that provider branch.

## Hybrid retrieval

`retrieve_hybrid` executes the semantic branch and then the lexical branch.
Semantic failure or staleness does not prevent the lexical call. Each provider
returns at most `candidate_limit` candidates, and its original ranks are
preserved.

Validated current-revision candidates are fused by reciprocal rank fusion:

```text
fused score = sum(1 / (60 + provider rank))
```

Raw provider scores are neither accepted nor returned. Exact fragment
identities are deduplicated across providers. Identical payloads contribute
both ranks; conflicting payloads produce a content-free `failed` result. Hits
sort by descending fused score, then by this opaque identity tuple:

```text
(corpus_id, document_id, revision_id, fragment_id, start, end)
```

That final comparison is deterministic raw string/integer ordering, not text
normalization or semantic relevance. `truncated` is true only when the number
of validated, unique, current-revision fragments exceeds `hit_limit`.

## Outcomes

| Outcome | Meaning and host response |
| --- | --- |
| `complete` | All required branches completed. Hits may be empty. Resolve every returned identity against current authorized source data before use. |
| `partial` | At least one validated hit is available, but a branch was stale or failed. Revalidate and cite usable hits; apply host policy before showing or injecting them. |
| `unavailable` | State is missing or the loaded manifest belongs to another corpus. Do not treat this as an empty authoritative answer. |
| `stale` | No hits are usable and a provider identity or candidate revision is stale. Reconcile or rebuild the projection. |
| `failed` | No hits are usable because state, collaborator output, or an ordinary collaborator call failed. Apply host retry, fallback, and audit policy. |

If stale and failed conditions coexist without hits, `failed` takes precedence.
Ordinary `Exception` failures at collaborator boundaries contribute only a
content-free failure state: the failed branch contributes no fragment or
exception text. A terminal `failed` result is empty; a `partial` result contains
only independently validated hits from another branch. `BaseException`
subclasses such as cancellation signals propagate and must be handled by the
host runtime.

## User and agent utilization

The required host flow is:

1. Authorize documents before projection and publish the resulting manifest
   and provider index under host-controlled persistence and concurrency.
2. Reauthorize the user or agent, corpus, and documents for every query.
3. Load the published manifest that matches the provider index being searched.
4. Invoke semantic or hybrid retrieval with caller-owned collaborators.
5. Resolve every returned `FragmentIdentity` against the still-authorized,
   authoritative source revision. Verify the exact source slice still equals
   the returned text, then create a host citation.
6. Show cited results to the authorized user, or inject only bounded cited
   context into an agent. Agents must retain the host citations in any derived
   answer or artifact.

Fragment text is derived, non-authoritative data. A collaborator candidate can
prove neither that its text equals the authoritative source slice nor that the
requester remains authorized. Matching deterministic identities and the
published revision is necessary but insufficient; host revalidation is
mandatory.

The package never authenticates or authorizes users, creates citations, logs
queries or results, loads providers, opens network connections, owns credentials
or provider resources, or decides prompt, tool, display, retry, or fallback
policy. See [Lifecycle and ownership](lifecycle.md) and
[Security and privacy](security-and-privacy.md) for host responsibilities.

## Operational ownership

The host owns provider selection and acquisition, network and credential
handling, retry and timeout policy, manifest and index persistence, concurrency
control, audit, disable/rebuild/purge procedures, and orderly shutdown.
`Borrowed` is only a no-op ownership marker: retrieval neither enters nor exits
the underlying collaborator. See [Projection](projection.md) for how to create
the published state and [API reference](api.md) for exact imports and
signatures.
