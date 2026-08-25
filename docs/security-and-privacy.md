# Security and Privacy

Version 0.1.0 defines in-process values, collaborator boundaries, and explicit
projection and retrieval workflows. The package itself performs no
persistence, network setup, provider discovery, credential loading, telemetry,
or background work. Projection passes derived fragment text and metadata to
caller-supplied providers. Retrieval passes query data to caller-supplied
providers and accepts fragment candidates from them. Those collaborator
effects are outside the package.

## Caller responsibility

Before constructing a `ProjectionRequest`, the caller must authorize every
source, revision, attribute, and intended destination. The package does not
authenticate an authoritative source or decide whether a user, tool, or agent
may project it.

Before every retrieval, the caller must reauthorize the requester, corpus, and
source revisions, then load the published manifest paired with the index being
searched. After retrieval, it must resolve each fragment identity against
still-authorized authoritative data, verify the exact source slice, and create
a citation before display or agent use. An agent must retain those host
citations in derived output.

The caller also controls:

- corpus and tenant isolation;
- which adapter implementations receive document text, fragment text,
  embeddings, attributes, identifiers, or queries;
- provider account, region, transport, and credential configuration;
- vector-index and manifest-store access control, retention, replacement,
  deletion, backup, and recovery;
- synchronization between vector mutations and manifest publication;
- logging, tracing, metrics, exception rendering, redaction, and incident
  response; and
- whether retrieved fragments are displayed, persisted, or supplied to another
  tool or agent, and the prompt/tool policy applied to them.

Do not place credentials or other secrets in attributes, opaque identifiers,
or checkpoint tokens. These fields preserve caller input and do not apply
redaction, escaping, authorization, or tenant isolation.

## Adapter effects

Calling `project_documents` can invoke the supplied embedder and writer.
Calling `rebuild_projection` can additionally reset all projected data for the
requested corpus. Those adapters may persist or transmit data according to
their implementations. Review their transport, storage, subprocess, network,
credential, and deletion behavior before use.

Calling `retrieve_semantic` passes the exact query text to the embedder, then
passes the query and derived query embedding to the vector reader. Calling
`retrieve_hybrid` also passes the query to the lexical retriever. A provider may
transmit, retain, correlate, or log those values according to its
implementation. The package has no hidden network or runtime dependency, but
injected collaborators can have both.

`Borrowed` only marks resources as caller-owned. It does not acquire, close,
authenticate, synchronize, sandbox, or reduce the privileges of an adapter.
The package supplies no transaction or rollback across the vector index and
caller-owned manifest store. See [resource lifecycle](lifecycle.md).

## Sensitive source and derived data

Treat all of the following as potentially sensitive:

- document and query text;
- ordered attributes and opaque identities;
- fragments and their source ranges;
- embeddings and vector records;
- source digests, fragment IDs, manifests, and checkpoint tokens;
- retrieval ranks and outcomes; and
- adapter exceptions and host logs.

Fragments and embeddings may reveal source information. Deterministic IDs and
digests may allow equality correlation or guessing attacks against predictable
content. Their `sha256:` representation provides neither encryption nor access
control and should not be used as proof of source ownership.

Retrieval results expose no raw provider score, but their ranks, identities,
fragment text, truncation flag, and outcome can still reveal content,
correlation, or availability information. Do not treat the score-free boundary
as anonymization.

Deleting an authoritative source does not automatically delete copies,
backups, logs, embeddings, or derived records held by an application or
provider. Incremental deletion and corpus reset cover only the behavior promised
by the supplied vector-index adapter.

## Limits and resource policy

`ProjectionLimits` rejects a request that exceeds its configured document count
or per-document code-point cap and bounds each embedding batch.
`ChunkingPolicy` bounds fragment size and overlap. These checks prevent one
accepted request from exceeding caller-selected values; they are not global
quotas, rate limits, memory isolation, provider billing controls, timeouts, or
admission control.

`RetrievalLimits.max_query_codepoints` bounds query text before collaborator
effects. `candidate_limit` bounds the tuple accepted from each provider, and
`hit_limit` bounds returned hits. These are likewise caller-selected work
budgets, not quotas, authorization, cost controls, or proof that a provider did
only bounded internal work.

Choose limits from trusted application policy rather than untrusted request
parameters. Account for the fact that a small fragment size and large permitted
document set can still produce many fragments and provider operations. Supply
external cancellation, concurrency, cost, and capacity controls where needed.

## Errors and logging

`ProjectionStateError` and the direct message of
`ProjectionOperationError` are content-free. Ordinary adapter exceptions are
preserved as chained causes, so rendering the full exception chain may expose
an adapter's message or fields. Adapter implementations must avoid placing
document text, fragments, embeddings, credentials, or sensitive identifiers in
exceptions and logs. Applications should apply redaction before exporting
traces or error reports.

The package does not retry collaborator operations. A failure can leave earlier
document mutations in place, and a rebuild failure can occur after the corpus
was reset. Do not publish a failed or partial receipt as a completed checkpoint;
isolate the affected projection and recover under caller-owned policy.

Retrieval maps ordinary collaborator exceptions and malformed collaborator
returns to a failed provider branch, producing `failed` when no validated hit
remains or `partial` when another branch supplies one. A `failed` result is
content-free, and a `partial` result contains only independently validated
hits; provider exception text and rejected candidates do not cross the result
boundary. Stale provider identity or revision data is reported separately as
`stale`, or as `partial` when another branch supplies a validated hit. The host
decides whether to retry, fall back, audit, or suppress a result.
`BaseException` subclasses propagate unchanged, so the host runtime must handle
its own cancellation and shutdown signals. The package does not log queries,
fragments, provider failures, or outcomes.

## Trust boundary

A fragment range is a half-open Python code-point range. Standalone contracts
check its width but do not prove that text came from the indicated source.
Projection derives its own fragment text from the supplied document, but the
package still cannot prove that the supplied document or revision was
authoritative or authorized.

Retrieval candidates are collaborator-supplied derived data. Deterministic
fragment identity and a current manifest revision do not prove that candidate
text equals the authoritative source slice or that the requester remains
authorized. Host reauthorization, source resolution, exact-slice validation,
and citation creation are mandatory before use.

Version 0.1.0 provides no built-in encryption, authentication, authorization,
ACL, content filter, persistence security, network security, citation
creation or validation, secret management, vendor guarantee, logging, or
user/agent policy. A consuming application must select and assess those
controls for its environment.

The complete public value and collaborator boundaries are listed in the [API
reference](api.md), and deterministic projection behavior is documented in the
[projection guide](projection.md). Retrieval behavior and the required host
flow are documented in the [retrieval guide](retrieval.md).
