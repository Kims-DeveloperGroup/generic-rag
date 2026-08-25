# Security and Privacy

Version 0.1.0 defines in-process values, collaborator boundaries, and an
explicit projection workflow. The package itself performs no persistence,
network setup, provider discovery, credential loading, telemetry, or background
work. A projection call does pass derived fragment text and metadata to the
caller-supplied embedder and vector writer, whose effects are outside the
package.

## Caller responsibility

Before constructing a `ProjectionRequest`, the caller must authorize every
source, revision, attribute, and intended destination. The package does not
authenticate an authoritative source or decide whether a user, tool, or agent
may project it.

The caller also controls:

- corpus and tenant isolation;
- which adapter implementations receive document text, fragment text,
  embeddings, attributes, identifiers, or future queries;
- provider account, region, transport, and credential configuration;
- vector-index and manifest-store access control, retention, replacement,
  deletion, backup, and recovery;
- synchronization between vector mutations and manifest publication;
- logging, tracing, metrics, exception rendering, redaction, and incident
  response; and
- whether future retrieved fragments are displayed, persisted, or supplied to
  another tool or agent.

Do not place credentials or other secrets in attributes, opaque identifiers,
or checkpoint tokens. These fields preserve caller input and do not apply
redaction, escaping, authorization, or tenant isolation.

## Adapter effects

Calling `project_documents` can invoke the supplied embedder and writer.
Calling `rebuild_projection` can additionally reset all projected data for the
requested corpus. Those adapters may persist or transmit data according to
their implementations. Review their transport, storage, subprocess, network,
credential, and deletion behavior before use.

`Borrowed` only marks resources as caller-owned. It does not acquire, close,
authenticate, synchronize, sandbox, or reduce the privileges of an adapter.
The package supplies no transaction or rollback across the vector index and
caller-owned manifest store. See [resource lifecycle](lifecycle.md).

## Sensitive source and derived data

Treat all of the following as potentially sensitive:

- document and future query text;
- ordered attributes and opaque identities;
- fragments and their source ranges;
- embeddings and vector records;
- source digests, fragment IDs, manifests, and checkpoint tokens; and
- adapter exceptions and logs.

Fragments and embeddings may reveal source information. Deterministic IDs and
digests may allow equality correlation or guessing attacks against predictable
content. Their `sha256:` representation provides neither encryption nor access
control and should not be used as proof of source ownership.

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

## Trust boundary

A fragment range is a half-open Python code-point range. Standalone contracts
check its width but do not prove that text came from the indicated source.
Projection derives its own fragment text from the supplied document, but the
package still cannot prove that the supplied document or revision was
authoritative or authorized.

Version 0.1.0 provides no built-in encryption, authentication, authorization,
ACL, content filter, persistence security, network security, citation
validation, secret management, vendor guarantee, retrieval workflow, or
user/agent policy. A consuming application must select and assess those
controls for its environment.

The complete public value and collaborator boundaries are listed in the [API
reference](api.md), and deterministic projection behavior is documented in the
[projection guide](projection.md).
