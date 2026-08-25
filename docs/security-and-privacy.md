# Security and Privacy

Version 0.1.0 defines in-process values and collaborator boundaries. By itself,
the package performs no persistence, network transmission, provider discovery,
credential loading, telemetry, or background work. Contract objects do retain
caller-supplied values in process memory.

## Caller responsibility

Before constructing a `Document`, `Fragment`, or `RetrievalQuery`, the caller
must perform its own authorization and policy checks. The caller also controls:

- which source content and metadata enter the contracts;
- which provider implementations receive document text, fragment text,
  embeddings, attributes, identifiers, or queries;
- provider account, region, transport, and credential configuration;
- retention, replacement, deletion, backup, and recovery behavior;
- logging, tracing, metrics, redaction, and incident response; and
- whether retrieved fragments are displayed, persisted, or supplied to another
  tool or agent.

Do not place credentials or other secrets in attributes, opaque identifiers, or
checkpoint tokens. These fields deliberately preserve caller input and do not
apply redaction, escaping, access control, or tenant isolation.

## Provider effects

The protocols describe operations that an external implementation may perform.
Calling an embedder, vector index, or lexical retriever can store or transmit
data according to that implementation. Version 0.1.0 supplies no such provider
and does not call one on the caller's behalf.

`Borrowed` does not reduce this responsibility. It only marks the wrapped
resource as caller-owned and does not acquire, close, reset, authenticate, or
synchronize it. See [resource lifecycle](lifecycle.md).

## Sensitive derived data

Treat all of the following as potentially sensitive:

- document and query text;
- ordered attributes and opaque identities;
- fragments and their source ranges;
- embeddings and vector records; and
- projection checkpoint tokens.

Fragments and embeddings are derived data but may reveal information from the
source. Deleting an authoritative source does not automatically delete copies
or derived values held by an application or provider.

## Trust boundary

The package does not own, authenticate, or prove an authoritative source or
revision. A fragment range checks only a half-open code-point width against the
fragment text length; it does not verify the text against a source document.
Ranges and attributes are not a citation or provenance-verification mechanism.

Version 0.1.0 provides no built-in encryption, authentication, authorization,
ACL, content filter, persistence, network security, citation validation, or
vendor guarantee. A consuming application must select and assess those controls
for its environment.

The complete public value and provider boundaries are listed in the
[API reference](api.md).
