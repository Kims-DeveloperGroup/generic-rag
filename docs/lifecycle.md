# Resource Lifecycle

Version 0.1.0 uses an explicit caller-owned lifecycle. The package defines
collaborator protocols and `Borrowed[T]`; it does not acquire, configure,
discover, persist, synchronize, or release provider resources. Projection calls
borrowed collaborators only during an explicit workflow invocation. See the
[API reference](api.md) for exact port signatures.

## Ownership rule

The caller or provider integration owns every lifecycle decision:

1. Acquire and configure the embedder, vector index, projection-state store,
   and any required synchronization.
2. Authorize the complete source set and construct a bounded
   `ProjectionRequest`.
3. Wrap the application-owned collaborators in `Borrowed` and call
   `project_documents` for a compatible incremental update, or deliberately
   call `rebuild_projection` for bootstrap or destructive recovery.
4. On complete success, publish the returned manifest through caller-owned
   persistence and synchronization.
5. Release, close, or shut down the real resources according to their provider
   contracts.

`generic-rag` performs none of the acquisition, manifest persistence,
publication, synchronization, or release steps implicitly. It provides no
transaction across collaborators and the manifest store.

## `Borrowed[T]`

`Borrowed` is a frozen, slotted context manager that retains a reference. It
has exactly these context semantics:

- `__enter__` returns the exact wrapped resource.
- `__exit__` returns `False`, so it never suppresses an exception.
- It never calls the resource's `__enter__`, `__exit__`, `close`, `shutdown`,
  reset, acquisition, release, or synchronization operations.
- It does not translate a resource failure into `CollaborationError`.
- It provides no locking, thread safety, asynchronous context management, or
  provider health checks.

These guarantees still apply if the wrapped provider defines its own context
manager or lifecycle methods. Projection itself translates ordinary method
failures into `ProjectionOperationError`; that workflow behavior does not
change `Borrowed` semantics.

## Successful borrowed scope

The resource that leaves the scope is still the application-owned object:

```python
from generic_rag.ports import Borrowed

provider = object()  # Acquired and owned by the application.

with Borrowed(provider) as active_provider:
    assert active_provider is provider

assert active_provider is provider  # Borrowed did not close or replace it.
```

Application code decides when and how to release the real provider afterward.

## Exceptional borrowed scope

An exception leaves the borrowed scope unchanged and unsuppressed:

```python
from generic_rag.ports import Borrowed

provider = object()
failure = RuntimeError("provider failed")

try:
    with Borrowed(provider) as active_provider:
        assert active_provider is provider
        raise failure
except RuntimeError as caught:
    assert caught is failure
```

If a caller needs cleanup after success or failure, it must arrange that
cleanup around the borrowed scope according to the provider's contract.

## Projection call scope

`project_documents` and `rebuild_projection` receive exact `Borrowed` wrappers.
They enter only those no-op wrappers; they never enter, close, or shut down the
underlying adapter objects.

The caller must keep each adapter alive for the complete synchronous call.
Providers that require sessions, transactions, locks, or thread affinity must
be prepared before wrapping and must remain valid until the call returns or
raises.

An incremental call may perform multiple ordered document mutations. A rebuild
performs a corpus reset before its document replacements. The package does not
roll back or retry earlier effects if a later operation fails. The
`ProjectionOperationError.receipt` reports completed document operations but
never supplies a checkpoint for partial work. Keep partially updated state out
of service and recover under application-owned coordination.

The caller should persist `ProjectionResult.manifest` only after complete
success. The package does not retain it, and a vector index without the matching
published manifest cannot be used safely by the next incremental operation.
See the [projection guide](projection.md) for the state matrix and destructive
rebuild ordering.

## Retrieval lifecycle is not implemented

The query and reader contracts do not create a package retrieval workflow.
Retrieval and composition remain planned for Issue #4, and no user or agent
query lifecycle is implied by the current projection API.

Review the [security and privacy boundary](security-and-privacy.md) before
passing content to any adapter implementation.
