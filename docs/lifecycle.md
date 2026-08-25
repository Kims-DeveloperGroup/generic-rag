# Resource Lifecycle

Version 0.1.0 uses an explicit caller-owned lifecycle. The package defines
collaborator protocols and `Borrowed[T]`; it does not acquire or own provider
resources. See the [API reference](api.md) for the exact port signatures.

## Ownership rule

The caller or provider integration owns every lifecycle decision:

1. Acquire and configure the resource.
2. Establish any required synchronization or exclusive access.
3. Pass the resource explicitly to application code or a future generic
   workflow.
4. Reset, release, close, or shut down the resource according to the provider's
   rules after use.

`generic-rag` does none of those steps implicitly.

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
manager or lifecycle methods.

## Successful scope

The resource that leaves the scope is still the application-owned object:

```python
from generic_rag.ports import Borrowed

provider = object()  # Acquired and owned by the application.

with Borrowed(provider) as active_provider:
    assert active_provider is provider

assert active_provider is provider  # Borrowed did not close or replace it.
```

Application code decides when and how to release the real provider afterward.

## Exceptional scope

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

If a caller needs cleanup after either success or failure, it must arrange that
cleanup around the borrowed scope according to the provider's contract.

## Future workflows

Projection and retrieval orchestration are planned for Issues #3 and #4. Their
collaborators are intended to remain explicitly injected and caller-owned. The
precise future APIs are not part of version 0.1.0, and `Borrowed` must not be
read as a promise that a workflow already exists.

Review the [security and privacy boundary](security-and-privacy.md) before
passing content to a provider implementation.
