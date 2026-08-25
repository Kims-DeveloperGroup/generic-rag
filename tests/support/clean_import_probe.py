"""Probe one installed generic_rag module in an isolated interpreter."""

from __future__ import annotations

import argparse
import importlib
import multiprocessing
import os
import socket
import subprocess
import sys
import threading
from contextlib import ExitStack
from pathlib import Path
from types import FrameType, ModuleType
from typing import NoReturn, cast
from unittest.mock import patch

_PACKAGE_ROOT = "generic_rag"
_FORBIDDEN_AUDIT_EVENTS: list[str] = []
_AUDIT_HOOK_ACTIVE = False


def _called_from_package() -> bool:
    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        module_name = frame.f_globals.get("__name__")
        if isinstance(module_name, str) and (
            module_name == _PACKAGE_ROOT or module_name.startswith(f"{_PACKAGE_ROOT}.")
        ):
            return True
        frame = frame.f_back
    return False


def _is_loader_source_read(arguments: tuple[object, ...]) -> bool:
    if not arguments:
        return False
    raw_path = arguments[0]
    if not isinstance(raw_path, str | bytes | os.PathLike):
        return False
    path = Path(os.fsdecode(raw_path))
    if path.suffix not in {".py", ".pyc"}:
        return False

    mode = arguments[1] if len(arguments) > 1 else None
    if isinstance(mode, str) and any(marker in mode for marker in "wax+"):
        return False
    flags = arguments[2] if len(arguments) > 2 else None
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    if isinstance(flags, int) and flags & write_flags:
        return False

    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        module_name = frame.f_globals.get("__name__")
        if (
            module_name
            in {
                "_frozen_importlib_external",
                "importlib._bootstrap_external",
            }
            and frame.f_code.co_name == "get_data"
        ):
            return True
        frame = frame.f_back
    return False


def _audit_hook(event: str, arguments: tuple[object, ...]) -> None:
    global _AUDIT_HOOK_ACTIVE

    if _AUDIT_HOOK_ACTIVE:
        return
    _AUDIT_HOOK_ACTIVE = True
    try:
        if not _called_from_package():
            return
        if event == "open" and _is_loader_source_read(arguments):
            return
        forbidden = (
            event == "open"
            or event.startswith("socket.")
            or event.startswith("subprocess.")
            or event == "os.system"
            or event.startswith("os.spawn")
        )
        if forbidden:
            _FORBIDDEN_AUDIT_EVENTS.append(event)
            raise RuntimeError(f"forbidden import-time operation: {event}")
    finally:
        _AUDIT_HOOK_ACTIVE = False


def _forbidden_operation(*arguments: object, **keywords: object) -> NoReturn:
    del arguments, keywords
    raise RuntimeError("forbidden import-time process, network, or thread operation")


def _import_with_guards(module_name: str) -> ModuleType:
    sys.addaudithook(_audit_hook)
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(threading.Thread, "start", _forbidden_operation)
        )
        stack.enter_context(
            patch.object(multiprocessing.Process, "start", _forbidden_operation)
        )
        stack.enter_context(patch.object(socket, "socket", _forbidden_operation))
        stack.enter_context(
            patch.object(socket, "create_connection", _forbidden_operation)
        )
        stack.enter_context(patch.object(subprocess, "Popen", _forbidden_operation))
        stack.enter_context(patch.object(os, "system", _forbidden_operation))
        stack.enter_context(patch.object(os, "popen", _forbidden_operation))
        for name in (
            "spawnl",
            "spawnle",
            "spawnlp",
            "spawnlpe",
            "spawnv",
            "spawnve",
            "spawnvp",
            "spawnvpe",
        ):
            if hasattr(os, name):
                stack.enter_context(patch.object(os, name, _forbidden_operation))
        imported = importlib.import_module(module_name)

    if _FORBIDDEN_AUDIT_EVENTS:
        joined = ", ".join(_FORBIDDEN_AUDIT_EVENTS)
        raise AssertionError(f"package import attempted forbidden events: {joined}")
    return imported


def _assert_no_external_imports(before: set[str], after: set[str]) -> None:
    allowed_roots = set(sys.stdlib_module_names)
    allowed_roots.update({"builtins", _PACKAGE_ROOT})
    unexpected = sorted(
        module_name
        for module_name in after - before
        if module_name.partition(".")[0] not in allowed_roots
    )
    if unexpected:
        raise AssertionError(
            "package import loaded non-stdlib modules: " + ", ".join(unexpected)
        )


def _assert_forbidden_paths_absent(paths: list[str]) -> None:
    resolved_entries = [Path(entry).resolve() for entry in sys.path if entry]
    for raw_path in paths:
        forbidden = Path(raw_path).resolve()
        for entry in resolved_entries:
            try:
                entry.relative_to(forbidden)
            except ValueError:
                continue
            raise AssertionError(
                f"forbidden checkout path leaked into sys.path: {entry}"
            )


def _bootstrap_source_root(raw_source_root: str | None) -> Path | None:
    if raw_source_root is None:
        return None
    source_root = Path(raw_source_root).resolve(strict=True)
    if not source_root.is_dir():
        raise AssertionError(f"source root is not a directory: {source_root}")
    sys.path.insert(0, os.fspath(source_root))
    return source_root


def _assert_import_origin(imported: ModuleType, source_root: Path | None) -> None:
    if source_root is None:
        return
    raw_origin = getattr(imported, "__file__", None)
    if not isinstance(raw_origin, str):
        raise AssertionError(f"source import has no file origin: {imported.__name__}")
    origin = Path(raw_origin).resolve(strict=True)
    expected_package = source_root / _PACKAGE_ROOT
    try:
        origin.relative_to(expected_package)
    except ValueError:
        raise AssertionError(
            f"{imported.__name__} came from {origin}, not {expected_package}"
        ) from None


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("module")
    parser.add_argument(
        "--forbid-path",
        action="append",
        default=[],
        help="Fail when this path or one of its children occurs on sys.path.",
    )
    parser.add_argument(
        "--source-root",
        help="Explicit resolved src root for an isolated source-tree probe.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    module_name = cast(str, arguments.module)
    forbidden_paths = cast(list[str], arguments.forbid_path)
    raw_source_root = cast(str | None, arguments.source_root)
    if module_name != _PACKAGE_ROOT and not module_name.startswith(f"{_PACKAGE_ROOT}."):
        raise ValueError(f"probe is restricted to {_PACKAGE_ROOT} modules")

    _assert_forbidden_paths_absent(forbidden_paths)
    source_root = _bootstrap_source_root(raw_source_root)
    before = set(sys.modules)
    imported = _import_with_guards(module_name)
    after = set(sys.modules)
    _assert_no_external_imports(before, after)
    _assert_import_origin(imported, source_root)

    if imported.__name__ != module_name:
        raise AssertionError(f"requested {module_name}, imported {imported.__name__}")
    if module_name == _PACKAGE_ROOT:
        if getattr(imported, "__all__", None) != ():
            raise AssertionError("generic_rag root __all__ must be empty")
        loaded_submodules = sorted(
            name for name in after if name.startswith(f"{_PACKAGE_ROOT}.")
        )
        if loaded_submodules:
            raise AssertionError(
                "root import loaded submodules: " + ", ".join(loaded_submodules)
            )

    print(f"clean import passed: {module_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
