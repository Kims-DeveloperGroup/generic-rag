"""Architecture, export, source-index, and clean-import contract tests."""

from __future__ import annotations

import ast
import importlib
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import generic_rag
import generic_rag.contracts as contracts
import generic_rag.errors as errors
import generic_rag.ports as ports

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
_PACKAGE_ROOT = _SOURCE_ROOT / "generic_rag"
_CLEAN_IMPORT_PROBE = _PROJECT_ROOT / "tests" / "support" / "clean_import_probe.py"
_EXPECTED_SOURCES = {
    "generic_rag": "src/generic_rag/__init__.py",
    "generic_rag.contracts": "src/generic_rag/contracts.py",
    "generic_rag.errors": "src/generic_rag/errors.py",
    "generic_rag.ports": "src/generic_rag/ports.py",
}
_EXPECTED_DEPENDENCIES = {
    "generic_rag": set(),
    "generic_rag.contracts": {"generic_rag.errors"},
    "generic_rag.errors": set(),
    "generic_rag.ports": {"generic_rag.contracts"},
}
_EXPECTED_EXPORTS = {
    "generic_rag": (),
    "generic_rag.errors": (
        "GenericRagError",
        "ContractValidationError",
        "CollaborationError",
        "StateCompatibilityError",
    ),
    "generic_rag.contracts": (
        "DocumentKey",
        "DocumentIdentity",
        "Document",
        "FragmentIdentity",
        "Fragment",
        "EmbeddingIdentity",
        "EmbeddingVector",
        "VectorRecord",
        "ProjectionIdentity",
        "ProjectionCheckpoint",
        "ProjectionOutcome",
        "ProjectionReceipt",
        "RetrievalQuery",
        "RetrievalOutcome",
        "RetrievalHit",
        "RetrievalResult",
    ),
    "generic_rag.ports": (
        "Borrowed",
        "Embedder",
        "VectorIndexWriter",
        "VectorIndexReader",
        "LexicalRetriever",
    ),
}


def _production_sources() -> tuple[Path, ...]:
    return tuple(sorted(_PACKAGE_ROOT.rglob("*.py")))


def _module_name(path: Path) -> str:
    relative = path.relative_to(_SOURCE_ROOT)
    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        parts.pop()
    else:
        parts[-1] = path.stem
    return ".".join(parts)


def _syntax_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _resolve_import_from(module_name: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = module_name.split(".")[:-1]
    upward_steps = node.level - 1
    if upward_steps > len(package_parts):
        return ""
    prefix = package_parts[: len(package_parts) - upward_steps]
    if node.module:
        prefix.extend(node.module.split("."))
    return ".".join(prefix)


def _internal_dependencies(module_name: str, tree: ast.Module) -> set[str]:
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "generic_rag" or alias.name.startswith("generic_rag."):
                    dependencies.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imported_module = _resolve_import_from(module_name, node)
            if imported_module == "generic_rag" or imported_module.startswith(
                "generic_rag."
            ):
                dependencies.add(imported_module)
    return dependencies


def _probe_environment() -> dict[str, str]:
    credential_markers = (
        "AUTH",
        "CREDENTIAL",
        "KEY",
        "PASSWORD",
        "SECRET",
        "TOKEN",
    )
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("PYTHON")
        and not any(marker in name.upper() for marker in credential_markers)
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


class PackageBoundaryTests(unittest.TestCase):
    def test_production_module_inventory_is_exact(self) -> None:
        actual = {
            _module_name(path): path.relative_to(_PROJECT_ROOT).as_posix()
            for path in _production_sources()
        }

        self.assertEqual(actual, _EXPECTED_SOURCES)

    def test_supported_exports_are_exact_and_owned(self) -> None:
        modules = {
            "generic_rag": generic_rag,
            "generic_rag.errors": errors,
            "generic_rag.contracts": contracts,
            "generic_rag.ports": ports,
        }

        for module_name, expected_exports in _EXPECTED_EXPORTS.items():
            module = modules[module_name]
            with self.subTest(module=module_name):
                self.assertEqual(module.__all__, expected_exports)
                for name in expected_exports:
                    exported = getattr(module, name)
                    self.assertEqual(exported.__module__, module_name)

        for name in (
            *_EXPECTED_EXPORTS["generic_rag.errors"],
            *_EXPECTED_EXPORTS["generic_rag.contracts"],
            *_EXPECTED_EXPORTS["generic_rag.ports"],
        ):
            with self.subTest(root_reexport=name):
                self.assertFalse(hasattr(generic_rag, name))

    def test_each_supported_module_imports_from_its_owning_path(self) -> None:
        for module_name in _EXPECTED_EXPORTS:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertEqual(module.__name__, module_name)
                self.assertEqual(module.__all__, _EXPECTED_EXPORTS[module_name])

    def test_internal_dependency_graph_is_exact_and_acyclic(self) -> None:
        actual = {
            module_name: _internal_dependencies(
                module_name,
                _syntax_tree(_PROJECT_ROOT / relative_path),
            )
            for module_name, relative_path in _EXPECTED_SOURCES.items()
        }

        self.assertEqual(actual, _EXPECTED_DEPENDENCIES)
        root_tree = _syntax_tree(_PROJECT_ROOT / _EXPECTED_SOURCES["generic_rag"])
        root_imports = [
            node
            for node in ast.walk(root_tree)
            if isinstance(node, ast.Import | ast.ImportFrom)
        ]
        self.assertEqual(root_imports, [])

    def test_production_imports_are_stdlib_or_declared_internal_dependencies(
        self,
    ) -> None:
        allowed_roots = set(sys.stdlib_module_names)
        allowed_roots.update({"__future__", "generic_rag"})
        forbidden_roots = {
            "MySQLdb",
            "mysql",
            "pymysql",
            "sqlalchemy",
            "sqlite3",
            "story_writing_agents",
            "tests",
        }

        for path in _production_sources():
            module_name = _module_name(path)
            tree = _syntax_tree(path)
            for node in ast.walk(tree):
                imported_modules: tuple[str, ...] = ()
                if isinstance(node, ast.Import):
                    imported_modules = tuple(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported_modules = (_resolve_import_from(module_name, node),)
                    for alias in node.names:
                        self.assertNotEqual(
                            alias.name,
                            "*",
                            f"{module_name} uses a wildcard import",
                        )
                        if imported_modules[0].startswith("generic_rag"):
                            self.assertFalse(
                                alias.name.startswith("_"),
                                f"{module_name} imports private name {alias.name}",
                            )

                for imported_module in imported_modules:
                    if not imported_module:
                        continue
                    root = imported_module.partition(".")[0]
                    with self.subTest(module=module_name, imported=imported_module):
                        self.assertNotIn(root, forbidden_roots)
                        self.assertIn(
                            root,
                            allowed_roots,
                            f"{module_name} imports third-party module "
                            f"{imported_module}",
                        )

    def test_production_has_no_any_dynamic_import_or_sys_path_access(self) -> None:
        dynamic_functions = {
            "__import__",
            "import_module",
            "module_from_spec",
            "spec_from_file_location",
        }

        for path in _production_sources():
            module_name = _module_name(path)
            tree = _syntax_tree(path)
            importlib_aliases = {"importlib"}
            sys_aliases = {"sys"}
            direct_dynamic_aliases = set(dynamic_functions)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "importlib":
                            importlib_aliases.add(alias.asname or alias.name)
                        if alias.name == "sys":
                            sys_aliases.add(alias.asname or alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "importlib":
                        for alias in node.names:
                            if alias.name in dynamic_functions:
                                direct_dynamic_aliases.add(alias.asname or alias.name)

            violations: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == "Any":
                    violations.append("typing.Any")
                elif isinstance(node, ast.Attribute) and node.attr == "Any":
                    violations.append("typing.Any")
                elif (
                    isinstance(node, ast.Attribute)
                    and node.attr == "path"
                    and isinstance(node.value, ast.Name)
                    and node.value.id in sys_aliases
                ):
                    violations.append("sys.path")
                elif isinstance(node, ast.Call):
                    function = node.func
                    if (
                        isinstance(function, ast.Name)
                        and function.id in direct_dynamic_aliases
                    ):
                        violations.append(f"dynamic import {function.id}")
                    elif (
                        isinstance(function, ast.Attribute)
                        and isinstance(function.value, ast.Name)
                        and function.value.id in importlib_aliases
                        and function.attr in dynamic_functions
                    ):
                        violations.append(f"dynamic import {function.attr}")

            self.assertEqual(
                violations,
                [],
                f"{module_name} has forbidden boundary operations",
            )

    def test_module_index_has_exact_source_path_parity(self) -> None:
        index_path = _PROJECT_ROOT / "PYTHON_MODULE_INDEX.md"
        entries: dict[str, str] = {}
        current_module: str | None = None
        heading_pattern = re.compile(r"^## \x60(generic_rag(?:\.[a-z_]+)?)\x60$")
        source_pattern = re.compile(r"^- Source: \x60([^ \x60]+)\x60$")

        for line in index_path.read_text(encoding="utf-8").splitlines():
            heading_match = heading_pattern.fullmatch(line)
            if heading_match:
                current_module = heading_match.group(1)
                self.assertNotIn(current_module, entries)
                continue
            source_match = source_pattern.fullmatch(line)
            if source_match and current_module is not None:
                entries[current_module] = source_match.group(1)
                current_module = None

        self.assertEqual(entries, _EXPECTED_SOURCES)

    def test_each_module_imports_in_an_isolated_side_effect_guarded_process(
        self,
    ) -> None:
        environment = _probe_environment()
        for module_name in _EXPECTED_EXPORTS:
            with self.subTest(module=module_name):
                with tempfile.TemporaryDirectory(
                    prefix="generic-rag-import-"
                ) as working_directory:
                    completed = subprocess.run(
                        (
                            sys.executable,
                            "-I",
                            "-B",
                            str(_CLEAN_IMPORT_PROBE),
                            module_name,
                        ),
                        cwd=working_directory,
                        env=environment,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                self.assertEqual(
                    completed.returncode,
                    0,
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
                )


if __name__ == "__main__":
    unittest.main()
