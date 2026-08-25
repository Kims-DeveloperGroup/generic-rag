"""Validate generic-rag wheel and source-distribution artifacts."""

from __future__ import annotations

import argparse
import stat
import tarfile
import zipfile
from email import message_from_bytes
from email.message import Message
from pathlib import Path, PurePosixPath
from typing import cast

_PACKAGE_FILES = {
    "generic_rag/__init__.py",
    "generic_rag/contracts.py",
    "generic_rag/errors.py",
    "generic_rag/ports.py",
    "generic_rag/projection.py",
    "generic_rag/projection_integrity.py",
    "generic_rag/retrieval.py",
}
_PACKAGE_DATA = {"generic_rag/py.typed"}


class ArtifactVerificationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactVerificationError(message)


def _safe_archive_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    _require(bool(path.parts), "archive contains an empty member name")
    _require(not path.is_absolute(), f"archive member is absolute: {name}")
    _require(".." not in path.parts, f"archive member traverses upward: {name}")
    _require("\\" not in name, f"archive member uses a backslash: {name}")
    return path


def _required_header(metadata: Message, name: str) -> str:
    value = metadata.get(name)
    _require(value is not None and bool(value.strip()), f"missing {name} metadata")
    assert value is not None
    return value


def _assert_metadata(metadata: Message) -> str:
    _require(_required_header(metadata, "Name") == "generic-rag", "wrong Name")
    version = _required_header(metadata, "Version")
    _require(
        _required_header(metadata, "Requires-Python") == ">=3.11",
        "Requires-Python must be exactly >=3.11",
    )
    requirements = metadata.get_all("Requires-Dist")
    _require(
        not requirements,
        "runtime dependencies leaked into metadata: " + ", ".join(requirements or []),
    )
    return version


def _assert_no_shipped_tests(names: set[str], artifact: str) -> None:
    shipped = sorted(
        name
        for name in names
        if any(
            part.lower() in {"test", "tests"} or part.lower().startswith("test_")
            for part in PurePosixPath(name).parts
        )
    )
    _require(
        not shipped,
        f"{artifact} ships test files: {', '.join(shipped)}",
    )


def _verify_wheel(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        entries = archive.infolist()
        names = {entry.filename.rstrip("/") for entry in entries}
        for entry in entries:
            _safe_archive_name(entry.filename)
            file_type = (entry.external_attr >> 16) & 0o170000
            _require(
                file_type != stat.S_IFLNK,
                f"wheel contains a symlink: {entry.filename}",
            )

        _assert_no_shipped_tests(names, "wheel")
        _require(
            _PACKAGE_FILES <= names,
            "wheel is missing package modules: "
            + ", ".join(sorted(_PACKAGE_FILES - names)),
        )
        _require(
            _PACKAGE_DATA <= names,
            "wheel is missing py.typed",
        )
        shipped_python = {
            name
            for name in names
            if name.startswith("generic_rag/") and name.endswith(".py")
        }
        _require(
            shipped_python == _PACKAGE_FILES,
            "wheel has an unexpected production module set: "
            + ", ".join(sorted(shipped_python)),
        )
        _require(
            not any("__pycache__" in PurePosixPath(name).parts for name in names),
            "wheel contains bytecode caches",
        )
        _require(
            not any(name.endswith((".so", ".pyd", ".dylib", ".pyc")) for name in names),
            "wheel is not a clean pure-Python artifact",
        )

        metadata_names = sorted(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        _require(
            len(metadata_names) == 1,
            "wheel must contain exactly one METADATA file",
        )
        metadata = message_from_bytes(archive.read(metadata_names[0]))
        version = _assert_metadata(metadata)

        allowed_roots = {
            "generic_rag",
            metadata_names[0].partition("/")[0],
        }
        unexpected_roots = sorted(
            {
                PurePosixPath(name).parts[0]
                for name in names
                if PurePosixPath(name).parts[0] not in allowed_roots
            }
        )
        _require(
            not unexpected_roots,
            "wheel contains unexpected roots: " + ", ".join(unexpected_roots),
        )

    expected_name = f"generic_rag-{version}-py3-none-any.whl"
    _require(
        wheel.name == expected_name,
        f"wheel filename must be {expected_name}, got {wheel.name}",
    )
    return version


def _verify_sdist(sdist: Path, expected_version: str) -> None:
    expected_root = f"generic_rag-{expected_version}"
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = archive.getmembers()
        names = {member.name.rstrip("/") for member in members}
        for member in members:
            path = _safe_archive_name(member.name)
            _require(
                path.parts[0] == expected_root,
                f"sdist has an unexpected root: {member.name}",
            )
            _require(
                not member.issym() and not member.islnk(),
                f"sdist contains a link: {member.name}",
            )
            _require(
                member.isfile() or member.isdir(),
                f"sdist contains a special file: {member.name}",
            )

        required = {
            f"{expected_root}/README.md",
            f"{expected_root}/pyproject.toml",
            f"{expected_root}/PKG-INFO",
            *{
                f"{expected_root}/src/{package_file}"
                for package_file in _PACKAGE_FILES | _PACKAGE_DATA
            },
        }
        _require(
            required <= names,
            "sdist is missing required files: " + ", ".join(sorted(required - names)),
        )

        metadata_member = archive.getmember(f"{expected_root}/PKG-INFO")
        metadata_file = archive.extractfile(metadata_member)
        _require(metadata_file is not None, "sdist PKG-INFO is not readable")
        assert metadata_file is not None
        with metadata_file:
            metadata = message_from_bytes(metadata_file.read())
        _require(
            _assert_metadata(metadata) == expected_version,
            "wheel and sdist versions differ",
        )

    expected_name = f"{expected_root}.tar.gz"
    _require(
        sdist.name == expected_name,
        f"sdist filename must be {expected_name}, got {sdist.name}",
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_directory", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    dist_directory = cast(Path, arguments.dist_directory)
    _require(
        dist_directory.is_dir(),
        f"distribution directory does not exist: {dist_directory}",
    )
    wheel_files = sorted(dist_directory.glob("*.whl"))
    sdist_files = sorted(dist_directory.glob("*.tar.gz"))
    _require(len(wheel_files) == 1, "expected exactly one wheel")
    _require(len(sdist_files) == 1, "expected exactly one sdist")
    unexpected = sorted(
        path.name
        for path in dist_directory.iterdir()
        if path.is_file() and path not in {*wheel_files, *sdist_files}
    )
    _require(
        not unexpected,
        "distribution directory contains unexpected files: " + ", ".join(unexpected),
    )

    version = _verify_wheel(wheel_files[0])
    _verify_sdist(sdist_files[0], version)
    print(f"verified wheel: {wheel_files[0]}")
    print(f"verified sdist: {sdist_files[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
