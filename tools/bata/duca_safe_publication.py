from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.bata.duca_evidence_io import verify_content_sha256


RUNTIME_ROOT_REGISTRY_SCHEMA = "duca_admission_v2_1_runtime_root_registry_v1"
_DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)


def _absolute_lexical(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ValueError("runtime path must be absolute")
    return Path(os.path.abspath(os.fspath(candidate)))


def _require_posix_openat() -> None:
    if os.name != "posix":
        raise OSError(
            "authoritative filesystem operations require POSIX openat/mkdirat semantics"
        )


def _safe_component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\x00" in value:
        raise ValueError("unsafe filesystem path component")
    return value


def _open_absolute_directory(path: str | Path) -> int:
    """Open an absolute directory one component at a time without following links."""

    _require_posix_openat()
    candidate = _absolute_lexical(path)
    if candidate.anchor != "/":
        raise ValueError("authoritative POSIX paths must be rooted at /")
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for part in candidate.parts[1:]:
            child = os.open(_safe_component(part), _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise OSError("opened runtime root is not a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory_chain(root_fd: int, parts: Sequence[str], *, create: bool) -> int:
    """Traverse relative directories from a trusted fd; return a new final fd."""

    descriptor = os.dup(root_fd)
    try:
        for raw in parts:
            part = _safe_component(raw)
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
    )


def _assert_path_matches_fd(path: str | Path, descriptor: int) -> os.stat_result:
    """Fail if a pathname no longer names the object held by ``descriptor``."""

    fd_metadata = os.fstat(descriptor)
    path_metadata = os.stat(_absolute_lexical(path), follow_symlinks=False)
    if not _same_identity(fd_metadata, path_metadata):
        raise OSError("registered pathname identity changed during operation")
    return fd_metadata


def require_no_symlink_ancestors(
    path: str | Path, *, include_leaf: bool = True
) -> None:
    """Diagnostic lexical walk; authoritative operations additionally use openat."""

    candidate = _absolute_lexical(path)
    anchor = Path(candidate.anchor)
    current = anchor
    parts = candidate.parts[1:] if candidate.anchor else candidate.parts
    limit = len(parts) if include_leaf else max(0, len(parts) - 1)
    for part in parts[:limit]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"FILESYSTEM_POLICY_FAILED: symlink ancestor {current}")


def _runtime_root_registry_paths(payload: Mapping[str, Any]) -> list[Path]:
    if set(payload) != {"schema", "allowlisted_base_roots", "content_sha256"}:
        raise ValueError("runtime root registry is not closed-world")
    verify_content_sha256(payload)
    if payload.get("schema") != RUNTIME_ROOT_REGISTRY_SCHEMA:
        raise ValueError("invalid runtime root registry schema")
    roots = payload.get("allowlisted_base_roots")
    if not isinstance(roots, Sequence) or isinstance(roots, (str, bytes)) or not roots:
        raise ValueError("runtime root registry must list at least one base root")
    canonical: list[Path] = []
    for raw in roots:
        if not isinstance(raw, str):
            raise ValueError("allowlisted base roots must be strings")
        path = _absolute_lexical(raw)
        canonical.append(path)
    if len(canonical) != len(set(canonical)):
        raise ValueError("runtime root registry contains duplicate roots")
    return canonical


def validate_runtime_root_registry(payload: Mapping[str, Any]) -> list[Path]:
    """Validate the closed registry and bind every root through a no-follow fd."""

    roots = _runtime_root_registry_paths(payload)
    for path in roots:
        descriptor = _open_absolute_directory(path)
        try:
            _assert_path_matches_fd(path, descriptor)
        except FileNotFoundError as exc:
            raise ValueError(f"allowlisted base root does not exist: {path}") from exc
        finally:
            os.close(descriptor)
    return roots


def _open_registered_base(
    target: str | Path, *, root_registry: Mapping[str, Any]
) -> tuple[Path, Path, Path, int]:
    """Select and open the one registered base used for an authoritative operation."""

    target_path = _absolute_lexical(target)
    bases = _runtime_root_registry_paths(root_registry)
    matches: list[tuple[Path, Path]] = []
    for base in bases:
        try:
            relative = target_path.relative_to(base)
        except ValueError:
            continue
        matches.append((base, relative))
    if len(matches) != 1:
        raise ValueError("target must lie under exactly one allowlisted base root")
    base, relative = matches[0]
    descriptor = _open_absolute_directory(base)
    try:
        _assert_path_matches_fd(base, descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return target_path, base, relative, descriptor


def select_allowlisted_base(
    target: str | Path, *, root_registry: Mapping[str, Any]
) -> tuple[Path, Path]:
    target_path, base, _relative, descriptor = _open_registered_base(
        target, root_registry=root_registry
    )
    os.close(descriptor)
    return target_path, base


def create_fresh_run_root(
    target: str | Path,
    *,
    root_registry: Mapping[str, Any],
) -> Path:
    target_path, base, relative, base_fd = _open_registered_base(
        target, root_registry=root_registry
    )
    if len(relative.parts) != 1:
        os.close(base_fd)
        raise ValueError(
            "run root must be one fresh leaf directly below its registered base"
        )
    leaf = _safe_component(relative.parts[0])
    try:
        os.mkdir(leaf, mode=0o700, dir_fd=base_fd)
        run_fd = os.open(leaf, _DIRECTORY_FLAGS, dir_fd=base_fd)
        try:
            metadata = os.fstat(run_fd)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise OSError("fresh run root mode/type verification failed")
            _assert_path_matches_fd(target_path, run_fd)
        finally:
            os.close(run_fd)
        os.fsync(base_fd)
    finally:
        os.close(base_fd)
    return target_path


def read_file_beneath_allowlisted_roots(
    path: str | Path,
    *,
    root_registry: Mapping[str, Any],
) -> tuple[bytes, os.stat_result]:
    """Read and attest one regular file through a single no-follow descriptor."""

    target_path, base, relative, base_fd = _open_registered_base(
        path, root_registry=root_registry
    )
    if len(relative.parts) < 1:
        os.close(base_fd)
        raise ValueError("allowlisted input must be a file below its base root")
    try:
        parent_fd = _open_directory_chain(base_fd, relative.parts[:-1], create=False)
    finally:
        os.close(base_fd)
    try:
        leaf = _safe_component(relative.parts[-1])
        descriptor = os.open(
            leaf,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("allowlisted input is not a regular file")
            payload = _read_all(descriptor)
            final_metadata = os.fstat(descriptor)
            if (
                final_metadata.st_dev != metadata.st_dev
                or final_metadata.st_ino != metadata.st_ino
                or final_metadata.st_mode != metadata.st_mode
                or final_metadata.st_size != metadata.st_size
                or final_metadata.st_mtime_ns != metadata.st_mtime_ns
            ):
                raise OSError("allowlisted input metadata changed during read")
            path_metadata = os.stat(target_path, follow_symlinks=False)
            if not _same_identity(final_metadata, path_metadata):
                raise OSError("allowlisted input pathname changed during read")
            return payload, metadata
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def _open_registered_run_root(
    run_root: str | Path, *, root_registry: Mapping[str, Any]
) -> tuple[Path, int]:
    root, _base, relative, base_fd = _open_registered_base(
        run_root, root_registry=root_registry
    )
    if len(relative.parts) != 1:
        os.close(base_fd)
        raise ValueError("run root must be one leaf directly below its registered base")
    try:
        root_fd = _open_directory_chain(base_fd, relative.parts, create=False)
    finally:
        os.close(base_fd)
    try:
        metadata = _assert_path_matches_fd(root, root_fd)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise OSError("registered run root must be a mode-0700 directory")
    except BaseException:
        os.close(root_fd)
        raise
    return root, root_fd


def read_file_without_symlinks(path: str | Path) -> tuple[bytes, os.stat_result]:
    """Read one absolute/working-directory-relative file through no-follow dirfds."""

    target = _absolute_lexical(path)
    parent_fd = _open_absolute_directory(target.parent)
    try:
        descriptor = os.open(
            _safe_component(target.name),
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("authoritative input is not a regular file")
            payload = _read_all(descriptor)
            final_metadata = os.fstat(descriptor)
            if (
                not _same_identity(metadata, final_metadata)
                or metadata.st_size != final_metadata.st_size
                or metadata.st_mtime_ns != final_metadata.st_mtime_ns
            ):
                raise OSError("authoritative input metadata changed during read")
            path_metadata = os.stat(target, follow_symlinks=False)
            if not _same_identity(final_metadata, path_metadata):
                raise OSError("authoritative input pathname changed during read")
            return payload, metadata
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def atomic_publication_self_test(
    run_root: str | Path, *, root_registry: Mapping[str, Any]
) -> dict[str, Any]:
    root, root_fd = _open_registered_run_root(run_root, root_registry=root_registry)
    payload = bytes(range(256)) * 16
    temporary = f".duca-v21-atomic-selftest.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    final = f".duca-v21-atomic-selftest.{os.getpid()}.{secrets.token_hex(8)}.final"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_fd,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(
            temporary,
            final,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
            follow_symlinks=False,
        )
        temp_fd = os.open(
            temporary,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        try:
            final_fd = os.open(
                final,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_fd,
            )
            try:
                temp_stat = os.fstat(temp_fd)
                final_stat = os.fstat(final_fd)
                parent_stat = os.fstat(root_fd)
                if not (temp_stat.st_dev == final_stat.st_dev == parent_stat.st_dev):
                    raise OSError("cross-filesystem atomic publication is forbidden")
                if temp_stat.st_ino != final_stat.st_ino or temp_stat.st_nlink != 2:
                    raise OSError("hard-link publication semantics are unavailable")
                if _read_all(temp_fd) != payload or _read_all(final_fd) != payload:
                    raise OSError("atomic publication self-test payload drift")
                _assert_path_matches_fd(root, root_fd)
                os.fsync(root_fd)
                return {
                    "status": "PASSED",
                    "payload_size": len(payload),
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                    "device": int(temp_stat.st_dev),
                    "inode": int(temp_stat.st_ino),
                    "link_count": int(temp_stat.st_nlink),
                }
            finally:
                os.close(final_fd)
        finally:
            os.close(temp_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for name in (final, temporary):
            try:
                os.unlink(name, dir_fd=root_fd)
            except FileNotFoundError:
                pass
        os.close(root_fd)


def publish_json_under_run_root(
    target: str | Path,
    payload: Mapping[str, Any],
    *,
    run_root: str | Path,
    root_registry: Mapping[str, Any],
) -> Path:
    root, root_fd = _open_registered_run_root(run_root, root_registry=root_registry)
    target_path = _absolute_lexical(target)
    try:
        relative = target_path.relative_to(root)
    except ValueError as exc:
        os.close(root_fd)
        raise ValueError("publication target escapes the run root") from exc
    if len(relative.parts) < 1 or target_path.suffix != ".json":
        os.close(root_fd)
        raise ValueError("publication target must be a JSON file below the run root")
    try:
        safe_parts = tuple(_safe_component(part) for part in relative.parts)
    except BaseException:
        os.close(root_fd)
        raise
    try:
        verify_content_sha256(payload)
        encoded = (
            json.dumps(
                dict(payload),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        parent_fd = _open_directory_chain(root_fd, safe_parts[:-1], create=True)
    except BaseException:
        os.close(root_fd)
        raise
    leaf = safe_parts[-1]
    temporary = f".{leaf}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(
            temporary,
            leaf,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.fsync(parent_fd)
        final_fd = os.open(
            leaf,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            final_metadata = os.fstat(final_fd)
            if not stat.S_ISREG(final_metadata.st_mode):
                raise OSError("published JSON is not a regular file")
            if stat.S_IMODE(final_metadata.st_mode) != 0o600:
                raise OSError("published JSON mode drifted")
            if _read_all(final_fd) != encoded:
                raise OSError("published JSON payload drifted")
        finally:
            os.close(final_fd)
        _assert_path_matches_fd(root, root_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)
        os.close(root_fd)
    return target_path
