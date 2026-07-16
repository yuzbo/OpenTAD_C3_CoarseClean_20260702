"""Descriptor-bound filesystem and Python-source identity for formal r2 runs.

Formal ChronoTransport artifacts are security boundaries, not convenience
files.  This module therefore never validates one pathname and later reopens
it.  Every input is opened component-by-component with ``openat`` and
``O_NOFOLLOW``; bytes, hashes, JSON/torch deserialization, and retained media
handles all refer to that same open file description.  Publications use a
directory descriptor and a no-clobber hard-link commit.

The implementation intentionally fails closed outside a POSIX runtime with
the required descriptor APIs.  Formal r2 execution is registered for the
N16R4 Linux/Slurm environment; a weaker Windows pathname fallback would make
the audit claim false.
"""

from __future__ import annotations

import ast
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import secrets
import stat
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


FILESYSTEM_IDENTITY_SCHEMA = "chronotransport-r2-descriptor-file-v1"
PYTHON_RUNTIME_IDENTITY_SCHEMA = "chronotransport-r2-python-runtime-v1"


def _require_descriptor_runtime() -> None:
    required_flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    missing = [name for name in required_flags if not hasattr(os, name)]
    required_dir_fd = (os.open, os.mkdir, os.stat, os.unlink)
    unsupported = [
        function.__name__
        for function in required_dir_fd
        if function not in os.supports_dir_fd
    ]
    if os.name != "posix" or missing or unsupported or os.link not in os.supports_dir_fd:
        raise RuntimeError(
            "formal ChronoTransport requires POSIX openat/O_NOFOLLOW/dir_fd; "
            f"os={os.name!r}, missing_flags={missing}, unsupported_dir_fd={unsupported}"
        )


def _absolute_lexical_path(path: str | os.PathLike[str], *, label: str) -> str:
    """Return one canonical absolute POSIX path without resolving symlinks."""

    _require_descriptor_runtime()
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError(f"{label} must be a non-empty text path")
    if "\\" in raw:
        raise ValueError(f"{label} must use POSIX path separators")
    if not raw.startswith("/"):
        raw_parts = PurePosixPath(raw).parts
        if any(part in ("", ".", "..") for part in raw_parts):
            raise ValueError(f"{label} contains a non-canonical path component")
        raw = os.path.join(os.getcwd(), raw)
    if raw != "/" and ("//" in raw or raw.endswith("/")):
        raise ValueError(f"{label} must be a canonical absolute path")
    parts = raw.split("/")[1:]
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{label} contains a non-canonical path component")
    if os.path.normpath(raw) != raw:
        raise ValueError(f"{label} must be lexically normalized")
    return raw


def _require_lexical_containment(path: str, root: str, *, label: str) -> None:
    try:
        common = os.path.commonpath((path, root))
    except ValueError as error:
        raise ValueError(f"{label} is not comparable with its allowed root") from error
    if common != root or path == root:
        raise ValueError(f"{label} escapes or aliases its allowed root")


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _file_flags() -> int:
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


def _open_directory_component(parent: int, component: str, *, label: str) -> int:
    try:
        return os.open(component, _directory_flags(), dir_fd=parent)
    except FileNotFoundError:
        raise
    except OSError as error:
        try:
            metadata = os.stat(component, dir_fd=parent, follow_symlinks=False)
        except OSError:
            metadata = None
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} contains a symlink component: {component}") from error
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ValueError(
                f"{label} contains a non-directory or replaced component: {component}"
            ) from error
        raise


def _open_regular_component(parent: int, leaf: str, *, label: str) -> int:
    try:
        return os.open(leaf, _file_flags(), dir_fd=parent)
    except FileNotFoundError:
        raise
    except OSError as error:
        try:
            metadata = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        except OSError:
            metadata = None
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} must not be a symlink") from error
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ValueError(f"{label} must be a regular non-symlink file") from error
        raise


def _open_directory_absolute(path: str, *, label: str, create: bool = False) -> int:
    """Open an absolute directory by walking every component from ``/``."""

    exact = _absolute_lexical_path(path, label=label)
    descriptor = os.open("/", _directory_flags())
    try:
        components = () if exact == "/" else exact.split("/")[1:]
        for component in components:
            try:
                child = _open_directory_component(
                    descriptor, component, label=label
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                child = _open_directory_component(
                    descriptor, component, label=label
                )
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise ValueError(f"{label} component is not a directory: {component}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


@dataclass(frozen=True)
class DescriptorIdentity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> "DescriptorIdentity":
        return cls(
            device=int(metadata.st_dev),
            inode=int(metadata.st_ino),
            mode=int(metadata.st_mode),
            size=int(metadata.st_size),
            mtime_ns=int(metadata.st_mtime_ns),
            ctime_ns=int(metadata.st_ctime_ns),
        )

    def as_dict(self) -> dict[str, int | str]:
        return {
            "schema": FILESYSTEM_IDENTITY_SCHEMA,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


class BoundRegularFile:
    """One retained regular-file descriptor with stable same-fd reads."""

    def __init__(self, *, path: str, descriptor: int, label: str) -> None:
        self.path = path
        self.label = label
        self._descriptor = descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            self._descriptor = -1
            raise ValueError(f"{label} must be a regular file")
        self.identity = DescriptorIdentity.from_stat(metadata)

    @property
    def descriptor(self) -> int:
        if self._descriptor < 0:
            raise RuntimeError(f"{self.label} descriptor is closed")
        return self._descriptor

    @property
    def proc_path(self) -> str:
        """A decoder path that reuses this open file description on Linux."""

        proc = f"/proc/self/fd/{self.descriptor}"
        if not os.path.exists("/proc/self/fd"):
            raise RuntimeError("formal retained media descriptors require /proc/self/fd")
        return proc

    def _assert_stable(self, before: DescriptorIdentity) -> None:
        after = DescriptorIdentity.from_stat(os.fstat(self.descriptor))
        if after != before:
            raise RuntimeError(f"{self.label} inode mutated while its bytes were consumed")

    def assert_stable(self) -> None:
        if DescriptorIdentity.from_stat(os.fstat(self.descriptor)) != self.identity:
            raise RuntimeError(f"{self.label} inode changed after descriptor binding")

    def read_bytes(self) -> bytes:
        before = DescriptorIdentity.from_stat(os.fstat(self.descriptor))
        if before != self.identity:
            raise RuntimeError(f"{self.label} inode changed after descriptor binding")
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        blocks: list[bytes] = []
        while True:
            block = os.read(self.descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        payload = b"".join(blocks)
        self._assert_stable(before)
        if len(payload) != before.size:
            raise RuntimeError(f"{self.label} byte count differs from bound inode size")
        return payload

    def bytes_and_sha256(self) -> tuple[bytes, str]:
        payload = self.read_bytes()
        return payload, hashlib.sha256(payload).hexdigest()

    def size_and_sha256(self) -> tuple[int, str]:
        """Hash this same descriptor without materializing a large media file."""

        before = DescriptorIdentity.from_stat(os.fstat(self.descriptor))
        if before != self.identity:
            raise RuntimeError(f"{self.label} inode changed after descriptor binding")
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(self.descriptor, 1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
        self._assert_stable(before)
        if size != before.size:
            raise RuntimeError(f"{self.label} byte count differs from bound inode size")
        return size, digest.hexdigest()

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def __enter__(self) -> "BoundRegularFile":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - defensive leak guard
        try:
            self.close()
        except Exception:
            pass


class BoundDirectory:
    """One retained directory inode used as the root for relative openat walks."""

    def __init__(self, *, path: str, descriptor: int, label: str) -> None:
        self.path = path
        self.label = label
        self._descriptor = descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            os.close(descriptor)
            self._descriptor = -1
            raise ValueError(f"{label} must be a directory")
        self.identity = DescriptorIdentity.from_stat(metadata)

    @property
    def descriptor(self) -> int:
        if self._descriptor < 0:
            raise RuntimeError(f"{self.label} directory descriptor is closed")
        return self._descriptor

    @property
    def proc_path(self) -> str:
        proc = f"/proc/self/fd/{self.descriptor}"
        if not os.path.exists("/proc/self/fd"):
            raise RuntimeError("formal bound directories require /proc/self/fd")
        return proc

    @staticmethod
    def _relative_parts(relative: str | os.PathLike[str], *, label: str) -> tuple[str, ...]:
        raw = os.fspath(relative)
        if not isinstance(raw, str) or not raw or "\\" in raw or raw.startswith("/"):
            raise ValueError(f"{label} must be a non-empty relative POSIX path")
        pure = PurePosixPath(raw)
        if (
            any(part in ("", ".", "..") for part in pure.parts)
            or pure.as_posix() != raw
        ):
            raise ValueError(f"{label} must be a canonical relative POSIX path")
        return tuple(pure.parts)

    def open_regular(
        self, relative: str | os.PathLike[str], *, label: str
    ) -> BoundRegularFile:
        parts = self._relative_parts(relative, label=label)
        directory = os.dup(self.descriptor)
        try:
            for component in parts[:-1]:
                child = _open_directory_component(
                    directory, component, label=label
                )
                os.close(directory)
                directory = child
            descriptor = _open_regular_component(
                directory, parts[-1], label=label
            )
        finally:
            os.close(directory)
        return BoundRegularFile(
            path=f"{self.path.rstrip('/')}/{PurePosixPath(*parts).as_posix()}",
            descriptor=descriptor,
            label=label,
        )

    def read_bytes(
        self, relative: str | os.PathLike[str], *, label: str
    ) -> tuple[bytes, str]:
        with self.open_regular(relative, label=label) as bound:
            return bound.bytes_and_sha256()

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def __enter__(self) -> "BoundDirectory":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - defensive leak guard
        try:
            self.close()
        except Exception:
            pass


def open_bound_directory(
    path: str | os.PathLike[str], *, label: str
) -> BoundDirectory:
    exact = _absolute_lexical_path(path, label=label)
    descriptor = _open_directory_absolute(exact, label=label)
    return BoundDirectory(path=exact, descriptor=descriptor, label=label)


def open_bound_regular_file(
    path: str | os.PathLike[str],
    *,
    label: str,
    allowed_root: str | os.PathLike[str] | None = None,
) -> BoundRegularFile:
    """Component-open and retain one regular file without following symlinks."""

    exact = _absolute_lexical_path(path, label=label)
    if exact == "/":
        raise ValueError(f"{label} must name a file")
    if allowed_root is not None:
        root = _absolute_lexical_path(allowed_root, label=f"{label} allowed root")
        _require_lexical_containment(exact, root, label=label)
    parent, leaf = exact.rsplit("/", 1)
    parent = parent or "/"
    directory = _open_directory_absolute(parent, label=f"{label} parent")
    try:
        descriptor = _open_regular_component(directory, leaf, label=label)
    finally:
        os.close(directory)
    return BoundRegularFile(path=exact, descriptor=descriptor, label=label)


def read_bound_bytes(
    path: str | os.PathLike[str],
    *,
    label: str,
    allowed_root: str | os.PathLike[str] | None = None,
) -> tuple[Path, bytes, str]:
    """Return lexical path, exact same-fd bytes, and SHA-256."""

    with open_bound_regular_file(path, label=label, allowed_root=allowed_root) as bound:
        payload, digest = bound.bytes_and_sha256()
        return Path(bound.path), payload, digest


def decode_json_bytes(payload: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error


def load_bound_json(
    path: str | os.PathLike[str],
    *,
    label: str,
    allowed_root: str | os.PathLike[str] | None = None,
) -> tuple[Path, Any, bytes, str]:
    exact, payload, digest = read_bound_bytes(
        path, label=label, allowed_root=allowed_root
    )
    return exact, decode_json_bytes(payload, label=label), payload, digest


def load_bound_torch(
    path: str | os.PathLike[str],
    *,
    label: str,
    allowed_root: str | os.PathLike[str] | None = None,
    weights_only: bool | None = None,
) -> tuple[Path, Any, bytes, str]:
    """Deserialize torch data only from the bytes read through the bound fd."""

    exact, payload, digest = read_bound_bytes(
        path, label=label, allowed_root=allowed_root
    )
    import torch

    buffer = io.BytesIO(payload)
    try:
        if weights_only is None:
            value = torch.load(buffer, map_location="cpu")
        else:
            value = torch.load(buffer, map_location="cpu", weights_only=weights_only)
    except TypeError:
        if weights_only is None:
            raise
        buffer.seek(0)
        value = torch.load(buffer, map_location="cpu")
    except Exception as error:
        raise ValueError(f"{label} is not a valid/loadable torch artifact") from error
    return exact, value, payload, digest


def _merge_config_mapping(child: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the subset of MMEngine inheritance used by the frozen r2 configs."""

    merged = copy.deepcopy(dict(base))
    for key, raw_value in child.items():
        value = copy.deepcopy(raw_value)
        if isinstance(value, Mapping):
            value = dict(value)
            replace = value.pop("_delete_", False)
            if not isinstance(replace, bool):
                raise ValueError(f"config {key}._delete_ must be boolean")
            if not replace and isinstance(merged.get(key), Mapping):
                merged[key] = _merge_config_mapping(value, merged[key])
                continue
        merged[key] = value
    return merged


def load_registered_python_config(
    *,
    repository_root: str | os.PathLike[str],
    config_relative: str,
    registered_sources: Mapping[str, str],
) -> tuple[Any, dict[str, str]]:
    """Load the r2 Python config closure from same-fd registered bytes.

    MMEngine's normal ``Config.fromfile`` reopens each pathname.  The formal
    route instead reads and hashes each inherited config through one retained
    repository directory descriptor, compiles those exact bytes, and performs
    the frozen mapping inheritance directly.
    """

    from mmengine.config import Config

    loaded: dict[str, str] = {}
    active: set[str] = set()

    def normalize_relative(value: str, *, parent: str | None = None) -> str:
        if not isinstance(value, str) or not value or "\\" in value:
            raise ValueError("registered config path must be a non-empty POSIX string")
        joined = posixpath.join(parent, value) if parent is not None else value
        normalized = posixpath.normpath(joined)
        pure = PurePosixPath(normalized)
        if (
            pure.is_absolute()
            or normalized.startswith("../")
            or normalized in ("", ".", "..")
            or any(part in ("", ".", "..") for part in pure.parts)
        ):
            raise ValueError("registered config inheritance escapes the repository")
        return pure.as_posix()

    entry = normalize_relative(config_relative)
    with open_bound_directory(repository_root, label="registered config repository") as root:

        def load_one(relative: str) -> dict[str, Any]:
            if relative in active:
                raise ValueError(f"registered config inheritance cycle at {relative}")
            active.add(relative)
            try:
                payload, digest = root.read_bytes(
                    relative, label=f"registered config {relative}"
                )
                expected = registered_sources.get(relative)
                if not isinstance(expected, str) or digest != expected:
                    raise RuntimeError(
                        f"registered config bytes are absent or mismatched: {relative}"
                    )
                loaded[relative] = digest
                try:
                    source = payload.decode("utf-8")
                    tree = ast.parse(source, filename=relative)
                except (UnicodeDecodeError, SyntaxError) as error:
                    raise ValueError(f"registered config is not valid UTF-8 Python: {relative}") from error
                base_assignments = [
                    node
                    for node in tree.body
                    if isinstance(node, (ast.Assign, ast.AnnAssign))
                    and (
                        (
                            isinstance(node, ast.Assign)
                            and any(
                                isinstance(target, ast.Name) and target.id == "_base_"
                                for target in node.targets
                            )
                        )
                        or (
                            isinstance(node, ast.AnnAssign)
                            and isinstance(node.target, ast.Name)
                            and node.target.id == "_base_"
                        )
                    )
                ]
                if len(base_assignments) > 1:
                    raise ValueError(f"registered config has multiple _base_ assignments: {relative}")
                namespace: dict[str, Any] = {
                    "__file__": f"{root.path}/{relative}",
                    "__name__": f"_chronotransport_config_{hashlib.sha256(relative.encode()).hexdigest()}",
                }
                exec(compile(tree, relative, "exec"), namespace, namespace)
                raw_bases = namespace.get("_base_", ())
                if isinstance(raw_bases, str):
                    bases = (raw_bases,)
                elif isinstance(raw_bases, (list, tuple)) and all(
                    isinstance(item, str) for item in raw_bases
                ):
                    bases = tuple(raw_bases)
                else:
                    raise ValueError(f"registered config _base_ must be strings: {relative}")
                merged: dict[str, Any] = {}
                parent = posixpath.dirname(relative)
                for base in bases:
                    base_relative = normalize_relative(base, parent=parent)
                    merged = _merge_config_mapping(load_one(base_relative), merged)
                child = {
                    key: value
                    for key, value in namespace.items()
                    if not key.startswith("__")
                    and key != "_base_"
                    and not isinstance(value, ModuleType)
                    and not callable(value)
                }
                return _merge_config_mapping(child, merged)
            finally:
                active.remove(relative)

        resolved = load_one(entry)
    return Config(resolved, filename=f"{Path(repository_root) / entry}"), dict(
        sorted(loaded.items())
    )


def ensure_bound_directory(path: str | os.PathLike[str], *, label: str) -> Path:
    exact = _absolute_lexical_path(path, label=label)
    descriptor = _open_directory_absolute(exact, label=label, create=True)
    os.close(descriptor)
    return Path(exact)


def secure_lexical_path(
    path: str | os.PathLike[str],
    *,
    label: str,
    allow_missing: bool = False,
) -> Path:
    """Validate every currently existing component with openat/O_NOFOLLOW."""

    exact = _absolute_lexical_path(path, label=label)
    if exact == "/":
        return Path(exact)
    parts = exact.split("/")[1:]
    directory = os.open("/", _directory_flags())
    try:
        for index, component in enumerate(parts):
            last = index == len(parts) - 1
            if last:
                try:
                    metadata = os.stat(component, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    if allow_missing:
                        return Path(exact)
                    raise
                if stat.S_ISLNK(metadata.st_mode):
                    raise ValueError(f"{label} contains a symlink leaf")
                return Path(exact)
            try:
                child = os.open(component, _directory_flags(), dir_fd=directory)
            except FileNotFoundError:
                if allow_missing:
                    return Path(exact)
                raise
            except NotADirectoryError as error:
                try:
                    metadata = os.stat(
                        component, dir_fd=directory, follow_symlinks=False
                    )
                except OSError:
                    metadata = None
                detail = (
                    "symlink parent"
                    if metadata is not None and stat.S_ISLNK(metadata.st_mode)
                    else "non-directory or replaced parent"
                )
                raise ValueError(f"{label} contains a {detail}: {component}") from error
            os.close(directory)
            directory = child
    finally:
        os.close(directory)
    raise AssertionError("unreachable secure lexical path state")


def path_exists_no_follow(
    path: str | os.PathLike[str], *, label: str
) -> bool:
    exact = _absolute_lexical_path(path, label=label)
    if exact == "/":
        return True
    parent, leaf = exact.rsplit("/", 1)
    try:
        directory = _open_directory_absolute(parent or "/", label=f"{label} parent")
    except FileNotFoundError:
        return False
    try:
        try:
            metadata = os.stat(leaf, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} must not be a symlink")
        return True
    finally:
        os.close(directory)


def _read_existing_at(directory: int, leaf: str, *, label: str) -> bytes:
    descriptor = _open_regular_component(directory, leaf, label=label)
    bound = BoundRegularFile(path=leaf, descriptor=descriptor, label=label)
    try:
        return bound.read_bytes()
    finally:
        bound.close()


def publish_bytes_exclusive(
    path: str | os.PathLike[str],
    payload: bytes,
    *,
    label: str,
    allow_existing_exact: bool = False,
    mode: int = 0o600,
) -> Path:
    """Durably publish bytes without overwrite, bound to one parent dir fd."""

    if not isinstance(payload, bytes):
        raise TypeError(f"{label} payload must be bytes")
    exact = _absolute_lexical_path(path, label=label)
    if exact == "/":
        raise ValueError(f"{label} must name a file")
    parent, leaf = exact.rsplit("/", 1)
    parent = parent or "/"
    directory = _open_directory_absolute(parent, label=f"{label} parent", create=True)
    temporary = f".{leaf}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    descriptor = -1
    linked = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        descriptor = os.open(temporary, flags, mode, dir_fd=directory)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise RuntimeError(f"{label} temporary write made no progress")
            written += count
        os.fsync(descriptor)
        temporary_identity = DescriptorIdentity.from_stat(os.fstat(descriptor))
        try:
            os.link(
                temporary,
                leaf,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
            linked = True
        except FileExistsError:
            if not allow_existing_exact:
                raise
            if _read_existing_at(directory, leaf, label=label) != payload:
                raise FileExistsError(f"{label} exists with different bytes: {exact}")
        if linked:
            published = os.stat(leaf, dir_fd=directory, follow_symlinks=False)
            published_identity = DescriptorIdentity.from_stat(published)
            if (
                published_identity.device != temporary_identity.device
                or published_identity.inode != temporary_identity.inode
                or not stat.S_ISREG(published_identity.mode)
            ):
                raise RuntimeError(f"{label} publication inode identity mismatch")
            os.fsync(directory)
        return Path(exact)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass
        os.close(directory)


@contextmanager
def exclusive_file_lock(
    path: str | os.PathLike[str], *, label: str, payload: bytes
):
    """Create, retain, and inode-check one exclusive lock in a bound parent."""

    if not isinstance(payload, bytes) or not payload:
        raise ValueError(f"{label} lock payload must be non-empty bytes")
    exact = _absolute_lexical_path(path, label=label)
    parent, leaf = exact.rsplit("/", 1)
    directory = _open_directory_absolute(
        parent or "/", label=f"{label} parent", create=True
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(leaf, flags, 0o600, dir_fd=directory)
    identity = DescriptorIdentity.from_stat(os.fstat(descriptor))
    try:
        view = memoryview(payload)
        cursor = 0
        while cursor < len(view):
            written = os.write(descriptor, view[cursor:])
            if written <= 0:
                raise RuntimeError(f"{label} lock payload write made no progress")
            cursor += written
        os.fsync(descriptor)
        os.fsync(directory)
        yield Path(exact)
    finally:
        os.close(descriptor)
        try:
            current = os.stat(leaf, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None:
            current_identity = DescriptorIdentity.from_stat(current)
            if (
                current_identity.device != identity.device
                or current_identity.inode != identity.inode
            ):
                os.close(directory)
                raise RuntimeError(f"{label} lock pathname changed identity")
            os.unlink(leaf, dir_fd=directory)
            os.fsync(directory)
        os.close(directory)


def _module_source_path(module: ModuleType, *, label: str) -> str:
    source = getattr(module, "__file__", None)
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if not isinstance(source, str) or not isinstance(origin, str):
        raise RuntimeError(f"{label} lacks a concrete source/origin identity")
    if source.endswith((".pyc", ".pyo")):
        source = source[:-1]
    if origin.endswith((".pyc", ".pyo")):
        origin = origin[:-1]
    source = _absolute_lexical_path(source, label=f"{label} __file__")
    origin = _absolute_lexical_path(origin, label=f"{label} __spec__.origin")
    if source != origin or not source.endswith(".py"):
        raise RuntimeError(f"{label} source and import origin are not one Python file")
    return source


def audit_formal_python_runtime(
    *,
    repository_root: str | os.PathLike[str],
    registered_sources: Mapping[str, str],
    entrypoint_relative: str,
    module_prefixes: Sequence[str] = ("opentad", "tools.bata"),
) -> dict[str, Any]:
    """Bind entrypoint plus every loaded repository Python module to registration."""

    root = _absolute_lexical_path(repository_root, label="formal repository root")
    pure_entrypoint = PurePosixPath(entrypoint_relative)
    if (
        pure_entrypoint.is_absolute()
        or "\\" in entrypoint_relative
        or any(part in ("", ".", "..") for part in pure_entrypoint.parts)
        or pure_entrypoint.as_posix() != entrypoint_relative
    ):
        raise ValueError("formal entrypoint must be a canonical repository path")
    expected_entrypoint = f"{root}/{entrypoint_relative}"
    actual_entrypoint = _absolute_lexical_path(sys.argv[0], label="formal entrypoint")
    if actual_entrypoint != expected_entrypoint:
        raise RuntimeError(
            "formal entrypoint path differs from its registered repository location"
        )

    audited: dict[str, str] = {}

    def audit_path(absolute: str, *, owner: str) -> None:
        _require_lexical_containment(absolute, root, label=owner)
        relative = os.path.relpath(absolute, root).replace(os.sep, "/")
        expected = registered_sources.get(relative)
        if not isinstance(expected, str):
            raise RuntimeError(f"loaded repository source is unregistered: {relative}")
        _, payload, digest = read_bound_bytes(
            absolute, label=owner, allowed_root=root
        )
        if digest != expected:
            raise RuntimeError(f"loaded repository source bytes differ: {relative}")
        if relative in audited and audited[relative] != digest:
            raise RuntimeError(f"repository source identity changed during audit: {relative}")
        audited[relative] = digest

    audit_path(expected_entrypoint, owner="formal entrypoint")
    for name, module in sorted(sys.modules.items()):
        if module is None or not any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in module_prefixes
        ):
            continue
        source = getattr(module, "__file__", None)
        if not isinstance(source, str):
            continue
        candidate = source[:-1] if source.endswith((".pyc", ".pyo")) else source
        candidate = os.path.abspath(candidate)
        try:
            inside = os.path.commonpath((candidate, root)) == root
        except ValueError:
            inside = False
        if not inside:
            continue
        audit_path(_module_source_path(module, label=f"module {name}"), owner=f"module {name}")

    return {
        "schema": PYTHON_RUNTIME_IDENTITY_SCHEMA,
        "entrypoint": entrypoint_relative,
        "entrypoint_sha256": audited[entrypoint_relative],
        "loaded_repository_sources": dict(sorted(audited.items())),
        "loaded_repository_sources_sha256": hashlib.sha256(
            json.dumps(
                dict(sorted(audited.items())),
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


__all__ = [
    "BoundDirectory",
    "BoundRegularFile",
    "DescriptorIdentity",
    "FILESYSTEM_IDENTITY_SCHEMA",
    "PYTHON_RUNTIME_IDENTITY_SCHEMA",
    "audit_formal_python_runtime",
    "decode_json_bytes",
    "ensure_bound_directory",
    "exclusive_file_lock",
    "load_bound_json",
    "load_bound_torch",
    "load_registered_python_config",
    "open_bound_directory",
    "open_bound_regular_file",
    "path_exists_no_follow",
    "publish_bytes_exclusive",
    "read_bound_bytes",
    "secure_lexical_path",
]
