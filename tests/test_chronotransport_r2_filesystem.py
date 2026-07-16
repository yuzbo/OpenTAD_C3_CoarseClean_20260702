from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import pytest

from opentad.models.chronotransport.filesystem import (
    audit_formal_python_runtime,
    load_registered_python_config,
    open_bound_directory,
    open_bound_regular_file,
    publish_bytes_exclusive,
    read_bound_bytes,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "O_NOFOLLOW"),
    reason="formal descriptor contract is POSIX-only",
)


def test_component_open_rejects_parent_and_leaf_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    source = real / "input.bin"
    source.write_bytes(b"registered")
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real, target_is_directory=True)
    leaf_link = real / "leaf-link.bin"
    leaf_link.symlink_to(source)

    with pytest.raises(ValueError, match="symlink"):
        open_bound_regular_file(
            parent_link / source.name,
            label="parent symlink probe",
        )
    with pytest.raises(ValueError, match="symlink"):
        open_bound_regular_file(leaf_link, label="leaf symlink probe")


def test_bound_descriptor_survives_pathname_inode_swap(tmp_path: Path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"first inode")
    with open_bound_regular_file(source, label="inode swap probe") as bound:
        replacement = tmp_path / "replacement.bin"
        replacement.write_bytes(b"second inode")
        os.replace(replacement, source)
        payload, digest = bound.bytes_and_sha256()

    assert payload == b"first inode"
    assert digest == hashlib.sha256(payload).hexdigest()
    assert source.read_bytes() == b"second inode"


def test_bound_directory_survives_root_path_replacement(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "source.py").write_bytes(b"ORIGINAL = True\n")
    with open_bound_directory(root, label="repository root") as bound_root:
        original = tmp_path / "renamed-root"
        root.rename(original)
        root.mkdir()
        (root / "source.py").write_bytes(b"SUBSTITUTED = True\n")
        payload, digest = bound_root.read_bytes("source.py", label="registered source")

    assert payload == b"ORIGINAL = True\n"
    assert digest == hashlib.sha256(payload).hexdigest()


def test_bound_descriptor_rejects_same_inode_mutation(tmp_path: Path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"stable")
    with open_bound_regular_file(source, label="mutation probe") as bound:
        source.write_bytes(b"mutated")
        with pytest.raises(RuntimeError, match="changed after descriptor binding"):
            bound.read_bytes()


def test_same_fd_bytes_and_hash_are_one_value(tmp_path: Path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"same descriptor")
    exact, payload, digest = read_bound_bytes(source, label="same-fd probe")
    assert exact == source
    assert payload == b"same descriptor"
    assert digest == hashlib.sha256(payload).hexdigest()


def test_publication_is_no_clobber_and_parent_descriptor_bound(tmp_path: Path) -> None:
    output = tmp_path / "run" / "terminal.json"
    publish_bytes_exclusive(output, b"first\n", label="terminal")
    assert output.read_bytes() == b"first\n"
    with pytest.raises(FileExistsError):
        publish_bytes_exclusive(output, b"second\n", label="terminal")
    publish_bytes_exclusive(
        output,
        b"first\n",
        label="terminal",
        allow_existing_exact=True,
    )

    actual = tmp_path / "actual"
    actual.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        publish_bytes_exclusive(
            linked_parent / "artifact.json",
            b"forbidden\n",
            label="symlink-parent publication",
        )
    assert not (actual / "artifact.json").exists()


def test_registered_config_loader_uses_exact_inherited_bytes(tmp_path: Path) -> None:
    base = tmp_path / "base.py"
    child = tmp_path / "child.py"
    base.write_bytes(b"model = dict(depth=12, nested=dict(width=384, keep=True))\n")
    child.write_bytes(
        b"_base_ = ['./base.py']\nmodel = dict(nested=dict(width=512), enabled=True)\n"
    )
    registered = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (base, child)
    }
    config, loaded = load_registered_python_config(
        repository_root=tmp_path,
        config_relative="child.py",
        registered_sources=registered,
    )
    assert config.model.depth == 12
    assert config.model.nested == {"width": 512, "keep": True}
    assert config.model.enabled is True
    assert loaded == registered


def test_registered_r2_config_is_semantically_identical_to_mmengine_loader() -> None:
    from mmengine.config import Config

    root = Path(__file__).resolve().parents[1]
    relative = "configs/adatad/thumos/c3_chronotransport_r2_stage_b.py"
    expected_closure = (
        "configs/_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py",
        "configs/_base_/models/actionformer.py",
        "configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_a.py",
        "configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_b.py",
        relative,
        "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py",
    )
    registered = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in expected_closure
    }
    exact, loaded = load_registered_python_config(
        repository_root=root,
        config_relative=relative,
        registered_sources=registered,
    )
    reference = Config.fromfile(root / relative)

    assert set(loaded) == set(expected_closure)
    assert exact.to_dict() == reference.to_dict()


def test_runtime_audit_binds_entrypoint_module_origin_and_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entrypoint = tmp_path / "entrypoint.py"
    module_path = tmp_path / "w3probe.py"
    entrypoint.write_bytes(b"# entrypoint\n")
    module_path.write_bytes(b"VALUE = 7\n")
    spec = importlib.util.spec_from_file_location("w3probe", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["w3probe"] = module
    try:
        spec.loader.exec_module(module)
        monkeypatch.setattr(sys, "argv", [str(entrypoint)])
        registered = {
            entrypoint.name: hashlib.sha256(entrypoint.read_bytes()).hexdigest(),
            module_path.name: hashlib.sha256(module_path.read_bytes()).hexdigest(),
        }
        report = audit_formal_python_runtime(
            repository_root=tmp_path,
            registered_sources=registered,
            entrypoint_relative=entrypoint.name,
            module_prefixes=("w3probe",),
        )
        assert report["loaded_repository_sources"] == registered

        with pytest.raises(RuntimeError, match="unregistered"):
            audit_formal_python_runtime(
                repository_root=tmp_path,
                registered_sources={entrypoint.name: registered[entrypoint.name]},
                entrypoint_relative=entrypoint.name,
                module_prefixes=("w3probe",),
            )
    finally:
        sys.modules.pop("w3probe", None)
