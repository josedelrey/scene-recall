"""Verified evidence survives destructive output compaction."""

import runpy
import tarfile
from pathlib import Path

import pytest

module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/compact_outputs.py"))
compact = module["compact"]
snapshot = module["snapshot"]


def inputs(tmp_path):
    root = tmp_path / "outputs"
    case = root / "study" / "run"
    case.mkdir(parents=True)
    (case / "report.json").write_bytes(b'{"score":0.125}\n')
    (case / "trajectory.npz").write_bytes(b"exact saved trajectory")
    (case / "timed_factors.npz").write_bytes(b"exact owned measurements")
    (case / "features.npz").write_bytes(b"regenerable feature cache")
    (case / "map.ply").write_bytes(b"regenerable geometry")
    return root, case


def test_original_reports_and_traces_restore_exactly_after_compaction(tmp_path):
    root, case = inputs(tmp_path)
    expected = {
        p.name: p.read_bytes()
        for p in case.iterdir()
        if p.name not in ("features.npz", "map.ply")
    }
    result = compact(root)
    assert result["original_files"] == 5
    assert result["retained_files"] == 3
    assert not case.exists()
    with tarfile.open(root / "evidence/study.tar.xz") as archive:
        archive.extractall(tmp_path, filter="data")
    assert {p.name: p.read_bytes() for p in case.iterdir()} == expected


def test_snapshot_does_not_mutate_files(tmp_path):
    root, case = inputs(tmp_path)
    before = {p: p.read_bytes() for p in case.iterdir()}
    assert len(snapshot(root)) == 5
    assert all(p.read_bytes() == content for p, content in before.items())


def test_archive_verification_failure_cannot_delete_originals(tmp_path, monkeypatch):
    root, case = inputs(tmp_path)

    def fail(*_):
        raise ValueError("injected damaged archive")

    monkeypatch.setitem(compact.__globals__, "verify_archive", fail)
    with pytest.raises(ValueError, match="damaged archive"):
        compact(root)
    assert len(list(case.iterdir())) == 5
    assert not (root / "evidence").exists()


def test_changed_outputs_abort_before_removal(tmp_path, monkeypatch):
    root, case = inputs(tmp_path)
    verify = compact.__globals__["verify_archive"]

    def mutate(*args):
        verify(*args)
        (case / "map.ply").write_bytes(b"new concurrent output must survive")

    monkeypatch.setitem(compact.__globals__, "verify_archive", mutate)
    with pytest.raises(ValueError, match="outputs changed"):
        compact(root)
    assert len(list(case.iterdir())) == 5
    assert (case / "map.ply").read_bytes() == b"new concurrent output must survive"


def test_symlinks_and_existing_capsules_are_never_followed_or_overwritten(tmp_path):
    root, case = inputs(tmp_path)
    external = tmp_path / "external.txt"
    external.write_text("preserve me")
    link = case / "external.txt"
    try:
        link.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links unavailable")
    with pytest.raises(ValueError, match="non-regular"):
        compact(root)
    assert external.read_text() == "preserve me"
    link.unlink()
    (root / "evidence").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        compact(root)


def test_symlinked_output_root_cannot_redirect_deletion(tmp_path):
    external, case = inputs(tmp_path / "external")
    parent = tmp_path / "project"
    parent.mkdir()
    link = parent / "outputs"
    try:
        link.symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links unavailable")
    with pytest.raises(ValueError, match="symlinked outputs"):
        compact(link)
    assert len(list(case.iterdir())) == 5
