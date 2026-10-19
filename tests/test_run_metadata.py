"""Code fingerprints have the same manifest order on every platform."""

import hashlib

import scene_recall
from scene_recall.run_metadata import code_hash


def test_code_hash_uses_portable_paths_and_case_sensitive_order(tmp_path, monkeypatch):
    expected = hashlib.sha256(
        b"B.pyupper\n__init__.pyinit\na.pylower\nnested/module.pynested\n"
    ).hexdigest()
    for location in ("first", "relocated"):
        root = tmp_path / location
        (root / "nested").mkdir(parents=True)
        for name, contents in (
            ("nested/module.py", b"nested\n"),
            ("a.py", b"lower\n"),
            ("__init__.py", b"init\n"),
            ("B.py", b"upper\n"),
            ("notes.txt", b"excluded"),
            ("excluded.PY", b"excluded"),
        ):
            (root / name).write_bytes(contents)
        monkeypatch.setattr(scene_recall, "__file__", str(root / "__init__.py"))
        assert code_hash() == expected
        (root / "nested/module.py").write_bytes(b"changed\n")
        assert code_hash() != expected


def test_code_hash_preserves_additional_source_contents_and_order(
    tmp_path, monkeypatch
):
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_bytes(b"")
    monkeypatch.setattr(scene_recall, "__file__", str(package / "__init__.py"))
    first, second = tmp_path / "first.py", tmp_path / "second.py"
    first.write_bytes(b"first\n")
    second.write_bytes(b"second\n")
    assert (
        code_hash(first, second)
        == hashlib.sha256(b"__init__.pyfirst\nsecond\n").hexdigest()
    )
    assert code_hash(first, second) != code_hash(second, first)
