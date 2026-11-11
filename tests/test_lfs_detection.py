"""The data-dependent tests must skip, not crash, on Git-LFS pointer files."""

import numpy as np

from tests import conftest


def _use_tmp_root(monkeypatch, tmp_path):
    monkeypatch.setattr(conftest, "ROOT", str(tmp_path))


def test_missing_file_is_not_real(monkeypatch, tmp_path):
    _use_tmp_root(monkeypatch, tmp_path)
    assert conftest.is_real_file("nope.npy") is False


def test_lfs_pointer_is_not_real(monkeypatch, tmp_path):
    _use_tmp_root(monkeypatch, tmp_path)
    (tmp_path / "X.npy").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "0" * 64 + "\nsize 123\n")
    assert conftest.is_real_file("X.npy") is False


def test_real_npy_is_real(monkeypatch, tmp_path):
    _use_tmp_root(monkeypatch, tmp_path)
    np.save(tmp_path / "y.npy", np.zeros((2, 5), dtype=np.float32))
    assert conftest.is_real_file("y.npy") is True


def test_real_checkpoint_zip_is_real(monkeypatch, tmp_path):
    _use_tmp_root(monkeypatch, tmp_path)
    import torch
    torch.save({"w": torch.zeros(2)}, tmp_path / "m.pt")
    assert conftest.is_real_file("m.pt") is True
