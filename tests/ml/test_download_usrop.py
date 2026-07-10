"""Tests for the USROP download-and-verify script. No real network requests are made."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ml.data.download_usrop import (
    ChecksumMismatchError,
    DownloadError,
    download_usrop_dataset,
)

FILENAME = "USROP_A 0 N-NA_F-9_Ad.csv"
CONTENT = b"header\n1,2,3\n"
WRONG_CONTENT = b"not the real file"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def checksums_file(tmp_path: Path) -> Path:
    path = tmp_path / "data_checksums.json"
    path.write_text(json.dumps({FILENAME: _sha256(CONTENT)}), encoding="utf-8")
    return path


def test_skips_download_when_file_already_matches_checksum(
    tmp_path: Path, checksums_file: Path
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / FILENAME).write_bytes(CONTENT)

    with patch("ml.data.download_usrop._download_file") as mock_download:
        results = download_usrop_dataset(
            raw_data_dir=raw_dir, checksums_path=checksums_file, filenames=(FILENAME,)
        )

    mock_download.assert_not_called()
    assert results[0].skipped is True


def test_downloads_missing_file_and_verifies_checksum(tmp_path: Path, checksums_file: Path) -> None:
    raw_dir = tmp_path / "raw"

    def fake_download(url: str, destination: Path) -> None:
        destination.write_bytes(CONTENT)

    with patch("ml.data.download_usrop._download_file", side_effect=fake_download) as mock_download:
        results = download_usrop_dataset(
            raw_data_dir=raw_dir, checksums_path=checksums_file, filenames=(FILENAME,)
        )

    mock_download.assert_called_once()
    assert results[0].skipped is False
    assert (raw_dir / FILENAME).read_bytes() == CONTENT


def test_raises_and_removes_file_on_checksum_mismatch(tmp_path: Path, checksums_file: Path) -> None:
    raw_dir = tmp_path / "raw"

    def fake_download(url: str, destination: Path) -> None:
        destination.write_bytes(WRONG_CONTENT)

    with (
        patch("ml.data.download_usrop._download_file", side_effect=fake_download),
        pytest.raises(ChecksumMismatchError),
    ):
        download_usrop_dataset(
            raw_data_dir=raw_dir, checksums_path=checksums_file, filenames=(FILENAME,)
        )

    assert not (raw_dir / FILENAME).exists()


def test_raises_download_error_when_source_is_unreachable(
    tmp_path: Path, checksums_file: Path
) -> None:
    raw_dir = tmp_path / "raw"

    with (
        patch("ml.data.download_usrop._download_file", side_effect=DownloadError("timeout")),
        pytest.raises(DownloadError),
    ):
        download_usrop_dataset(
            raw_data_dir=raw_dir, checksums_path=checksums_file, filenames=(FILENAME,)
        )


def test_raises_when_checksums_file_is_missing(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    missing_checksums = tmp_path / "does_not_exist.json"

    with pytest.raises(DownloadError):
        download_usrop_dataset(
            raw_data_dir=raw_dir, checksums_path=missing_checksums, filenames=(FILENAME,)
        )


def test_raises_when_filename_has_no_expected_checksum(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    checksums_path = tmp_path / "data_checksums.json"
    checksums_path.write_text(json.dumps({}), encoding="utf-8")

    with pytest.raises(DownloadError):
        download_usrop_dataset(
            raw_data_dir=raw_dir, checksums_path=checksums_path, filenames=(FILENAME,)
        )
