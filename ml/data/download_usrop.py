"""Download and integrity-verify the USROP dataset CSV files.

Source: https://github.com/AndrzejTunkiel/USROP (branch ``master``), the official
repository for Tunkiel, Sui & Wiktorski (2021). The 7 CSV files live directly in the
repository root -- not in GitHub releases nor on an external host -- so they are fetched
via ``raw.githubusercontent.com``. See docs/adr/001-dataset-selection.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

USROP_BASE_URL = "https://raw.githubusercontent.com/AndrzejTunkiel/USROP/master/"
DEFAULT_RAW_DATA_DIR = _REPO_ROOT / "data" / "raw"
DEFAULT_CHECKSUMS_PATH = _REPO_ROOT / "docs" / "data_checksums.json"

USROP_FILENAMES: tuple[str, ...] = (
    "USROP_A 0 N-NA_F-9_Ad.csv",
    "USROP_A 1 N-S_F-7d.csv",
    "USROP_A 2 N-SH_F-14d.csv",
    "USROP_A 3 N-SH-F-15d.csv",
    "USROP_A 4 N-SH_F-15Sd.csv",
    "USROP_A 5 N-SH-F-5d.csv",
    "USROP_A 6 N-SH_F-9d.csv",
)


class DownloadError(RuntimeError):
    """Raised when a USROP source file cannot be retrieved from the repository."""


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file's SHA256 does not match the pinned expected value."""


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of processing a single USROP file."""

    filename: str
    path: Path
    skipped: bool


def _sha256_of(path: Path) -> str:
    """Compute the SHA256 hex digest of a file on disk, reading it in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_expected_checksums(checksums_path: Path) -> dict[str, str]:
    """Load the {filename: sha256} mapping pinned in the repository."""
    try:
        with checksums_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)  # type: ignore[no-any-return]
    except FileNotFoundError as exc:
        raise DownloadError(
            f"No se encontro el archivo de checksums esperados en {checksums_path}."
        ) from exc


def _download_file(url: str, destination: Path) -> None:
    """Stream a single file from `url` to `destination`.

    Isolated in its own function so tests can mock it without performing real requests.
    """
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            destination.write_bytes(response.read())
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise DownloadError(f"No se pudo descargar {url}: {exc}") from exc


def download_usrop_dataset(
    raw_data_dir: Path = DEFAULT_RAW_DATA_DIR,
    checksums_path: Path = DEFAULT_CHECKSUMS_PATH,
    base_url: str = USROP_BASE_URL,
    filenames: tuple[str, ...] = USROP_FILENAMES,
) -> list[DownloadResult]:
    """Download the USROP CSV files into `raw_data_dir`, verifying each against `checksums_path`.

    Idempotent: a file already present whose SHA256 matches the pinned expected value is not
    re-downloaded. Raises `DownloadError` if the source is unreachable or a filename has no
    pinned checksum, and `ChecksumMismatchError` (removing the bad file) if a freshly downloaded
    file's content does not match the pinned checksum -- signalling the upstream source changed.
    """
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    expected_checksums = _load_expected_checksums(checksums_path)
    results: list[DownloadResult] = []

    for filename in filenames:
        destination = raw_data_dir / filename
        expected = expected_checksums.get(filename)
        if expected is None:
            raise DownloadError(
                f"No hay checksum esperado registrado para '{filename}' en {checksums_path}."
            )

        if destination.exists() and _sha256_of(destination) == expected:
            logger.info("Ya existe con checksum valido, se omite: %s", filename)
            results.append(DownloadResult(filename=filename, path=destination, skipped=True))
            continue

        url = base_url + urllib.parse.quote(filename)
        logger.info("Descargando %s", filename)
        _download_file(url, destination)

        actual = _sha256_of(destination)
        if actual != expected:
            destination.unlink(missing_ok=True)
            raise ChecksumMismatchError(
                f"Checksum invalido para '{filename}': esperado {expected}, obtenido {actual}. "
                "La fuente pudo haber cambiado de contenido sin aviso."
            )

        results.append(DownloadResult(filename=filename, path=destination, skipped=False))

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_usrop_dataset()
