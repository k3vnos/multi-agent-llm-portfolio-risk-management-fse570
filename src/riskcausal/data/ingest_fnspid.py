"""FNSPID (Dong, Fan & Peng, KDD'24) on Hugging Face — public, free. Used only by the explanation layer."""

from __future__ import annotations

import io

import pandas as pd
import requests

REPO = "Zihan1004/FNSPID"
API_TREE = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true"
API_INFO = f"https://huggingface.co/api/datasets/{REPO}"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{path}}"


def repo_info() -> dict:
    r = requests.get(API_INFO, timeout=30)
    r.raise_for_status()
    js = r.json()
    return {"private": js.get("private"), "gated": js.get("gated"), "last_modified": js.get("lastModified"),
            "license": (js.get("cardData") or {}).get("license")}


def list_files() -> list[dict]:
    r = requests.get(API_TREE, timeout=30)
    r.raise_for_status()
    return [{"path": f["path"], "size_bytes": f.get("size")} for f in r.json() if f.get("type") == "file"]


def read_byte_range(path: str, start: int, length: int) -> bytes:
    r = requests.get(RESOLVE.format(path=path), headers={"Range": f"bytes={start}-{start + length - 1}"},
                     timeout=120, allow_redirects=True)
    r.raise_for_status()
    return r.content


def sample_csv(path: str, nbytes: int = 4_000_000, offset: int = 0, header: list[str] | None = None) -> pd.DataFrame:
    """Parse complete CSV rows from a byte range (partial first/last lines discarded)."""
    raw = read_byte_range(path, offset, nbytes).decode("utf-8", errors="replace")
    if offset > 0:
        raw = raw.split("\n", 1)[1]
    raw = raw.rsplit("\n", 1)[0]
    if header is not None:
        return pd.read_csv(io.StringIO(raw), header=None, names=header, on_bad_lines="skip", engine="python")
    return pd.read_csv(io.StringIO(raw), on_bad_lines="skip", engine="python")
