"""
Parses ATE (automated test equipment) wafer datalog CSV files — the
"dlogTDO" export format used by CRAFT-based testers — into a wafer bin map
array compatible with the WM-811K-trained classifier.

Expected layout of one CSV (see sample RK30906-xx_dlogTDO.csv):

    Lot ID                  : RK30906
    Wafer ID                : RK30906-01
    ...
     Sample   Pass   Pass%   Fail   Fail%
       654   574    87.77%      80    12.23%

    Serial#,Site#,Bin#,SBin#,XAdr,YAdr,...
    0,7,9,9,20,6,338.615,...
    1,4,9,9,21,6,327.726,...
    ...

Bin# == 1 is treated as a passing die; any other Bin# is a fail.
This mirrors the WM-811K pixel convention: 0 = background/untested,
1 = pass, 2 = fail.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

import numpy as np

HEADER_ROW_PREFIX = "Serial#"
META_FIELDS = {
    "lot id": "lot_id",
    "wafer id": "wafer_id",
    "test start date": "test_date",
    "production id": "production_id",
}


class WaferLogParseError(ValueError):
    pass


@dataclass
class WaferLogResult:
    filename: str
    lot_id: str
    wafer_id: str
    n_die: int
    n_pass: int
    n_fail: int
    bin_map: np.ndarray = field(repr=False)


def _decode(raw: bytes) -> str:
    """These logs are typically Big5/CP950 or Latin-1 with occasional
    garbled bytes in free-text fields we don't care about. Decode
    permissively so a bad byte never breaks the whole file."""
    for enc in ("utf-8-sig", "cp950", "big5", "latin1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin1", errors="replace")


def parse_dlog_csv(raw: bytes, filename: str = "") -> WaferLogResult:
    text = _decode(raw)
    lines = text.splitlines()

    meta = {}
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith(HEADER_ROW_PREFIX):
            header_idx = i
            break
        if ":" in line:
            key, _, val = line.partition(":")
            key_norm = key.strip().lower()
            if key_norm in META_FIELDS:
                meta[META_FIELDS[key_norm]] = val.strip()

    if header_idx is None:
        raise WaferLogParseError(
            f"{filename}：找不到資料表頭（'{HEADER_ROW_PREFIX}' 那一列），"
            "檔案格式可能不是預期的 ATE datalog"
        )

    header_cols = [c.strip() for c in lines[header_idx].split(",")]
    try:
        idx_bin = header_cols.index("Bin#")
        idx_x = header_cols.index("XAdr")
        idx_y = header_cols.index("YAdr")
    except ValueError as e:
        raise WaferLogParseError(
            f"{filename}：資料表頭缺少必要欄位（Bin# / XAdr / YAdr）：{e}"
        )

    xs, ys, bins = [], [], []
    reader = csv.reader(lines[header_idx + 1:])
    for row in reader:
        if len(row) <= max(idx_bin, idx_x, idx_y):
            continue
        try:
            x = int(row[idx_x])
            y = int(row[idx_y])
            b = int(row[idx_bin])
        except (ValueError, IndexError):
            continue
        xs.append(x)
        ys.append(y)
        bins.append(b)

    if not xs:
        raise WaferLogParseError(f"{filename}：資料列解析結果為空，請確認檔案內容")

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    bins_arr = np.array(bins)

    x_min, x_max = xs_arr.min(), xs_arr.max()
    y_min, y_max = ys_arr.min(), ys_arr.max()

    height = int(y_max - y_min + 1)
    width = int(x_max - x_min + 1)

    bin_map = np.zeros((height, width), dtype=np.float32)
    row_idx = ys_arr - y_min
    col_idx = xs_arr - x_min
    is_pass = bins_arr == 1
    bin_map[row_idx[is_pass], col_idx[is_pass]] = 1.0
    bin_map[row_idx[~is_pass], col_idx[~is_pass]] = 2.0

    n_pass = int(is_pass.sum())
    n_die = len(bins_arr)

    lot_id = meta.get("lot_id") or _guess_lot_from_filename(filename)
    wafer_id = meta.get("wafer_id") or filename

    return WaferLogResult(
        filename=filename,
        lot_id=lot_id,
        wafer_id=wafer_id,
        n_die=n_die,
        n_pass=n_pass,
        n_fail=n_die - n_pass,
        bin_map=bin_map,
    )


def _guess_lot_from_filename(filename: str) -> str:
    m = re.match(r"([A-Za-z0-9]+)-\d+", filename)
    return m.group(1) if m else "unknown"
