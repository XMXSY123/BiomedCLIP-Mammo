# -*- coding: utf-8 -*-
"""
从 Metadata.xlsx 构建 (image_id, view) / image_id -> 提示词。
规则：E 列 M=恶性、B=良性。提示词：D 为 NORM -> "Benign"；D 非 NORM 且 E=B -> "Benign Mass"；D 非 NORM 且 E=M -> "Malignant Mass"。
排除：D 列为 ARCH/ASYM 的图像，以及 E 列为 N 的图像。
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd

# 需排除的 D 列病灶码
D_EXCLUDE = {"ARCH", "ASYM"}


def _normalize_d_value(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip().upper()
    s = re.sub(r"\s*\+\s*", "+", s)
    return s


def _normalize_e_value(val) -> str:
    """E 列：B=良性, M=恶性, N=排除。"""
    if pd.isna(val):
        return ""
    return str(val).strip().upper()


def _d_contains_excluded(d_val: str) -> bool:
    """D 列是否包含 ARCH 或 ASYM（含组合如 ARCH+CALC）。"""
    for part in d_val.split("+"):
        if part.strip() in D_EXCLUDE:
            return True
    return False


def _all_d_norm(d_vals: Set[str]) -> bool:
    """是否全部为 NORM。"""
    expanded = set()
    for c in d_vals:
        for part in c.split("+"):
            expanded.add(part.strip())
    return expanded <= {"NORM"} or expanded == set()


def _parse_metadata_with_e(df: pd.DataFrame) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[Tuple[str, str], List[Tuple[str, str]]]]:
    """
    解析 Metadata，得到 (image_id, view) / image_id -> [(D, E), ...]。
    """
    start = 0
    for i in range(min(50, len(df))):
        v0 = str(df.iloc[i, 0]).strip()
        if v0 and re.match(r"^IMG\d+", v0, re.IGNORECASE):
            start = i
            break
    data = df.iloc[start:]
    by_id = {}
    by_id_view = {}
    for _, row in data.iterrows():
        pid = str(row[0]).strip()
        if not pid or not re.match(r"^IMG\d+", pid, re.IGNORECASE):
            continue
        view = (str(row[1]).strip().upper() if not pd.isna(row[1]) else "")
        d_val = _normalize_d_value(row[3])
        e_val = _normalize_e_value(row[4])  # E 列 index 4
        if not d_val:
            continue
        key_v = (pid, view)
        if pid not in by_id:
            by_id[pid] = []
        by_id[pid].append((d_val, e_val))
        if key_v not in by_id_view:
            by_id_view[key_v] = []
        by_id_view[key_v].append((d_val, e_val))
    return by_id, by_id_view


# 提示词前缀，组成完整句子
PROMPT_PREFIX = "Mammography showing "


def _label_from_rows(rows: List[Tuple[str, str]]) -> str:
    """
    根据 (D, E) 列表得到提示词句子：前缀 + Benign / Benign Mass / Malignant Mass，或 None 表示排除。
    """
    if not rows:
        return None
    # 排除：任一 D 含 ARCH/ASYM，或任一 E 为 N
    for d_val, e_val in rows:
        if _d_contains_excluded(d_val):
            return None
        if e_val == "N":
            return None
    d_vals = {r[0] for r in rows}
    e_vals = {r[1] for r in rows}
    if _all_d_norm(d_vals):
        return PROMPT_PREFIX + "healthy breast"
    if "M" in e_vals:
        return PROMPT_PREFIX + "breast with malignant mass"
    if "B" in e_vals:
        return PROMPT_PREFIX + "breast with benign mass"
    return None


def build_image_to_text_map(metadata_path: str) -> Dict[str, str]:
    """
    排除 D 含 ARCH/ASYM、E 为 N 的样本
    """
    df = pd.read_excel(metadata_path, header=None)
    by_id, by_id_view = _parse_metadata_with_e(df)
    out = {}
    for (pid, view), rows in by_id_view.items():
        label = _label_from_rows(rows)
        if label is None:
            continue
        key = f"{pid}|{view}" if view else pid
        out[key] = label
    for pid, rows in by_id.items():
        label = _label_from_rows(rows)
        if label is None:
            continue
        out[pid] = label
    return out


def image_id_and_view_from_path(file_path: str) -> Tuple[str, str]:
    """
    从文件路径解析 (image_id, view)。如 IMG001_MLOLT.png -> (IMG001, MLOLT)，IMG001.png -> (IMG001, "")。
    """
    stem = Path(file_path).stem
    m = re.match(r"^(IMG\d+)(?:_(.+))?$", stem, re.IGNORECASE)
    if m:
        pid, view = m.group(1).upper(), (m.group(2) or "").strip().upper()
        return pid, view
    return stem, ""
