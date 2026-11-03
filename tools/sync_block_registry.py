"""Keep the block registries in sync with the real map data.

Scans every chunk of the game map cache, finds block ids that neither
``docs/block_registry.json`` nor ``docs/external_config_block_registry.json``
know about, and adds a sensible external config for each of them:

* foliage / grass blocks render as a crossed sprite (like the game's plants);
* solid blocks render as a textured cube;
* blocks without a PDA icon fall back to a vegetation sprite, because in
  practice those are new decorative plants (e.g. id 4006 in 7.0 maps).

Usage:
    python tools/sync_block_registry.py            # scan + update registries
    python tools/sync_block_registry.py --dry-run  # only report what is missing
    python tools/sync_block_registry.py --ids-json scan.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPORT_TOOL = ROOT / "tools" / "mdat_obj_export.py"
BLOCK_REGISTRY = ROOT / "docs" / "block_registry.json"
EXTERNAL_REGISTRY = ROOT / "docs" / "external_config_block_registry.json"
MAP_BLOCKS = ROOT / "assets" / "configs" / "assets" / "pda" / "map_blocks.json"
GEOMETRY_REPORT = ROOT / "docs" / "north_mines_geometry_by_block_id.json"
REPORT = ROOT / "docs" / "block_id_report.json"
DEFAULT_MAP_CACHE = Path.home() / "AppData" / "Roaming" / "EXBO" / "runtime" / "stalcraft" / "map_cache"

PLANT_FALLBACK_ICON = "customitems:tallgrass"
CUBE_FALLBACK_ICON = "stalcraft:stone"


def load_export_tool():
    spec = importlib.util.spec_from_file_location("mdat_obj_export", EXPORT_TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mdat_obj_export"] = module
    spec.loader.exec_module(module)
    return module


def map_cache_version_dir(root: Path) -> Path:
    numeric = [path for path in root.iterdir() if path.is_dir() and path.name.replace(".", "").isdigit()]
    if not numeric:
        raise SystemExit(f"no version directory inside {root}")
    return max(numeric, key=lambda path: [int(part) for part in path.name.split(".")])


def scan_ids(module, cache_dir: Path) -> tuple[Counter, Counter, Counter]:
    by_id: Counter = Counter()
    by_id_meta: Counter = Counter()
    chunk_count = 0
    files = sorted(cache_dir.glob("*/*.mdat"))
    for index, path in enumerate(files, 1):
        for chunk in module.read_region(path):
            chunk_count += 1
            for state in chunk.blocks.values():
                by_id[state.block_id] += 1
                by_id_meta[(state.block_id, state.meta)] += 1
        if index % 25 == 0:
            print(f"  scanned {index}/{len(files)} regions, {chunk_count} chunks", flush=True)
    print(f"  done: {chunk_count} chunks, {len(by_id)} unique block ids", flush=True)
    return by_id, by_id_meta, Counter({"chunks": chunk_count})


def load_ids_json(path: Path) -> tuple[Counter, Counter]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    by_id = Counter({int(key): int(value) for key, value in (raw.get("ids") or {}).items()})
    by_id_meta = Counter(
        {(int(key.split(":")[0]), int(key.split(":")[1])): int(value) for key, value in (raw.get("id_meta") or {}).items()}
    )
    return by_id, by_id_meta


def pda_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        block_id = int(row.get("id") or 0)
        if block_id <= 0:
            continue
        # Prefer the meta=0 row: it describes the default look of the block.
        if block_id not in out or int(row.get("meta") or 0) == 0:
            out[block_id] = row
    return out


def plant_config(block_id: int, icon: str) -> dict[str, Any]:
    return {
        "type": "block",
        "item_id": block_id,
        "icon": icon,
        "model": True,
        "simple_render": True,
        "collide": False,
    }


def cube_config(block_id: int, icon: str, *, opaque: bool = True) -> dict[str, Any]:
    return {
        "type": "block",
        "item_id": block_id,
        "icon": icon,
        "collide": True,
        "opaque": opaque,
        "rotatable": False,
        "min_x": 0,
        "min_y": 0,
        "min_z": 0,
        "max_x": 1,
        "max_y": 1,
        "max_z": 1,
    }


def build_entry(block_id: int, row: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    if row is None:
        return plant_config(block_id, PLANT_FALLBACK_ICON), "vegetation-fallback"
    icon = str(row.get("iconName") or "").strip()
    kind = str(row.get("type") or "NORMAL").upper()
    if not icon:
        return plant_config(block_id, PLANT_FALLBACK_ICON), "vegetation-fallback"
    if kind in {"GRASS", "PLANT", "FLOWER"}:
        return plant_config(block_id, icon), "plant"
    if kind == "WATER":
        return cube_config(block_id, icon, opaque=False), "water"
    if kind == "LEAVES":
        return cube_config(block_id, icon, opaque=False), "leaves"
    return cube_config(block_id, icon), "cube"


def table_ids(pda: dict[int, dict[str, Any]], geometry_report: Path) -> dict[int, str | None]:
    """Block ids mentioned by the bundled tables (PDA + geometry report)."""
    out: dict[int, str | None] = {block_id: None for block_id in pda}
    if geometry_report.is_file():
        raw = json.loads(geometry_report.read_text(encoding="utf-8"))
        for row in raw.get("by_id", []):
            if not isinstance(row, dict) or "block_id" not in row:
                continue
            block_id = int(row["block_id"])
            out.setdefault(block_id, row.get("config_type"))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add missing block ids to the external registry.")
    parser.add_argument("--map-cache", type=Path, default=DEFAULT_MAP_CACHE)
    parser.add_argument("--ids-json", type=Path, default=None, help="Reuse a previous scan (ids/id_meta).")
    parser.add_argument(
        "--from-tables",
        action="store_true",
        help="Also add ids that only the bundled PDA table / geometry report know about (no map scan needed).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    known_block_registry = json.loads(BLOCK_REGISTRY.read_text(encoding="utf-8")) if BLOCK_REGISTRY.is_file() else {}
    external_raw = json.loads(EXTERNAL_REGISTRY.read_text(encoding="utf-8")) if EXTERNAL_REGISTRY.is_file() else {}
    known_external = {int(key) for key in external_raw}
    known = {int(key) for key in known_block_registry} | known_external

    if args.ids_json is not None:
        by_id, by_id_meta = load_ids_json(args.ids_json)
    elif args.from_tables:
        # Union of the bundled tables only: no map scan, instant result.
        by_id, by_id_meta = Counter(), Counter()
    else:
        module = load_export_tool()
        cache_dir = map_cache_version_dir(args.map_cache)
        print(f"scanning {cache_dir}", flush=True)
        by_id, by_id_meta, _ = scan_ids(module, cache_dir)

    pda = pda_rows(MAP_BLOCKS)
    from_tables: dict[int, str | None] = table_ids(pda, GEOMETRY_REPORT) if args.from_tables else {}
    candidates = set(by_id) | set(from_tables)
    missing = sorted(block_id for block_id in candidates if block_id not in known)
    print(f"block ids considered: {len(candidates)}; already registered: {len(candidates) - len(missing)}; missing: {len(missing)}")
    for block_id in missing:
        print(f"  id={block_id} count={by_id.get(block_id, 0)}", flush=True)

    report: dict[str, Any] = {"missing": {}, "added": 0}
    history: list[dict[str, Any]] = []
    if REPORT.is_file():
        try:
            history = json.loads(REPORT.read_text(encoding="utf-8")).get("history") or []
        except (OSError, ValueError):
            history = []
    added = 0
    for block_id in missing:
        entry_config, source = build_entry(block_id, pda.get(block_id))
        meta_top = [
            {"meta": meta, "count": count}
            for (bid, meta), count in sorted(by_id_meta.items(), key=lambda item: item[1], reverse=True)
            if bid == block_id
        ][:8]
        entry = {
            "count": by_id.get(block_id, 0),
            "meta_top": meta_top,
            "source": source,
            "icon": entry_config.get("icon"),
            "pda": pda.get(block_id),
            "tables": {"pda": block_id in pda, "geometry_report": block_id in from_tables},
        }
        report["missing"][str(block_id)] = entry
        if not args.dry_run:
            external_raw[str(block_id)] = {
                "block_id": block_id,
                "config_type": "block",
                "config": entry_config,
            }
            history = [row for row in history if str(row.get("block_id")) != str(block_id)]
            history.append({"block_id": block_id, **entry})
            added += 1

    if not args.dry_run and added:
        external_raw = dict(sorted(external_raw.items(), key=lambda item: int(item[0])))
        EXTERNAL_REGISTRY.write_text(json.dumps(external_raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["added"] = added
    report["total_ids"] = len(candidates)
    report["scanned_ids"] = len(by_id)
    report["history"] = history
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"added {added} registr{'y' if added == 1 else 'ies'}; report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
