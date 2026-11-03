from __future__ import annotations

import json
import math
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SECTOR_SIZE = 4096
REGION_BITS = 5
REGION_SIZE = 1 << REGION_BITS
REGION_CHUNKS = REGION_SIZE * REGION_SIZE
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
REGION_RE = re.compile(r"^reg\.(-?\d+)\.(-?\d+)\.mdat$")


@dataclass(frozen=True)
class RegionFormat:
    """Binary layout of a region file.

    ``header_ints`` ints per chunk-entry in the region header.
    ``payload_key_ints`` ints stored at the start of the chunk payload before
    the primary/add masks (0 for legacy files, 2 for 7.0+ files where the
    header entry and payload share an 8-byte chunk key).
    """

    name: str
    header_ints: int
    payload_key_ints: int

    @property
    def entry_size(self) -> int:
        return self.header_ints * 4

    @property
    def header_size(self) -> int:
        return REGION_CHUNKS * self.entry_size

    @property
    def min_sector(self) -> int:
        return math.ceil(self.header_size / SECTOR_SIZE)

    @property
    def payload_header_size(self) -> int:
        # declared length + optional chunk key + primary/add/raw/compressed + zstd magic
        return 4 + self.payload_key_ints * 4 + 20


# 5.0/6.0: header entry = sector, sector_count, uuid0..uuid3; payload starts
# with primary_mask/add_mask.
LEGACY_FORMAT = RegionFormat("6.0", header_ints=6, payload_key_ints=0)
# 7.0: header entry = sector, sector_count, key0, key1; the same 8-byte key is
# repeated inside the payload before the primary/add masks.
V7_FORMAT = RegionFormat("7.0", header_ints=4, payload_key_ints=2)
REGION_FORMATS = (V7_FORMAT, LEGACY_FORMAT)

# Kept for backwards compatibility with callers that used the legacy constants.
HEADER_INTS = LEGACY_FORMAT.header_ints
HEADER_ENTRY_SIZE = LEGACY_FORMAT.entry_size
HEADER_SIZE = LEGACY_FORMAT.header_size


def detect_region_format(data: bytes) -> RegionFormat | None:
    """Pick the region layout that validates against the file contents.

    Both candidate layouts are verified entry-by-entry: sector bounds, payload
    masks and the zstd frame marker have to line up for every populated entry.
    The 7.0 layout additionally requires the header key to match the copy
    stored inside the payload, which makes the detection unambiguous.
    """
    best: tuple[int, RegionFormat] | None = None
    for fmt in REGION_FORMATS:
        if len(data) < fmt.header_size + 4:
            continue
        valid = 0
        invalid = 0
        for local_index in range(REGION_CHUNKS):
            offset = local_index * fmt.entry_size
            values = struct.unpack_from(f">{fmt.header_ints}i", data, offset)
            sector, sector_count = values[0], values[1]
            if sector == 0 and sector_count == 0:
                continue
            if sector < fmt.min_sector or sector_count <= 0:
                invalid += 1
                break
            payload_pos = sector * SECTOR_SIZE
            payload_header = fmt.payload_header_size
            if payload_pos + payload_header > len(data):
                invalid += 1
                break
            declared_len = struct.unpack_from(">i", data, payload_pos)[0]
            if declared_len <= 0 or declared_len > len(data):
                invalid += 1
                break
            cursor = payload_pos + 4
            if fmt.payload_key_ints:
                key = struct.unpack_from(f">{fmt.payload_key_ints}i", data, cursor)
                if key != values[2:2 + fmt.payload_key_ints]:
                    invalid += 1
                    break
                cursor += fmt.payload_key_ints * 4
            primary_mask, add_mask, raw_len, compressed_len = struct.unpack_from(">4i", data, cursor)
            if not (0 <= primary_mask <= 0xFFFF and 0 <= add_mask <= 0xFFFF):
                invalid += 1
                break
            if raw_len <= 0 or raw_len > 8 * 1024 * 1024:
                invalid += 1
                break
            if compressed_len <= 0 or cursor + 16 + compressed_len > len(data):
                invalid += 1
                break
            if data[cursor + 16: cursor + 20] != ZSTD_MAGIC:
                invalid += 1
                break
            valid += 1
        if invalid == 0 and valid > 0 and (best is None or valid > best[0]):
            best = (valid, fmt)
    return best[1] if best else None


def resolve_region_format(data: bytes) -> RegionFormat | None:
    fmt = detect_region_format(data)
    if fmt is not None:
        return fmt
    if len(data) < LEGACY_FORMAT.header_size:
        return V7_FORMAT
    return LEGACY_FORMAT


@dataclass(frozen=True)
class ChunkRef:
    x: int
    z: int
    world: str
    region_x: int
    region_z: int
    local_index: int
    sector: int
    sectors: int
    file: Path
    format: str = LEGACY_FORMAT.name

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "z": self.z,
            "world": self.world,
            "region_x": self.region_x,
            "region_z": self.region_z,
            "local_index": self.local_index,
            "file": str(self.file),
            "format": self.format,
        }


def parse_region_name(path: Path) -> tuple[int, int] | None:
    match = REGION_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def iter_region_files(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("reg.*.*.mdat") if parse_region_name(path))


def list_worlds(map_cache: Path) -> list[dict]:
    worlds: list[dict] = []
    if not map_cache.is_dir():
        return worlds
    for child in sorted(map_cache.iterdir()):
        if not child.is_dir():
            continue
        regions = iter_region_files(child)
        if not regions:
            continue
        worlds.append({
            "name": child.name,
            "path": str(child),
            "regions": len(regions),
            "mtime": max((p.stat().st_mtime for p in regions), default=child.stat().st_mtime),
        })
    return worlds


def scan_region(path: Path, world: str) -> list[ChunkRef]:
    coords = parse_region_name(path)
    if coords is None:
        return []
    data = path.read_bytes()
    fmt = resolve_region_format(data)
    if fmt is None or len(data) < fmt.header_size:
        return []
    rx, rz = coords
    chunks: list[ChunkRef] = []
    for local_index in range(REGION_CHUNKS):
        offset = local_index * fmt.entry_size
        values = struct.unpack_from(f">{fmt.header_ints}i", data, offset)
        sector, sector_count = values[0], values[1]
        if sector < fmt.min_sector or sector_count <= 0:
            continue
        payload_pos = sector * SECTOR_SIZE
        if payload_pos + 4 > len(data):
            continue
        local_x = local_index & (REGION_SIZE - 1)
        local_z = local_index >> REGION_BITS
        chunks.append(ChunkRef(
            x=rx * REGION_SIZE + local_x,
            z=rz * REGION_SIZE + local_z,
            world=world,
            region_x=rx,
            region_z=rz,
            local_index=local_index,
            sector=sector,
            sectors=sector_count,
            file=path,
            format=fmt.name,
        ))
    return chunks


def scan_map_cache(map_cache: Path, *, excluded_worlds: set[str] | None = None) -> dict:
    excluded_worlds = excluded_worlds or set()
    worlds = list_worlds(map_cache)
    source_chunks: list[ChunkRef] = []
    errors: list[dict] = []
    world_summaries: list[dict] = []
    for world in worlds:
        name = world["name"]
        regions = iter_region_files(Path(world["path"]))
        raw_chunks: list[ChunkRef] = []
        for region in regions:
            try:
                raw_chunks.extend(scan_region(region, name))
            except Exception as exc:
                errors.append({"world": name, "file": str(region), "error": str(exc)})
        world_summaries.append({
            **world,
            "chunks": len(raw_chunks),
            "excluded": name in excluded_worlds,
        })
        source_chunks.extend(raw_chunks)

    chunks = [chunk for chunk in source_chunks if chunk.world not in excluded_worlds]

    by_coord: dict[tuple[int, int], list[ChunkRef]] = {}
    for chunk in chunks:
        by_coord.setdefault((chunk.x, chunk.z), []).append(chunk)

    deduped: list[dict] = []
    overlap_count = 0
    for (_x, _z), refs in sorted(by_coord.items()):
        # Prefer newest source file for overlapping worlds; excluding worlds
        # from the UI changes this choice without mutating source data.
        refs = sorted(refs, key=lambda c: (c.file.stat().st_mtime, c.world), reverse=True)
        chosen = refs[0]
        row = chosen.to_dict()
        row["overlaps"] = [ref.world for ref in refs]
        row["overlap_count"] = len(refs)
        if len(refs) > 1:
            overlap_count += 1
        deduped.append(row)

    xs = [c["x"] for c in deduped]
    zs = [c["z"] for c in deduped]
    return {
        "map_cache": str(map_cache),
        "worlds": world_summaries,
        "regions": sum(w["regions"] for w in world_summaries if not w["excluded"]),
        "source_chunks": [c.to_dict() for c in source_chunks],
        "chunks": deduped,
        "raw_chunk_count": len(chunks),
        "chunk_count": len(deduped),
        "overlap_chunks": overlap_count,
        "excluded_worlds": sorted(excluded_worlds),
        "bounds": {
            "min_x": min(xs) if xs else 0,
            "max_x": max(xs) if xs else 0,
            "min_z": min(zs) if zs else 0,
            "max_z": max(zs) if zs else 0,
        },
        "errors": errors,
    }


def write_filtered_scan_chunks(scan: dict, out_dir: Path, selected: Iterable[tuple[int, int]]) -> int:
    wanted = set(selected)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_rows = {
        (int(row["x"]), int(row["z"])): row
        for row in scan.get("chunks") or []
        if (int(row["x"]), int(row["z"])) in wanted
    }
    by_file: dict[Path, list[dict]] = {}
    for row in chunk_rows.values():
        by_file.setdefault(Path(row["file"]), []).append(row)

    staged: dict[tuple[int, int], tuple[RegionFormat, list[tuple[int, int, tuple[int, ...], bytes]]]] = {}
    for source, rows in by_file.items():
        coords = parse_region_name(source)
        if coords is None:
            continue
        data = source.read_bytes()
        fmt = resolve_region_format(data)
        if fmt is None:
            continue
        rx, rz = coords
        bucket = staged.setdefault(coords, (fmt, []))
        for row in sorted(rows, key=lambda item: (int(item["x"]), int(item["z"]))):
            x = int(row["x"])
            z = int(row["z"])
            local_x = x - rx * REGION_SIZE
            local_z = z - rz * REGION_SIZE
            if not (0 <= local_x < REGION_SIZE and 0 <= local_z < REGION_SIZE):
                continue
            local_index = local_z * REGION_SIZE + local_x
            entry_offset = local_index * fmt.entry_size
            if entry_offset + fmt.entry_size > len(data):
                continue
            values = struct.unpack_from(f">{fmt.header_ints}i", data, entry_offset)
            sector, sector_count = values[0], values[1]
            if sector <= 0 or sector_count <= 0:
                continue
            start = sector * SECTOR_SIZE
            end = start + sector_count * SECTOR_SIZE
            if start >= len(data):
                continue
            payload = data[start:end]
            if len(payload) < sector_count * SECTOR_SIZE:
                payload += bytes(sector_count * SECTOR_SIZE - len(payload))
            bucket[1].append((local_index, sector_count, tuple(values[2:]), payload))

    written = 0
    for (rx, rz), (fmt, payload_rows) in staged.items():
        header = bytearray(fmt.header_size)
        payloads = bytearray()
        next_sector = fmt.min_sector
        seen_local: set[int] = set()
        for local_index, sector_count, key_words, payload in sorted(payload_rows, key=lambda item: item[0]):
            if local_index in seen_local:
                continue
            seen_local.add(local_index)
            struct.pack_into(f">{fmt.header_ints}i", header, local_index * fmt.entry_size, next_sector, sector_count, *key_words)
            payloads.extend(payload)
            next_sector += sector_count
            written += 1
        if payloads:
            (out_dir / f"reg.{rx}.{rz}.mdat").write_bytes(bytes(header) + bytes(payloads))
    return written


def write_filtered_map_cache(source_map_cache: Path, out_dir: Path, selected: Iterable[tuple[int, int]]) -> int:
    scan = scan_map_cache(source_map_cache)
    return write_filtered_scan_chunks(scan, out_dir, selected)


def save_scan(path: Path, scan: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scan, ensure_ascii=False, indent=2), encoding="utf-8")
