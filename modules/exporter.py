from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .glb_merge import MAX_GLB_BYTES, merge_glbs
from .mdat_scan import write_filtered_scan_chunks
from .paths import EXPORTS_DIR, MDAT_EXPORT_SCRIPT, ROOT, TMP_DIR, find_blender, game_texture_paths

DEFAULT_BATCH_CHUNKS = 500
MIN_BATCH_CHUNKS = 25
MAX_BATCH_CHUNKS = 2000
MEMORY_WARN_CHUNKS = 750
# Leave a little headroom below the hard uint32 GLB container limit.
MERGE_BUDGET_BYTES = MAX_GLB_BYTES - 64 * 1024 * 1024


@dataclass
class ExportJob:
    job_id: str
    name: str
    game_dir: str
    map_cache: str
    scan: dict[str, Any]
    selected: list[tuple[int, int]]
    out_dir: Path
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    log: list[str] = field(default_factory=list)
    glb: Path | None = None
    blend: Path | None = None
    batches: int = 0
    parts: list[str] = field(default_factory=list)
    parts_bytes: int = 0
    mode: str = "blender"

    @property
    def running(self) -> bool:
        return self.finished_at is None and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "running": self.running,
            "success": self.finished_at is not None and self.error is None,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "chunks": len(self.selected),
            "batches": self.batches,
            "mode": self.mode,
            "glb": str(self.glb) if self.glb else None,
            "blend": str(self.blend) if self.blend else None,
            "parts": self.parts,
            "parts_bytes": self.parts_bytes,
            "log": self.log[-80:],
        }


class ExportRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, ExportJob] = {}

    def list(self) -> list[ExportJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda job: job.started_at, reverse=True)
        return jobs

    def submit(self, *, game_dir: Path, map_cache: Path, scan: dict[str, Any], chunks: list[tuple[int, int]], options: dict[str, Any]) -> ExportJob:
        if not chunks:
            raise ValueError("select at least one chunk")
        if not MDAT_EXPORT_SCRIPT.is_file():
            raise FileNotFoundError(str(MDAT_EXPORT_SCRIPT))
        name = safe_name(str(options.get("name") or f"mdat_{int(time.time())}"))
        job_id = uuid.uuid4().hex[:12]
        out_dir = EXPORTS_DIR / f"{name}_{job_id}"
        job = ExportJob(
            job_id=job_id,
            name=name,
            game_dir=str(game_dir),
            map_cache=str(map_cache),
            scan=scan,
            selected=chunks,
            out_dir=out_dir,
        )
        job.batches = len(split_batches(chunks, clamp_batch_chunks(options.get("batch_chunks") or options.get("batchChunks"))))
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(target=self._run, args=(job, options), daemon=True)
        thread.start()
        return job

    def _run(self, job: ExportJob, options: dict[str, Any]) -> None:
        try:
            run_export(job, options)
        except MemoryError:
            job.error = "MemoryError: not enough RAM for this batch, lower the batch size"
            job.log.append(job.error)
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            job.log.append(job.error)
        finally:
            job.finished_at = time.time()
            job.out_dir.mkdir(parents=True, exist_ok=True)
            (job.out_dir / "job.json").write_text(json.dumps(job.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in value).strip(" .")
    return cleaned or "mdat_export"


def clamp_batch_chunks(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return DEFAULT_BATCH_CHUNKS
    return max(MIN_BATCH_CHUNKS, min(MAX_BATCH_CHUNKS, number))


def _spread_bits(value: int) -> int:
    value &= 0xFFFF
    value = (value | (value << 8)) & 0x00FF00FF
    value = (value | (value << 4)) & 0x0F0F0F0F
    value = (value | (value << 2)) & 0x33333333
    value = (value | (value << 1)) & 0x55555555
    return value


def chunk_order_key(chunk: tuple[int, int]) -> int:
    """Z-order (Morton) key so each batch covers a compact square-ish area."""
    x = (int(chunk[0]) + 32768) & 0xFFFF
    z = (int(chunk[1]) + 32768) & 0xFFFF
    return (_spread_bits(x) << 1) | _spread_bits(z)


def split_batches(chunks: list[tuple[int, int]], batch_size: int) -> list[list[tuple[int, int]]]:
    ordered = sorted(chunks, key=chunk_order_key)
    return [ordered[index:index + batch_size] for index in range(0, len(ordered), batch_size)]


def group_parts(paths: list[Path], budget: int) -> list[list[Path]]:
    groups: list[list[Path]] = []
    current: list[Path] = []
    current_bytes = 0
    for path in paths:
        size = path.stat().st_size
        if current and current_bytes + size > budget:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(path)
        current_bytes += size
    if current:
        groups.append(current)
    return groups


def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{value} B"


def run_export(job: ExportJob, options: dict[str, Any]) -> None:
    game_dir = Path(job.game_dir)
    job.out_dir.mkdir(parents=True, exist_ok=True)
    mode = str(options.get("mode") or "blender").lower()
    if mode not in {"blender", "unreal"}:
        mode = "blender"
    job.mode = mode
    try:
        unreal_scale = float(options.get("unreal_scale") or 100.0)
    except (TypeError, ValueError):
        unreal_scale = 100.0
    if unreal_scale <= 0:
        unreal_scale = 100.0
    batch_size = clamp_batch_chunks(options.get("batch_chunks") or options.get("batchChunks"))
    batches = split_batches(job.selected, batch_size)
    job.batches = len(batches)
    job.log.append(f"mode={mode} chunks={len(job.selected)} batches={len(batches)} batch_chunks={batch_size}")
    if mode == "unreal":
        job.log.append(f"unreal_scale={unreal_scale:g} (1 block = {unreal_scale:g} UE units)")
    if batch_size > MEMORY_WARN_CHUNKS:
        job.log.append(f"warning: {batch_size} chunks per batch is heavy, 300-500 is safer on 32 GB RAM")

    paths = game_texture_paths(game_dir)
    parts_dir = job.out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []

    for index, batch in enumerate(batches, 1):
        tmp_cache = TMP_DIR / f"{job.job_id}_cache_{index:03d}"
        written = write_filtered_scan_chunks(job.scan, tmp_cache, batch)
        if written <= 0:
            shutil.rmtree(tmp_cache, ignore_errors=True)
            raise RuntimeError(f"batch {index}/{len(batches)}: selected chunks were not found in map_cache")
        part_glb = parts_dir / f"part_{index:03d}.glb"
        cmd = [
            sys.executable,
            str(MDAT_EXPORT_SCRIPT),
            str(tmp_cache),
            str(part_glb),
            "--double-sided",
            "--force-mask",
            "--hide-invisible",
            "--texarr", str(paths["texarr"]),
            "--ctm-dir", str(paths["ctm_dir"]),
            "--map-blocks", str(paths["map_blocks"]),
            "--weather-palettes", str(paths["weather_palettes"]),
        ]
        if mode == "unreal" and unreal_scale != 1.0:
            cmd.extend(["--scale", f"{unreal_scale:g}"])
        job.log.append(f"[batch {index}/{len(batches)}] chunks={written} glb={part_glb.name}")
        job.log.append(" ".join(quote_part(part) for part in cmd))
        try:
            run_logged(cmd, cwd=MDAT_EXPORT_SCRIPT.parent, log=job.log)
        finally:
            shutil.rmtree(tmp_cache, ignore_errors=True)
        if not part_glb.is_file():
            raise RuntimeError(f"GLB was not written: {part_glb}")
        part_paths.append(part_glb)
        job.log.append(f"[batch {index}/{len(batches)}] ok size={format_bytes(part_glb.stat().st_size)}")

    job.parts_bytes = sum(path.stat().st_size for path in part_paths)
    groups = group_parts(part_paths, MERGE_BUDGET_BYTES)
    outputs: list[Path] = []
    for group_index, group in enumerate(groups, 1):
        if len(groups) == 1:
            target = job.out_dir / f"{job.name}.glb"
        else:
            target = job.out_dir / f"{job.name}_{group_index:02d}.glb"
        if len(group) == 1:
            if group[0] != target:
                shutil.move(str(group[0]), str(target))
            size = target.stat().st_size
        else:
            stats = merge_glbs(group, target, part_names=[path.stem for path in group])
            size = stats.bytes
            job.log.append(
                f"[merge {group_index}/{len(groups)}] parts={stats.parts} vertices={stats.vertices} "
                f"faces={stats.faces} materials={stats.materials} images={stats.images} size={format_bytes(stats.bytes)}"
            )
        outputs.append(target)
        job.log.append(f"glb={target} size={format_bytes(size)}")
        # Free the raw batch files as soon as they are merged so a huge job
        # does not need twice its final size in disk space.
        for path in group:
            if path != target and path.is_file():
                path.unlink(missing_ok=True)

    if mode == "unreal":
        shutil.rmtree(parts_dir, ignore_errors=True)
        job.glb = outputs[0] if len(outputs) == 1 else None
        job.parts = [str(path) for path in outputs]
        helper = write_unreal_helpers(job.out_dir, outputs)
        job.log.append(f"unreal_scale={unreal_scale:g}")
        job.log.append(f"ue_import_script={helper}")
        job.log.append(
            "unreal export ready: import the .glb files through the generated ue_import.py "
            "(UE 5.x, glTF/Interchange importer enabled)"
        )
        return

    if len(outputs) == 1:
        shutil.rmtree(parts_dir, ignore_errors=True)
        job.glb = outputs[0]
        job.parts = [str(outputs[0])]
        blender = find_blender(str(options.get("blender") or "").strip() or None)
        if not blender.is_file():
            raise FileNotFoundError(f"Blender not found: {blender}")
        blend = job.out_dir / f"{job.name}.blend"
        blender_script = ROOT / "modules" / "glb_to_blend.py"
        blender_cmd = [
            str(blender),
            "--background",
            "--python", str(blender_script),
            "--",
            "--input", str(job.glb),
            "--output", str(blend),
        ]
        job.log.append(" ".join(quote_part(part) for part in blender_cmd))
        run_logged(blender_cmd, cwd=job.out_dir, log=job.log)
        if not blend.is_file():
            raise RuntimeError(f"Blend was not written: {blend}")
        job.blend = blend
        job.log.append(f"blend={blend}")
        return

    job.parts = [str(path) for path in outputs]
    shutil.rmtree(parts_dir, ignore_errors=True)
    job.log.append(
        f"split into {len(outputs)} glb files because one GLB cannot exceed 4 GiB; "
        f"total {format_bytes(job.parts_bytes)}"
    )
    job.log.append("blend skipped: import the glb parts separately (one .blend with everything would not fit in RAM)")


def write_unreal_helpers(out_dir: Path, outputs: list[Path]) -> Path:
    """Generate an Unreal Engine import script next to the exported GLB files."""
    script_path = out_dir / "ue_import.py"
    file_list = ",\n    ".join(repr(str(path)) for path in outputs)
    script = f'''"""Import the MapDump GLB parts into Unreal Engine 5.

Run from the editor: Tools -> Execute Python Script... -> select this file.
Requirements: UE 5.x with the glTF importer (Interchange) plugin enabled.
"""
import unreal

GLB_FILES = [
    {file_list},
]
DESTINATION_PATH = "/Game/MapDump"
SPAWN_IN_LEVEL = True


def import_glbs():
    tasks = []
    for path in GLB_FILES:
        task = unreal.AssetImportTask()
        task.filename = path
        task.destination_path = DESTINATION_PATH
        task.automated = True
        task.save = True
        task.replace_existing = True
        tasks.append(task)
    unreal.AssetToolsHelpers.get_asset_tools().import_assets_automated(tasks)
    unreal.log(f"MapDump: imported {{len(tasks)}} glb file(s)")


def spawn_in_level():
    if not SPAWN_IN_LEVEL:
        return
    try:
        subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    except Exception as exc:  # older engines
        unreal.log_warning(f"MapDump: actor subsystem unavailable: {{exc}}")
        return
    spawned = 0
    for asset_path in unreal.EditorAssetLibrary.list_assets(DESTINATION_PATH, recursive=True):
        data = unreal.EditorAssetLibrary.find_asset_data(asset_path)
        if data.asset_class_path.asset_name != "StaticMesh":
            continue
        mesh = data.get_asset()
        subsystem.spawn_actor_from_object(mesh, unreal.Vector(0.0, 0.0, 0.0), unreal.Rotator(0.0, 0.0, 0.0))
        spawned += 1
    unreal.log(f"MapDump: spawned {{spawned}} static mesh actor(s)")


import_glbs()
spawn_in_level()
'''
    script_path.write_text(script, encoding="utf-8")

    notes = out_dir / "UE_IMPORT.md"
    notes.write_text(
        "# Импорт в Unreal Engine 5\n\n"
        "1. Включите импорт glTF: Edit -> Plugins -> включите `glTF Importer` / `Interchange` (обычно уже включён в UE 5.x).\n"
        "2. Tools -> Execute Python Script... -> выберите `ue_import.py` из этой папки.\n"
        "   Либо скопируйте его содержимое в Output Log (режим Python).\n"
        "3. Ассеты появятся в `/Game/MapDump`, статические меши будут расставлены в текущем уровне в мировых координатах.\n\n"
        "Вершины уже пересчитаны в сантиметры (1 блок = 100 UE-единиц), оси и масштаб дополнительно настраивать не нужно.\n"
        "Если карта кажется в 100 раз больше или меньше, поменяйте поле `unreal scale` при экспорте.\n\n"
        "Текстуры встроены в GLB: Unreal создаст материалы и текстуры автоматически при импорте.\n",
        encoding="utf-8",
    )
    return script_path


def run_logged(cmd: list[str], *, cwd: Path, log: list[str]) -> None:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log.append(line.rstrip())
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"command failed with exit code {rc}")


def quote_part(value: object) -> str:
    text = str(value)
    if " " in text or "\t" in text:
        return f'"{text}"'
    return text
