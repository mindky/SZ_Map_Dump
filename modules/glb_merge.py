from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

GLB_MAGIC = 0x46546C67
CHUNK_TYPE_JSON = 0x4E4F534A
CHUNK_TYPE_BIN = 0x004E4942
ALIGNMENT = 4
# The GLB container stores its total length as uint32, so a single file can
# never exceed 4 GiB no matter how much RAM the machine has.
MAX_GLB_BYTES = 0xFFFFFFFF
COPY_BLOCK = 8 * 1024 * 1024

TEXTURE_SLOT_KEYS = {
    "baseColorTexture",
    "metallicRoughnessTexture",
    "normalTexture",
    "occlusionTexture",
    "emissiveTexture",
    "clearcoatTexture",
    "clearcoatRoughnessTexture",
    "clearcoatNormalTexture",
    "transmissionTexture",
    "thicknessTexture",
    "specularTexture",
    "specularColorTexture",
    "sheenColorTexture",
    "sheenRoughnessTexture",
    "iridescenceTexture",
    "iridescenceThicknessTexture",
    "anisotropyTexture",
    "diffuseTexture",
    "specularGlossinessTexture",
}


class GlbMergeError(RuntimeError):
    pass


@dataclass
class GlbPart:
    path: Path
    document: dict[str, Any]
    bin_offset: int
    bin_length: int


@dataclass
class _CopyOp:
    part_index: int
    source_offset: int
    length: int
    target_offset: int


@dataclass
class MergeStats:
    parts: int = 0
    vertices: int = 0
    faces: int = 0
    materials: int = 0
    images: int = 0
    bytes: int = 0
    part_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parts": self.parts,
            "vertices": self.vertices,
            "faces": self.faces,
            "materials": self.materials,
            "images": self.images,
            "bytes": self.bytes,
        }


def align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def read_glb(path: Path) -> GlbPart:
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) < 12:
            raise GlbMergeError(f"not a GLB file: {path}")
        magic, _version, length = struct.unpack("<III", header)
        if magic != GLB_MAGIC:
            raise GlbMergeError(f"not a GLB file: {path}")
        document: Optional[dict[str, Any]] = None
        bin_offset = 0
        bin_length = 0
        while handle.tell() < max(length, 12):
            chunk_header = handle.read(8)
            if len(chunk_header) < 8:
                break
            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            payload_offset = handle.tell()
            if chunk_type == CHUNK_TYPE_JSON:
                payload = handle.read(chunk_length)
                text = payload.decode("utf-8").rstrip(" \t\r\n\x00")
                document = json.loads(text)
            elif chunk_type == CHUNK_TYPE_BIN:
                bin_offset = payload_offset
                bin_length = chunk_length
                handle.seek(chunk_length, 1)
            else:
                handle.seek(chunk_length, 1)
    if document is None:
        raise GlbMergeError(f"GLB has no JSON chunk: {path}")
    if bin_offset and bin_offset + bin_length > path.stat().st_size:
        raise GlbMergeError(f"GLB is truncated: {path}")
    return GlbPart(path=path, document=document, bin_offset=bin_offset, bin_length=bin_length)


def _remap_texture_slots(value: Any, texture_map: dict[int, int]) -> Any:
    if isinstance(value, list):
        return [_remap_texture_slots(item, texture_map) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in TEXTURE_SLOT_KEYS and isinstance(item, dict) and "index" in item:
            remapped = dict(item)
            old_index = remapped.get("index")
            if isinstance(old_index, int) and old_index in texture_map:
                remapped["index"] = texture_map[old_index]
            result[key] = _remap_texture_slots(remapped, texture_map)
        else:
            result[key] = _remap_texture_slots(item, texture_map)
    return result


class _MergedDocument:
    def __init__(self) -> None:
        self.buffer_views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []
        self.materials: list[dict[str, Any]] = []
        self.images: list[dict[str, Any]] = []
        self.textures: list[dict[str, Any]] = []
        self.samplers: list[dict[str, Any]] = []
        self.meshes: list[dict[str, Any]] = []
        self.nodes: list[dict[str, Any]] = []
        self.scene_nodes: list[int] = []

        self.material_cache: dict[str, int] = {}
        self.image_cache: dict[tuple[Any, ...], int] = {}
        self.texture_cache: dict[tuple[Any, ...], int] = {}
        self.sampler_cache: dict[str, int] = {}

        self.copy_ops: list[_CopyOp] = []
        self.bin_size = 0
        self.vertices = 0
        self.faces = 0

    def add_buffer_view(self, part_index: int, part: GlbPart, view: dict[str, Any]) -> int:
        length = int(view.get("byteLength") or 0)
        offset = align(self.bin_size)
        merged: dict[str, Any] = {"buffer": 0, "byteOffset": offset, "byteLength": length}
        for key in ("target", "byteStride", "name"):
            if key in view:
                merged[key] = view[key]
        index = len(self.buffer_views)
        self.buffer_views.append(merged)
        source_offset = int(view.get("byteOffset") or 0)
        if length:
            self.copy_ops.append(_CopyOp(part_index, source_offset, length, offset))
        self.bin_size = offset + length
        return index


def merge_glbs(
    parts: Sequence[Path],
    output_path: Path,
    *,
    max_bytes: int = MAX_GLB_BYTES,
    part_names: Optional[Sequence[str]] = None,
) -> MergeStats:
    """Merge GLB files into one, streaming the binary payload from disk."""
    parts = [Path(part) for part in parts]
    if not parts:
        raise GlbMergeError("nothing to merge")
    document = _MergedDocument()
    stats = MergeStats(parts=len(parts))
    parsed: list[GlbPart] = []
    scene_node_groups: list[list[int]] = []

    for part_index, path in enumerate(parts):
        try:
            part = read_glb(path)
        except GlbMergeError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise GlbMergeError(f"failed to read {path}: {exc}") from exc
        parsed.append(part)
        gltf = part.document

        sampler_map: dict[int, int] = {}
        for old_index, sampler in enumerate(gltf.get("samplers") or []):
            key = json.dumps(sampler, sort_keys=True)
            merged_index = document.sampler_cache.get(key)
            if merged_index is None:
                merged_index = len(document.samplers)
                document.samplers.append(dict(sampler))
                document.sampler_cache[key] = merged_index
            sampler_map[old_index] = merged_index

        views = gltf.get("bufferViews") or []
        images = gltf.get("images") or []
        view_map: dict[int, int] = {}

        # Image buffer views are deduplicated by content, so a texture shared by
        # every part is stored once in the merged file instead of once per part.
        image_map: dict[int, int] = {}
        with part.path.open("rb") as image_handle:
            for old_index, image in enumerate(images):
                name = image.get("name") or ""
                view_index = image.get("bufferView")
                view = views[view_index] if isinstance(view_index, int) and 0 <= view_index < len(views) else {}
                payload = b""
                length = int(view.get("byteLength") or 0)
                if length and part.bin_offset:
                    image_handle.seek(part.bin_offset + int(view.get("byteOffset") or 0))
                    payload = image_handle.read(length)
                key = (name, image.get("mimeType"), len(payload), zlib.crc32(payload))
                merged_index = document.image_cache.get(key)
                if merged_index is None:
                    merged_image = {k: v for k, v in image.items() if k != "bufferView"}
                    merged_image.setdefault("name", name)
                    if view:
                        merged_image["bufferView"] = document.add_buffer_view(part_index, part, view)
                        if isinstance(view_index, int):
                            view_map[view_index] = merged_image["bufferView"]
                    merged_index = len(document.images)
                    document.images.append(merged_image)
                    document.image_cache[key] = merged_index
                else:
                    if isinstance(view_index, int):
                        view_map[view_index] = int(document.images[merged_index]["bufferView"])
                image_map[old_index] = merged_index

        texture_map: dict[int, int] = {}
        for old_index, texture in enumerate(gltf.get("textures") or []):
            source = texture.get("source")
            sampler = texture.get("sampler")
            mapped_source = image_map.get(source, source) if isinstance(source, int) else source
            mapped_sampler = sampler_map.get(sampler, sampler) if isinstance(sampler, int) else sampler
            key = (mapped_source, mapped_sampler)
            merged_index = document.texture_cache.get(key)
            if merged_index is None:
                merged_index = len(document.textures)
                document.textures.append({"sampler": mapped_sampler, "source": mapped_source})
                document.texture_cache[key] = merged_index
            texture_map[old_index] = merged_index

        material_map: dict[int, int] = {}
        for old_index, material in enumerate(gltf.get("materials") or []):
            remapped = _remap_texture_slots(material, texture_map)
            key = json.dumps(remapped, sort_keys=True)
            merged_index = document.material_cache.get(key)
            if merged_index is None:
                merged_index = len(document.materials)
                document.materials.append(remapped)
                document.material_cache[key] = merged_index
            material_map[old_index] = merged_index

        for old_index, view in enumerate(views):
            if old_index in view_map:
                continue
            view_map[old_index] = document.add_buffer_view(part_index, part, view)

        accessor_map: dict[int, int] = {}
        for old_index, accessor in enumerate(gltf.get("accessors") or []):
            merged_accessor = dict(accessor)
            view_index = accessor.get("bufferView")
            if isinstance(view_index, int) and view_index in view_map:
                merged_accessor["bufferView"] = view_map[view_index]
            merged_accessor.pop("sparse", None)
            accessor_map[old_index] = len(document.accessors)
            document.accessors.append(merged_accessor)

        mesh_map: dict[int, int] = {}
        for old_index, mesh in enumerate(gltf.get("meshes") or []):
            primitives = []
            for primitive in mesh.get("primitives") or []:
                merged_primitive = dict(primitive)
                attributes = {}
                for attribute_name, accessor_index in (primitive.get("attributes") or {}).items():
                    attributes[attribute_name] = accessor_map.get(accessor_index, accessor_index)
                    if attribute_name == "POSITION" and isinstance(accessor_index, int):
                        accessor = (gltf.get("accessors") or [])[accessor_index] if accessor_index < len(gltf.get("accessors") or []) else {}
                        document.vertices += int(accessor.get("count") or 0)
                merged_primitive["attributes"] = attributes
                indices = primitive.get("indices")
                if isinstance(indices, int):
                    merged_primitive["indices"] = accessor_map.get(indices, indices)
                    if indices < len(gltf.get("accessors") or []):
                        document.faces += int((gltf["accessors"][indices].get("count") or 0) // 3)
                material = primitive.get("material")
                if isinstance(material, int) and material in material_map:
                    merged_primitive["material"] = material_map[material]
                elif material is None:
                    merged_primitive.pop("material", None)
                primitives.append(merged_primitive)
            if not primitives:
                continue
            merged_mesh = {k: v for k, v in mesh.items() if k != "primitives"}
            merged_mesh["primitives"] = primitives
            mesh_map[old_index] = len(document.meshes)
            document.meshes.append(merged_mesh)

        part_scene_nodes = _merge_nodes(gltf, document, mesh_map, part_index, part_names)
        scene_node_groups.append(part_scene_nodes)

    # Rebuild the node tree with merged global indices.
    root_nodes: list[int] = []
    for group in scene_node_groups:
        root_nodes.extend(group)

    merged_gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "modules/glb_merge.py"},
        "scene": 0,
        "scenes": [{"nodes": root_nodes}],
        "nodes": document.nodes,
        "meshes": document.meshes,
        "accessors": document.accessors,
        "bufferViews": document.buffer_views,
        "buffers": [{"byteLength": document.bin_size}],
    }
    if document.materials:
        merged_gltf["materials"] = document.materials
    if document.images:
        merged_gltf["images"] = document.images
    if document.textures:
        merged_gltf["textures"] = document.textures
    if document.samplers:
        merged_gltf["samplers"] = document.samplers

    stats.vertices = document.vertices
    stats.faces = document.faces
    stats.materials = len(document.materials)
    stats.images = len(document.images)

    total_bytes = _write_glb(output_path, merged_gltf, parsed, document, max_bytes)
    stats.bytes = total_bytes
    return stats


def _merge_nodes(
    gltf: dict[str, Any],
    document: _MergedDocument,
    mesh_map: dict[int, int],
    part_index: int,
    part_names: Optional[Sequence[str]],
) -> list[int]:
    nodes = gltf.get("nodes") or []
    node_base = len(document.nodes)
    name = part_names[part_index] if part_names and part_index < len(part_names) else f"part_{part_index + 1:03d}"
    if not nodes:
        if not mesh_map:
            return []
        document.nodes.append({"mesh": next(iter(mesh_map.values())), "name": name})
        return [len(document.nodes) - 1]

    for node in nodes:
        merged_node: dict[str, Any] = {}
        for key in ("name", "translation", "rotation", "scale", "matrix", "extras", "extensions"):
            if key in node:
                merged_node[key] = node[key]
        mesh_index = node.get("mesh")
        if isinstance(mesh_index, int) and mesh_index in mesh_map:
            merged_node["mesh"] = mesh_map[mesh_index]
        children = node.get("children")
        if isinstance(children, list):
            merged_node["children"] = [node_base + child for child in children]
        document.nodes.append(merged_node)

    if len(nodes) == 1:
        document.nodes[node_base]["name"] = name

    scene_index = gltf.get("scene") or 0
    scenes = gltf.get("scenes") or []
    if 0 <= scene_index < len(scenes) and scenes[scene_index].get("nodes"):
        return [node_base + node for node in scenes[scene_index]["nodes"]]
    return [node_base + index for index in range(len(nodes))]


def _write_glb(
    output_path: Path,
    gltf: dict[str, Any],
    parts: Sequence[GlbPart],
    document: _MergedDocument,
    max_bytes: int,
) -> int:
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_pad = (-len(json_bytes)) & (ALIGNMENT - 1)
    json_chunk = json_bytes + b" " * json_pad
    bin_pad = (-document.bin_size) & (ALIGNMENT - 1)
    total_bytes = 12 + 8 + len(json_chunk) + 8 + document.bin_size + bin_pad
    if total_bytes > max_bytes:
        raise GlbMergeError(
            f"merged GLB would be {total_bytes / 1024 ** 3:.2f} GiB, above the {max_bytes / 1024 ** 3:.2f} GiB GLB container limit"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handles = [part.path.open("rb") for part in parts]
    try:
        with output_path.open("wb") as out:
            out.write(struct.pack("<III", GLB_MAGIC, 2, total_bytes))
            out.write(struct.pack("<II", len(json_chunk), CHUNK_TYPE_JSON))
            out.write(json_chunk)
            out.write(struct.pack("<II", document.bin_size + bin_pad, CHUNK_TYPE_BIN))
            written = 0
            for op in document.copy_ops:
                gap = op.target_offset - written
                if gap > 0:
                    out.write(b"\x00" * gap)
                    written += gap
                source = handles[op.part_index]
                source.seek(parts[op.part_index].bin_offset + op.source_offset)
                remaining = op.length
                while remaining > 0:
                    block = source.read(min(COPY_BLOCK, remaining))
                    if not block:
                        raise GlbMergeError(f"unexpected end of file: {parts[op.part_index].path}")
                    out.write(block)
                    remaining -= len(block)
                written += op.length
            if written < document.bin_size:
                out.write(b"\x00" * (document.bin_size - written))
            if bin_pad:
                out.write(b"\x00" * bin_pad)
    finally:
        for handle in handles:
            handle.close()
    return total_bytes


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Merge multiple .glb files into one.")
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    stats = merge_glbs(args.inputs, args.output)
    print(
        f"merged parts={stats.parts} vertices={stats.vertices} faces={stats.faces} "
        f"materials={stats.materials} images={stats.images} bytes={stats.bytes}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
