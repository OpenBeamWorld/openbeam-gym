# Copyright 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Every published robot carries visual meshes: a COLLADA file the game's
importer accepts, flat-colour materials, and JBeam flexbodies binding each
link's mesh to that link's node group. The test arm's physics is unchanged."""

import hashlib
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from openbeam_gym.robots import build_camera_boom, build_humanoid, build_test_arm

ASSETS = Path(__file__).resolve().parents[1] / "assets"
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
MODULES = [build_test_arm, build_camera_boom, build_humanoid]


def package(name):
    with zipfile.ZipFile(ASSETS / f"{name}.zip") as z:
        return {n: z.read(n) for n in z.namelist()}


def node_groups(part):
    """group -> [(node, position)], following the nodes section's group modifiers."""
    groups, current = {}, ""
    for row in part["nodes"][1:]:
        if isinstance(row, dict):
            current = row.get("group", current)
            continue
        if current:
            groups.setdefault(current, []).append((row[0], tuple(row[1:4])))
    return groups


def read_collada(text):
    """{mesh name: [(material name, [triangle as three points])]} from a .dae,
    checking the structure the importer requires on the way."""
    root = ET.fromstring(text)
    assert root.tag == "{%s}COLLADA" % NS["c"] and root.get("version").startswith("1.")
    assert root.find("c:asset/c:up_axis", NS).text == "Z_UP"
    assert float(root.find("c:asset/c:unit", NS).get("meter")) == 1.0
    scenes = root.findall("c:library_visual_scenes/c:visual_scene", NS)
    assert len(scenes) == 1
    assert root.find("c:scene/c:instance_visual_scene", NS).get("url") == "#" + scenes[0].get("id")
    materials = {m.get("id"): m for m in root.iterfind("c:library_materials/c:material", NS)}
    effects = {e.get("id") for e in root.iterfind("c:library_effects/c:effect", NS)}
    for m in materials.values():
        assert m.find("c:instance_effect", NS).get("url")[1:] in effects
    geometries = {g.get("id"): g for g in root.iterfind("c:library_geometries/c:geometry", NS)}
    out = {}
    for node in scenes[0].iterfind("c:node", NS):
        instances = node.findall("c:instance_geometry", NS)
        assert len(instances) == 1
        geo = geometries[instances[0].get("url")[1:]]
        bind = {im.get("symbol"): im.get("target")[1:] for im in instances[0].iterfind(".//c:instance_material", NS)}
        sources = {}
        for src in geo.iterfind("c:mesh/c:source", NS):
            arr = src.find("c:float_array", NS)
            values = [float(v) for v in arr.text.split()]
            acc = src.find("c:technique_common/c:accessor", NS)
            assert int(arr.get("count")) == len(values) == 3 * int(acc.get("count"))
            assert all(math.isfinite(v) for v in values)
            sources[src.get("id")] = [tuple(values[i:i + 3]) for i in range(0, len(values), 3)]
        vertices = geo.find("c:mesh/c:vertices", NS)
        positions = sources[vertices.find("c:input", NS).get("source")[1:]]
        prims = []
        assert not geo.findall("c:mesh/c:polygons", NS) and not geo.findall("c:mesh/c:tristrips", NS)
        for tri in geo.iterfind("c:mesh/c:triangles", NS):
            inputs = {i.get("semantic"): i for i in tri.iterfind("c:input", NS)}
            assert set(inputs) == {"VERTEX", "NORMAL"}
            assert inputs["VERTEX"].get("source") == "#" + vertices.get("id")
            stride = 1 + max(int(i.get("offset")) for i in inputs.values())
            normals = sources[inputs["NORMAL"].get("source")[1:]]
            idx = [int(v) for v in tri.find("c:p", NS).text.split()]
            count = int(tri.get("count"))
            assert len(idx) == 3 * stride * count
            vo, no = int(inputs["VERTEX"].get("offset")), int(inputs["NORMAL"].get("offset"))
            pts = []
            for t in range(count):
                corners = [idx[(3 * t + k) * stride + vo] for k in range(3)]
                n = normals[idx[3 * t * stride + no]]
                assert abs(math.hypot(*n) - 1) < 1e-4
                p = [positions[i] for i in corners]
                # Wound counter-clockwise about the stored (outward) normal.
                e1 = [p[1][i] - p[0][i] for i in range(3)]
                e2 = [p[2][i] - p[0][i] for i in range(3)]
                cr = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
                assert sum(a * b for a, b in zip(cr, n)) > 0
                pts.append(p)
            material = materials[bind[tri.get("material")]]
            prims.append((material.get("name"), pts))
        out[node.get("name")] = prims
    return out


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.NAME)
def test_the_package_is_deterministic_and_published(module, tmp_path):
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    digest = module.write_zip(a, module.members())
    assert module.write_zip(b, module.members()) == digest
    assert a.read_bytes() == b.read_bytes() == (ASSETS / f"{module.NAME}.zip").read_bytes()


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.NAME)
def test_every_link_has_a_bound_visual(module):
    name = module.NAME
    files = package(name)
    prefix = f"vehicles/{name}/"
    part = json.loads(files[prefix + f"{name}.jbeam"])[name]
    meshes = read_collada(files[prefix + f"{name}.dae"])
    materials = json.loads(files[prefix + f"{name}.materials.json"])
    groups = node_groups(part)
    rows = part["flexbodies"]
    assert rows[0] == ["mesh", "[group]:"]
    bound = {}
    for mesh, mesh_groups in rows[1:]:
        assert mesh in meshes and len(mesh_groups) == 1
        bound[mesh_groups[0]] = mesh
    # One mesh per link: every node group carries exactly one flexbody.
    assert set(bound) == set(groups) and len(rows) - 1 == len(groups)
    total = 0
    for group, mesh in bound.items():
        nodes = [p for _, p in groups[group]]
        # The importer binds a vertex to three nodes of the groups: at least
        # three, not all on one line.
        assert len(nodes) >= 3
        o = nodes[0]
        assert any(
            math.hypot(*[(a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1]),
                         (a[2] - o[2]) * (b[0] - o[0]) - (a[0] - o[0]) * (b[2] - o[2]),
                         (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])]) > 1e-6
            for a in nodes for b in nodes)
        prims = meshes[mesh]
        assert prims
        for material, tris in prims:
            record = materials[material]
            assert record["class"] == "Material" and record["mapTo"] == material
            colour = record["Stages"][0]["baseColorFactor"]
            assert len(colour) == 4 and all(0.0 <= c <= 1.0 for c in colour)
            for tri in tris:
                total += 1
                # Mesh and nodes share the JBeam frame: every vertex lies near its link's nodes.
                for v in tri:
                    assert min(math.dist(v, p) for p in nodes) < 0.35, (mesh, v)
    assert 100 < total < 3000


def test_the_test_arm_physics_is_unchanged():
    """Only visuals were added to the test arm: its JBeam without the
    flexbodies section, and every other member, are byte for byte the
    published physics-only package (f79f4df)."""
    files = package(build_test_arm.NAME)
    prefix = "vehicles/openbeam_test_arm/"
    before = {
        "openbeam_test_arm.jbeam": "589ae743fb4f8c0e3acf0859a48074fa2ad87f44ca6137c3e88a52a372e54bc2",
        "openbeam_robot.json": "bf8698019aa3031db2d704ca9aec76db430b90c9e5b9eb084e06ef580a16bda9",
        "lua/openbeam_test_arm.lua": "c5d73d60fceb04090ab411ea26796e287c5a8aa187b33cfbefaf7b71515372e3",
        "input_actions_openbeam_test_arm.json": "bdc98f32c21c22fa9dfaae8912fb9f8702e6e49a86b12155cd22825b76a9fd77",
        "inputmaps/keyboard.json": "ce8f6ca00b519b7d0865783c511723223152be0f44fa7a648d72246ff8644209",
        "info.json": "9aa80e98803bb4bc5e9cca15dc7b0498819023b51bde890107c1926b111156ec",
        "LICENSE.txt": "d61b4b36c9ffd39de4f4e4a9629a2b2220a10d12be4eb238759f789dab628e00",
    }
    jbeam = json.loads(files[prefix + "openbeam_test_arm.jbeam"])
    part = jbeam["openbeam_test_arm"]
    assert list(part)[-1] == "flexbodies"
    del part["flexbodies"]
    physics = (json.dumps(jbeam, indent=1) + "\n").encode()
    assert hashlib.sha256(physics).hexdigest() == before["openbeam_test_arm.jbeam"]
    for member, digest in before.items():
        if member != "openbeam_test_arm.jbeam":
            assert hashlib.sha256(files[prefix + member]).hexdigest() == digest, member
    added = set(files) - {prefix + m for m in before}
    assert added == {prefix + "openbeam_test_arm.dae", prefix + "openbeam_test_arm.materials.json"}
