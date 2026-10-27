# Copyright 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Visual meshes for the robot packages: flat-coloured boxes, prisms and
capsule-like prisms per link, written as one COLLADA file, a materials file
and the JBeam ``flexbodies`` rows that bind each link's mesh to its node group.

How the game draws an imported JBeam vehicle (BeamNG layout):

* ``flexbodies`` rows ``[mesh, [groups]]`` name a COLLADA scene node (by its
  ``name``) and the node groups (the ``{"group": ...}`` modifier rows of the
  ``nodes`` section) whose nodes carry it. Every vertex is bound to nearby
  nodes of those groups and follows them as the soft body moves. A group
  needs at least three non-collinear nodes.
* The ``.dae`` must declare ``Z_UP`` and ``<unit meter="1"/>`` and is authored
  in the JBeam vehicle frame (x left, y back, z up, metres), the frame the
  node positions are written in. Geometry is ``<triangles>`` with ``VERTEX``
  and ``NORMAL`` inputs; each primitive's ``material`` symbol is bound through
  ``<bind_material>``/``<instance_material>`` to a ``<material>`` whose name is
  looked up in the package's ``*.materials.json``.
* A material record ``{"class": "Material", "Stages": [{"baseColorFactor":
  [r, g, b, a]}, ...]}`` gives a flat colour (components in [0, 1]); the
  COLLADA effect's diffuse colour is the same colour, the fallback.

The builders describe their links with the primitives here in their own base
frame (x forward, y left, z up); ``Mesh`` converts to the JBeam frame. Output
is deterministic: fixed float formatting, insertion-ordered meshes, no dates.

Stdlib only.
"""

from __future__ import annotations

import json
import math

from .sizing import add, cross, dot, mul, norm, sub, unit

# name -> (rgba, roughness, metallic)
PALETTE = {
    "link": ((0.14, 0.14, 0.15, 1.0), 0.55, 0.3),        # dark grey structure
    "housing": ((0.95, 0.40, 0.04, 1.0), 0.45, 0.0),     # orange joint housings
    "stand": ((0.32, 0.32, 0.34, 1.0), 0.7, 0.2),        # grey base, mast, pedestal
    "shell": ((0.78, 0.78, 0.80, 1.0), 0.5, 0.0),        # light body shell
    "camera": ((0.02, 0.02, 0.02, 1.0), 0.4, 0.1),       # camera body
    "lens": ((0.04, 0.07, 0.16, 1.0), 0.1, 0.0),         # lens glass, visor
}


def _jbeam(p):
    """Base frame (x forward, y left, z up) -> JBeam (x left, y back, z up): a
    rotation, so windings and normals stay outward."""
    return (p[1], -p[0], p[2])


def frame_along(axis, hint=None):
    """Two unit vectors perpendicular to `axis` (and to each other)."""
    a = unit(axis)
    h = hint if hint is not None else ((0.0, 0.0, 1.0) if abs(a[2]) < 0.9 else (1.0, 0.0, 0.0))
    e1 = sub(h, mul(a, dot(h, a)))
    if norm(e1) < 1e-9:
        h = (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0)
        e1 = sub(h, mul(a, dot(h, a)))
    e1 = unit(e1)
    return e1, cross(a, e1)


class Mesh:
    """Triangles per material of one link (base frame in, JBeam frame out)."""

    def __init__(self, name: str, group: str):
        self.name = name
        self.group = group
        self.tris: dict[str, list[tuple]] = {}

    def triangle_count(self) -> int:
        return sum(len(t) for t in self.tris.values())

    def _convex(self, material, faces, centre):
        """Adds convex faces (polygons, as point lists) wound outward from `centre`."""
        assert material in PALETTE, material
        out = self.tris.setdefault(material, [])
        for poly in faces:
            for i in range(1, len(poly) - 1):
                a, b, c = poly[0], poly[i], poly[i + 1]
                n = cross(sub(b, a), sub(c, a))
                if norm(n) < 1e-12:
                    continue
                if dot(n, sub(a, centre)) < 0:
                    b, c = c, b
                out.append((_jbeam(a), _jbeam(b), _jbeam(c)))

    def box(self, centre, axes, half, material):
        """A box: `axes` three orthonormal directions, `half` its half sizes."""
        ex, ey, ez = (mul(unit(a), h) for a, h in zip(axes, half))

        def corner(i, j, k):
            return add(centre, mul(ex, i), mul(ey, j), mul(ez, k))

        faces = [
            [corner(s, -1, -1), corner(s, 1, -1), corner(s, 1, 1), corner(s, -1, 1)] for s in (-1, 1)
        ] + [
            [corner(-1, s, -1), corner(1, s, -1), corner(1, s, 1), corner(-1, s, 1)] for s in (-1, 1)
        ] + [
            [corner(-1, -1, s), corner(1, -1, s), corner(1, 1, s), corner(-1, 1, s)] for s in (-1, 1)
        ]
        self._convex(material, faces, centre)

    def bar(self, p, q, width, depth, material, hint=None):
        """A box from p to q with the given cross-section (width along the
        first perpendicular of ``frame_along``, depth along the second)."""
        axis = sub(q, p)
        e1, e2 = frame_along(axis, hint)
        self.box(mul(add(p, q), 0.5), (axis, e1, e2), (norm(axis) / 2, width / 2, depth / 2), material)

    def prism(self, p, q, radius, material, sides=12, cap=0.0):
        """A `sides`-gon prism of circumradius `radius` from p to q; with `cap`
        > 0 each end gets a pyramid of that height (a capsule-like prism)."""
        axis = sub(q, p)
        a = unit(axis)
        e1, e2 = frame_along(axis)

        def ring(c):
            return [add(c, mul(e1, radius * math.cos(2 * math.pi * k / sides)),
                        mul(e2, radius * math.sin(2 * math.pi * k / sides))) for k in range(sides)]

        r0, r1 = ring(p), ring(q)
        faces = [[r0[k], r0[(k + 1) % sides], r1[(k + 1) % sides], r1[k]] for k in range(sides)]
        centre = mul(add(p, q), 0.5)
        if cap > 0:
            tip0, tip1 = add(p, mul(a, -cap)), add(q, mul(a, cap))
            faces += [[r0[k], r0[(k + 1) % sides], tip0] for k in range(sides)]
            faces += [[r1[k], r1[(k + 1) % sides], tip1] for k in range(sides)]
        else:
            faces += [r0, r1]
        self._convex(material, faces, centre)

    def housing(self, centre, axis, length, radius, material="housing", sides=12):
        """A joint housing: a prism of `length` centred on `centre` along `axis`."""
        a = unit(axis)
        self.prism(add(centre, mul(a, -length / 2)), add(centre, mul(a, length / 2)), radius, material, sides)


# -- package members ------------------------------------------------------------------------

def _f(x: float) -> str:
    s = f"{x:.5f}"
    return "0.00000" if s == "-0.00000" else s


def _material_name(prefix: str, material: str) -> str:
    return f"{prefix}_{material}"


def collada(meshes: list[Mesh], prefix: str) -> str:
    """One COLLADA 1.4.1 document: a geometry and a scene node per mesh, named
    as the mesh (the flexbody row's ``mesh``), flat normals."""
    used = [m for m in PALETTE if any(m in mesh.tris for mesh in meshes)]
    out = ['<?xml version="1.0" encoding="utf-8"?>',
           '<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">',
           '<asset><contributor><authoring_tool>openbeam-gym</authoring_tool></contributor>'
           '<unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>',
           "<library_effects>"]
    for m in used:
        r, g, b, a = PALETTE[m][0]
        out.append(f'<effect id="{prefix}_{m}-effect"><profile_COMMON><technique sid="common"><lambert>'
                   f'<diffuse><color sid="diffuse">{_f(r)} {_f(g)} {_f(b)} {_f(a)}</color></diffuse>'
                   f"</lambert></technique></profile_COMMON></effect>")
    out.append("</library_effects>")
    out.append("<library_materials>")
    for m in used:
        name = _material_name(prefix, m)
        out.append(f'<material id="{name}-material" name="{name}"><instance_effect url="#{prefix}_{m}-effect"/>'
                   "</material>")
    out.append("</library_materials>")
    out.append("<library_geometries>")
    for mesh in meshes:
        positions: dict[tuple, int] = {}
        normals: dict[tuple, int] = {}
        prims = []
        for m in PALETTE:
            if m not in mesh.tris:
                continue
            idx = []
            for tri in mesh.tris[m]:
                n = unit(cross(sub(tri[1], tri[0]), sub(tri[2], tri[0])))
                nk = tuple(_f(c) for c in n)
                ni = normals.setdefault(nk, len(normals))
                for v in tri:
                    vk = tuple(_f(c) for c in v)
                    idx += [positions.setdefault(vk, len(positions)), ni]
            prims.append((m, len(mesh.tris[m]), idx))
        gid = f"{mesh.name}-mesh"
        pos = " ".join(" ".join(k) for k in positions)
        nor = " ".join(" ".join(k) for k in normals)
        out.append(f'<geometry id="{gid}" name="{mesh.name}"><mesh>')
        for src, data, count in (("positions", pos, len(positions)), ("normals", nor, len(normals))):
            out.append(f'<source id="{gid}-{src}"><float_array id="{gid}-{src}-array" count="{3 * count}">{data}'
                       f'</float_array><technique_common><accessor source="#{gid}-{src}-array" count="{count}" '
                       'stride="3"><param name="X" type="float"/><param name="Y" type="float"/>'
                       '<param name="Z" type="float"/></accessor></technique_common></source>')
        out.append(f'<vertices id="{gid}-vertices"><input semantic="POSITION" source="#{gid}-positions"/></vertices>')
        for m, count, idx in prims:
            out.append(f'<triangles material="{m}" count="{count}">'
                       f'<input semantic="VERTEX" source="#{gid}-vertices" offset="0"/>'
                       f'<input semantic="NORMAL" source="#{gid}-normals" offset="1"/>'
                       f'<p>{" ".join(map(str, idx))}</p></triangles>')
        out.append("</mesh></geometry>")
    out.append("</library_geometries>")
    out.append(f'<library_visual_scenes><visual_scene id="{prefix}-scene" name="{prefix}">')
    for mesh in meshes:
        binds = "".join(f'<instance_material symbol="{m}" target="#{_material_name(prefix, m)}-material"/>'
                        for m in PALETTE if m in mesh.tris)
        out.append(f'<node id="{mesh.name}" name="{mesh.name}" type="NODE">'
                   "<matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>"
                   f'<instance_geometry url="#{mesh.name}-mesh" name="{mesh.name}">'
                   f"<bind_material><technique_common>{binds}</technique_common></bind_material>"
                   "</instance_geometry></node>")
    out.append("</visual_scene></library_visual_scenes>")
    out.append(f'<scene><instance_visual_scene url="#{prefix}-scene"/></scene>')
    out.append("</COLLADA>")
    return "\n".join(out) + "\n"


def materials_json(meshes: list[Mesh], prefix: str) -> str:
    records = {}
    for m in PALETTE:
        if not any(m in mesh.tris for mesh in meshes):
            continue
        rgba, rough, metal = PALETTE[m]
        name = _material_name(prefix, m)
        records[name] = {"name": name, "mapTo": name, "class": "Material", "version": 1.5,
                         "Stages": [{"baseColorFactor": list(rgba), "roughnessFactor": rough,
                                     "metallicFactor": metal}, {}, {}, {}]}
    return json.dumps(records, indent=1) + "\n"


def flexbodies(meshes: list[Mesh]) -> list:
    """The JBeam ``flexbodies`` section: one row per mesh, bound to its group."""
    return [["mesh", "[group]:"]] + [[mesh.name, [mesh.group]] for mesh in meshes]


def package_members(prefix_dir: str, name: str, meshes: list[Mesh]) -> dict[str, bytes]:
    """The mesh and materials members of a vehicle package."""
    return {
        f"{prefix_dir}/{name}.dae": collada(meshes, name).encode(),
        f"{prefix_dir}/{name}.materials.json": materials_json(meshes, name).encode(),
    }
