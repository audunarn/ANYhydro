"""Explicit internal-lid profile for direct native radiation preparation.

The lid is a mathematical source surface, never a physical pressure surface.
Legacy family recipes and capacities remain unchanged. All native work is explicit;
restoring a profile and evaluating retained sources never calls a BEM solve.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np
from .wave_surface_fields import WaveFieldError

PROFILE_METHOD = "capytaine-3.0-direct-internal-lid-v1"
WATERLINE_PROFILE_METHOD = "anymesher-constrained-waterline-lid-v1"
PROFILE_KEY = "direct_lid_profile"
MAX_SOURCE_PANELS = 4000


class DirectLidProfileError(WaveFieldError):
    """Invalid, stale or unsupported direct native source profile."""


def _require(condition, message):
    if not condition:
        raise DirectLidProfileError(message)


def _freeze(value, dtype, name):
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise DirectLidProfileError(f"{name} is not a rectangular numeric array") from exc
    _require(not np.iscomplexobj(raw), f"{name} must be real")
    if np.dtype(dtype).kind in "iu":
        _require(raw.dtype.kind in "iu", f"{name} must contain integer indices")
    try:
        a = np.array(raw, dtype=dtype, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DirectLidProfileError(f"{name} is not a valid numeric array") from exc
    _require(bool(np.all(np.isfinite(a))), f"{name} must be finite")
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _number(value, name):
    _require(not isinstance(value, (bool, np.bool_)) and not np.iscomplexobj(value), f"{name} must be real")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DirectLidProfileError(f"{name} must be a finite scalar") from exc
    _require(math.isfinite(out), f"{name} must be finite")
    return out


@dataclass(frozen=True)
class DirectLidProfile:
    """Complete neutral mathematical-lid and full-source geometry manifest."""

    hull_hash: str
    lid_vertices: Any
    lid_faces: Any
    source_vertices: Any
    source_faces: Any
    hull_mask: Any
    lid_z: float
    faces_max_radius: float
    method: str = PROFILE_METHOD
    source_panel_limit: int = MAX_SOURCE_PANELS
    max_work_bytes: int = 512 * 1024 * 1024
    waterline_cell_size: Any = None

    def __post_init__(self):
        _require(isinstance(self.method, str) and self.method in (PROFILE_METHOD, WATERLINE_PROFILE_METHOD),
                 "unsupported direct lid profile method")
        if self.method == PROFILE_METHOD:
            _require(self.waterline_cell_size is None, 'legacy lid profiles have no waterline cell size')
        else:
            size = _number(self.waterline_cell_size, 'waterline_cell_size')
            _require(size > 0., 'waterline cell size must be positive')
            object.__setattr__(self, 'waterline_cell_size', size)
        _require(type(self.max_work_bytes) is int and 0 < self.max_work_bytes <= 512 * 1024 * 1024,
                 "generation work ceiling may only tighten 512 MiB")
        _require(isinstance(self.hull_hash, str) and len(self.hull_hash) == 64, "invalid hull hash")
        _require(type(self.source_panel_limit) is int and 1 <= self.source_panel_limit <= MAX_SOURCE_PANELS,
                 "full source panel limit may only tighten 4000")
        for name, dtype in (("lid_vertices", np.float64), ("lid_faces", np.int64),
                            ("source_vertices", np.float64), ("source_faces", np.int64)):
            raw = getattr(self, name)
            width = 4 if 'faces' in name else 3
            ceiling = MAX_SOURCE_PANELS if 'faces' in name else 8 * MAX_SOURCE_PANELS
            _require(isinstance(raw, (list, tuple, np.ndarray)) and 0 < len(raw) <= ceiling
                     and all(hasattr(row, '__len__') and len(row) == width for row in raw),
                     f'{name} exceeds bounded dimensions')
            object.__setattr__(self, name, _freeze(getattr(self, name), dtype, name))
        _require(isinstance(self.hull_mask, (list, tuple, np.ndarray))
                 and len(self.hull_mask) <= MAX_SOURCE_PANELS
                 and all(isinstance(v, (bool, np.bool_)) for v in self.hull_mask),
                 'hull_mask must be bounded scalar booleans')
        mask = np.asarray(self.hull_mask)
        _require(mask.dtype.kind == "b" and mask.ndim == 1 and 0 < len(mask) <= MAX_SOURCE_PANELS,
                 "hull_mask must be a bounded one-dimensional boolean array")
        object.__setattr__(self, "hull_mask", np.frombuffer(mask.tobytes(), dtype=np.bool_))
        for v, f, name in ((self.lid_vertices, self.lid_faces, "lid"),
                           (self.source_vertices, self.source_faces, "source")):
            _require(v.ndim == 2 and v.shape[1] == 3 and len(v) > 0, f"{name} vertices must be nonempty N by 3")
            _require(f.ndim == 2 and f.shape[1] == 4 and len(f) > 0, f"{name} faces must be nonempty N by 4")
            _require(int(f.min()) >= 0 and int(f.max()) < len(v), f"{name} face index out of range")
        _require(len(self.source_vertices) <= 8 * MAX_SOURCE_PANELS and len(self.lid_vertices) <= 8 * MAX_SOURCE_PANELS,
                 'direct source vertices exceed bounded count')
        _require(len(self.source_faces) <= self.source_panel_limit, "hull plus lid exceeds full source panel limit")
        _require(self.hull_mask.shape == (len(self.source_faces),) and bool(np.any(self.hull_mask))
                 and bool(np.any(~self.hull_mask)), "full source mask must identify hull and lid")
        _require(int(np.count_nonzero(~self.hull_mask)) == len(self.lid_faces), "lid/source face counts disagree")
        _require(np.array_equal(self.source_vertices[self.source_faces[~self.hull_mask]],
                                self.lid_vertices[self.lid_faces]), "lid/full source geometry or order differs")
        z, radius = _number(self.lid_z, "lid_z"), _number(self.faces_max_radius, "faces_max_radius")
        _require((z < 0. if self.method == PROFILE_METHOD else z == 0.) and radius > 0.,
                 "legacy lid z must be negative; waterline lid z must be zero; face radius must be positive")
        _require(bool(np.all(self.lid_vertices[:, 2] == z)), "lid must be exactly planar at its declared z")
        _require(z > float(np.min(self.source_vertices[self.source_faces[self.hull_mask], 2])),
                 "lid must lie strictly above the hull keel")
        for vertices, faces in ((self.source_vertices, self.source_faces), (self.lid_vertices, self.lid_faces)):
            polygons = [tuple(sorted(map(tuple, polygon))) for polygon in vertices[faces]]
            _require(len(set(polygons)) == len(polygons), "duplicate numerical source panels are forbidden")
        p = self.lid_vertices[self.lid_faces]
        normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        normal += np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])
        _require(bool(np.all(normal[:, 2] < 0.0)), "lid faces must already point downward; no silent flip")
        object.__setattr__(self, "lid_z", z)
        object.__setattr__(self, "faces_max_radius", radius)

    def to_dict(self):
        out = dict(method=self.method, hull_hash=self.hull_hash, lid_z=self.lid_z,
                    faces_max_radius=self.faces_max_radius, source_panel_limit=self.source_panel_limit,
                    lid_vertices=self.lid_vertices.tolist(), lid_faces=self.lid_faces.tolist(),
                    source_vertices=self.source_vertices.tolist(), source_faces=self.source_faces.tolist(),
                    hull_mask=self.hull_mask.tolist())
        out['max_work_bytes'] = self.max_work_bytes
        if self.waterline_cell_size is not None:
            out['waterline_cell_size'] = self.waterline_cell_size
        return out

    def content_hash(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()


def profile_from_dict(record):
    """Validate a complete neutral settings record, without importing native code."""
    from dataclasses import fields
    _require(isinstance(record, Mapping), 'direct lid profile must be a mapping')
    allowed = {f.name for f in fields(DirectLidProfile)}
    _require(allowed - {'waterline_cell_size'} <= set(record) <= allowed,
             'incomplete or unknown direct lid profile fields')
    # The request loader applies preallocation bounds and canonical hash validation.
    request = {PROFILE_KEY: dict(record, profile_hash=hashlib.sha256(json.dumps(dict(record), sort_keys=True,
               separators=(',', ':'), allow_nan=False).encode()).hexdigest()), 'geometry': record['hull_hash']}
    return profile_from_request(request)


def profile_from_request(request: Mapping[str, Any]):
    record = request.get(PROFILE_KEY)
    if record is None:
        return None
    _require(isinstance(record, Mapping), "source profile must be a neutral mapping")
    data = dict(record)
    expected = data.pop("profile_hash", None)
    allowed = {f.name for f in __import__('dataclasses').fields(DirectLidProfile)}
    _require(allowed - {'waterline_cell_size'} <= set(data) <= allowed,
             "incomplete or unknown direct lid profile fields")
    # Check dimensions before constructing private array copies.
    for name in ("lid_faces", "source_faces"):
        _require(isinstance(data.get(name), (list, tuple)) and 0 < len(data[name]) <= MAX_SOURCE_PANELS,
                 f"{name} exceeds source limits")
        _require(all(isinstance(row, (list, tuple)) and len(row) == 4 for row in data[name]),
                 f"{name} rows must contain exactly four indices")
    for name in ("lid_vertices", "source_vertices"):
        _require(isinstance(data.get(name), (list, tuple)) and 0 < len(data[name]) <= 8 * MAX_SOURCE_PANELS,
                 f"{name} exceeds source limits")
        _require(all(isinstance(row, (list, tuple)) and len(row) == 3 for row in data[name]),
                 f"{name} rows must contain exactly three coordinates")
    _require(isinstance(data.get('hull_mask'), (list, tuple))
             and len(data['hull_mask']) == len(data['source_faces'])
             and all(type(value) is bool for value in data['hull_mask']),
             'hull_mask must contain one boolean per source panel')
    try:
        profile = DirectLidProfile(**data)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DirectLidProfileError(f"invalid source profile: {exc}") from exc
    _require(expected == profile.content_hash(), "source profile hash differs from retained geometry/settings")
    _require(profile.hull_hash == request.get("geometry"), "source profile belongs to a different hull")
    return profile


def bind_profile(request, profile):
    _require(isinstance(profile, DirectLidProfile), "explicit DirectLidProfile required")
    _require(profile.hull_hash == request.get("geometry"), "source profile belongs to a different hull")
    out = dict(request)
    out[PROFILE_KEY] = dict(profile.to_dict(), profile_hash=profile.content_hash())
    return out


_CANONICAL_PROFILES = {}  # bounded successful immutable profile identities; no native objects


def verify_canonical_profile(profile, vertices, faces, surface_ids, *, water_depth):
    """Lazy native geometry-only proof of the declared canonical internal lid; never BEM/GF.

    Neutral loading checks shape, depth and duplicate invariants. Before a profile
    becomes an executable request, regenerate it with the native supported hull
    clipping/lid API and compare its complete geometry/order/mask exactly.
    """
    from .floating_radiation_fields import family_request
    hull = family_request(vertices, faces, surface_ids, heading=0., water_depth=water_depth,
                          rho=1025., gravity=9.81, Q0=np.eye(3), cg_world=np.zeros(3))
    _require(hull['geometry'] == profile.hull_hash, 'canonical profile belongs to a different physical hull')
    _require(profile.lid_z > -float(water_depth), 'lid intersects seabed')
    identity = (profile.content_hash(), float(water_depth))
    if identity in _CANONICAL_PROFILES:
        return profile
    canonical = create_source_profile(vertices, faces, surface_ids, lid_z=profile.lid_z,
        faces_max_radius=profile.faces_max_radius if profile.method == PROFILE_METHOD else None,
        method=profile.method, waterline_cell_size=profile.waterline_cell_size, water_depth=water_depth,
        source_panel_limit=profile.source_panel_limit, max_work_bytes=profile.max_work_bytes)
    _require(canonical.content_hash() == profile.content_hash(),
             'declared lid is not the canonical internal surface of this physical hull')
    if len(_CANONICAL_PROFILES) >= 128:
        _CANONICAL_PROFILES.pop(next(iter(_CANONICAL_PROFILES)))
    _CANONICAL_PROFILES[identity] = True
    return profile


def create_source_profile(vertices, faces, surface_ids, *, lid_z, faces_max_radius=None,
                          water_depth, source_panel_limit=MAX_SOURCE_PANELS,
                          max_work_bytes=512 * 1024 * 1024, method=PROFILE_METHOD, waterline_cell_size=None):
    """Explicit native lid generation; no physical hull clipping, repair or fallback."""
    import capytaine as cpt
    from .floating_radiation_fields import family_request
    from ._pressure_body import prepare_body
    _require(type(source_panel_limit) is int and 1 <= source_panel_limit <= MAX_SOURCE_PANELS,
             "full source panel limit may only tighten 4000")
    _require(type(max_work_bytes) is int and 0 < max_work_bytes <= 512 * 1024 * 1024,
             "working-memory ceiling may only tighten 512 MiB")
    _require(len(faces) <= source_panel_limit and len(vertices) <= 4 * source_panel_limit,
             "physical source manifest exceeds bounded panel/vertex counts")
    _require(str(cpt.__version__).split(".")[:2] == ["3", "0"], "profile requires Capytaine 3.0")
    _require(isinstance(method, str) and method in (PROFILE_METHOD, WATERLINE_PROFILE_METHOD),
             'unsupported direct lid profile method')
    v, f, labels, mesh, _ = prepare_body(cpt, vertices, faces, surface_ids, period=math.tau,
                                       heading=0., water_depth=water_depth, rho=1025., max_panels=source_panel_limit, panel_ceiling=MAX_SOURCE_PANELS)
    _require(not bool(np.any(np.all(v[f, 2] == 0., axis=1))),
             "physical hull includes a free-surface cap; provide the open wetted hull")
    z = _number(lid_z, "lid_z")
    if method == WATERLINE_PROFILE_METHOD:
        _require(z == 0. and faces_max_radius is None,
                 'waterline profiles require lid_z=0 and waterline_cell_size instead of faces_max_radius')
        from .waterline_lid import generate_waterline_lid
        lv, lf = generate_waterline_lid(v, f, cell_size=waterline_cell_size,
            max_panels=source_panel_limit-len(f), max_work_bytes=max_work_bytes)
        lid = cpt.Mesh(vertices=lv, faces=lf, auto_check=False)
        body = cpt.FloatingBody(mesh=mesh, lid_mesh=lid)
        _require(np.array_equal(body.lid_mesh.vertices, lv) and np.array_equal(body.lid_mesh.faces, lf),
                 'native body changed the fitted waterline lid')
        req = family_request(v, f, labels, heading=0., water_depth=water_depth, rho=1025., gravity=9.81,
                             Q0=np.eye(3), cg_world=np.zeros(3))
        return DirectLidProfile(req['geometry'], lv, lf, body.mesh_including_lid.vertices,
            body.mesh_including_lid.faces, np.asarray(body.hull_mask, dtype=bool), z,
            float(np.max(lid.faces_radiuses)), method=method, source_panel_limit=source_panel_limit,
            max_work_bytes=max_work_bytes, waterline_cell_size=waterline_cell_size)
    _require(waterline_cell_size is None, 'legacy lid profiles have no waterline cell size')
    _require(math.isfinite(z) and max(float(v[:, 2].min()), -float(water_depth)) < z < 0.,
             "explicit lid z must lie inside the wetted depth")
    radius = float(np.mean(mesh.faces_radiuses)) if faces_max_radius is None else _number(faces_max_radius, "faces_max_radius")
    _require(math.isfinite(radius) and radius > 0., "lid face radius must be positive")
    # Capytaine 3.0 generate_lid uses a rectangle bounded by 1.1 times the
    # clipped-hull spans and projects every candidate vertex onto every face.
    # Full-hull spans bound clipped spans, so admission precedes that broadcast.
    span = np.ptp(v[:, :2], axis=0)
    size = np.maximum(radius / 2., 1.1 * span)
    nx, ny = (max(5, math.ceil(float(s) / (math.sqrt(2.) * radius))) for s in size)
    candidate_points = (nx + 1) * (ny + 1)
    generation_work = candidate_points * len(f) * 4 * 2 * 8 * 6
    _require(generation_work <= max_work_bytes,
             f"native lid generation needs up to {generation_work} B, exceeding the working-memory ceiling")
    lid = mesh.generate_lid(z=z, faces_max_radius=radius)
    _require(lid.nb_faces > 0, "native lid generation produced no panels; revise explicit profile inputs")
    _require(mesh.nb_faces + lid.nb_faces <= min(source_panel_limit, MAX_SOURCE_PANELS),
             "hull plus lid exceeds full source panel limit")
    body = cpt.FloatingBody(mesh=mesh, lid_mesh=lid)
    # The native constructor must not heal/reorient our retained lid.
    _require(np.array_equal(body.lid_mesh.vertices, lid.vertices) and np.array_equal(body.lid_mesh.faces, lid.faces),
             "native body changed the declared lid geometry")
    req = family_request(v, f, labels, heading=0., water_depth=water_depth, rho=1025., gravity=9.81,
                         Q0=np.eye(3), cg_world=np.zeros(3))
    return DirectLidProfile(req["geometry"], lid.vertices, lid.faces,
                                  body.mesh_including_lid.vertices, body.mesh_including_lid.faces,
                                  np.asarray(body.hull_mask, dtype=bool), z, radius,
                                  source_panel_limit=source_panel_limit,
                                  max_work_bytes=max_work_bytes)


def body_for_profile(cpt, vertices, faces, labels, request, reference_omega):
    """Rebuild exact declared source geometry; no solve and no Green-function fit."""
    from ._pressure_body import prepare_body
    profile = profile_from_request(request)
    _require(profile is not None, "direct lid profile is missing")
    v, f, labels, mesh, _ = prepare_body(cpt, vertices, faces, labels, period=math.tau / float(reference_omega),
                                       heading=request["heading"], water_depth=request["water_depth"],
                                       rho=request["rho"], max_panels=profile.source_panel_limit, panel_ceiling=MAX_SOURCE_PANELS,
                                       source_quadrature=request.get('source_quadrature'))
    from .floating_radiation_fields import family_request, DOF_NAMES
    actual = family_request(v, f, labels, heading=request["heading"], water_depth=request["water_depth"],
                            rho=request["rho"], gravity=request["gravity"], Q0=request["Q0"], cg_world=request["cg_world"])
    _require(actual["geometry"] == profile.hull_hash, "profile hull geometry changed")
    verify_canonical_profile(profile, v, f, labels, water_depth=request['water_depth'])
    _require(profile.lid_z > -float(request["water_depth"]), "lid intersects seabed")
    from .source_quadrature import native_options, configure
    rule = request.get('source_quadrature')
    lid = cpt.Mesh(vertices=profile.lid_vertices.copy(), faces=profile.lid_faces.copy(), auto_check=False,
                   **native_options(rule))
    configure(lid, rule)
    body = cpt.FloatingBody(mesh=mesh, lid_mesh=lid, center_of_mass=np.asarray(request["cg_world"]))
    configure(body.lid_mesh, rule)
    configure(body.mesh_including_lid, rule)
    q, cg = np.asarray(request["Q0"]), np.asarray(request["cg_world"])
    for i, name in enumerate(DOF_NAMES[:3]):
        body.add_translation_dof(direction=q[:, i].copy(), name=name)
    for i, name in enumerate(DOF_NAMES[3:]):
        body.add_rotation_dof(rotation_center=cg.copy(), direction=q[:, i].copy(), name=name)
    verify_profile_body(body, request)
    return v, f, labels, mesh, body


def verify_profile_body(body, request):
    profile = profile_from_request(request)
    _require(profile is not None, "source profile missing")
    full = body.mesh_including_lid
    _require(np.array_equal(full.vertices, profile.source_vertices)
             and np.array_equal(full.faces, profile.source_faces)
             and np.array_equal(body.hull_mask, profile.hull_mask),
             "native full source mesh/order/mask differs from retained profile")
    _require(full.nb_faces <= profile.source_panel_limit, "full source unknown count exceeds profile limit")
    _require(getattr(full, 'quadrature_method', None) == request.get('source_quadrature'),
             'native full source quadrature differs from its request')
    from .source_quadrature import verify as verify_quadrature
    try:
        verify_quadrature(full, request.get('source_quadrature'))
    except ValueError as error:
        raise DirectLidProfileError(str(error)) from error
    return profile


def solve_profile_snapshot(vertices, faces, surface_ids, omega, *, profile, **kwargs):
    """Explicit Prepare operation, sharing canonical native solve/retention code."""
    from .floating_radiation_fields import solve_family_snapshot, DIRECT_LID_RECIPE
    return solve_family_snapshot(vertices, faces, surface_ids, omega, direct_lid_profile=profile,
                                 recipe=DIRECT_LID_RECIPE, **kwargs)

