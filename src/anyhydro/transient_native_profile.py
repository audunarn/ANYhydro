"""Explicit internal-lid profile for transient native radiation preparation.

The lid is a mathematical source surface, never a physical pressure surface.
The accepted spectral profile remains the default. All native work is explicit;
restoring a profile and evaluating retained sources never calls a BEM solve.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from functools import lru_cache
from typing import Any, Mapping

import numpy as np
from .wave_surface_fields import WaveFieldError

PROFILE_METHOD = "capytaine-3.0-indirect-internal-lid-v1"
PROFILE_KEY = "transient_source_profile"
MAX_SOURCE_PANELS = 600
PRONY_RECIPE = "capytaine-configurable-prony-v1"
TIGHT_GREEN_NAME = "TransientDelhommeau"


class TransientSourceProfileError(WaveFieldError):
    """Invalid, stale or unsupported transient native source profile."""


def _require(condition, message):
    if not condition:
        raise TransientSourceProfileError(message)


def _freeze(value, dtype, name):
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TransientSourceProfileError(f"{name} is not a rectangular numeric array") from exc
    _require(not np.iscomplexobj(raw), f"{name} must be real")
    if np.dtype(dtype).kind in "iu":
        _require(raw.dtype.kind in "iu", f"{name} must contain integer indices")
    try:
        a = np.array(raw, dtype=dtype, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TransientSourceProfileError(f"{name} is not a valid numeric array") from exc
    _require(bool(np.all(np.isfinite(a))), f"{name} must be finite")
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _number(value, name):
    _require(not isinstance(value, (bool, np.bool_)) and not np.iscomplexobj(value), f"{name} must be real")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TransientSourceProfileError(f"{name} must be a finite scalar") from exc
    _require(math.isfinite(out), f"{name} must be finite")
    return out


@dataclass(frozen=True)
class TransientSourceProfile:
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
    prony_mse_tolerance: float = 1.e-4
    prony_max_terms: int = 30
    prony_recipe: str = PRONY_RECIPE

    def __post_init__(self):
        _require(self.method == PROFILE_METHOD, "unsupported transient source profile method")
        tolerance = _number(self.prony_mse_tolerance, "prony_mse_tolerance")
        _require(1.e-14 <= tolerance <= 1.e-4, "Prony MSE tolerance may only tighten the native 1e-4")
        _require(type(self.prony_max_terms) is int and 4 <= self.prony_max_terms <= 160
                 and self.prony_max_terms % 2 == 0, "Prony maximum terms must be an even integer in 4..160")
        _require(self.prony_recipe == PRONY_RECIPE, "unknown Prony accuracy recipe")
        object.__setattr__(self, "prony_mse_tolerance", tolerance)
        _require(isinstance(self.hull_hash, str) and len(self.hull_hash) == 64, "invalid hull hash")
        _require(type(self.source_panel_limit) is int and 1 <= self.source_panel_limit <= MAX_SOURCE_PANELS,
                 "full source panel limit may only tighten 600")
        for name, dtype in (("lid_vertices", np.float64), ("lid_faces", np.int64),
                            ("source_vertices", np.float64), ("source_faces", np.int64)):
            object.__setattr__(self, name, _freeze(getattr(self, name), dtype, name))
        mask = np.asarray(self.hull_mask)
        _require(mask.dtype.kind == "b", "hull_mask must be boolean")
        object.__setattr__(self, "hull_mask", np.frombuffer(mask.tobytes(), dtype=np.bool_))
        for v, f, name in ((self.lid_vertices, self.lid_faces, "lid"),
                           (self.source_vertices, self.source_faces, "source")):
            _require(v.ndim == 2 and v.shape[1] == 3 and len(v) > 0, f"{name} vertices must be nonempty N by 3")
            _require(f.ndim == 2 and f.shape[1] == 4 and len(f) > 0, f"{name} faces must be nonempty N by 4")
            _require(int(f.min()) >= 0 and int(f.max()) < len(v), f"{name} face index out of range")
        _require(len(self.source_faces) <= self.source_panel_limit, "hull plus lid exceeds full source panel limit")
        _require(self.hull_mask.shape == (len(self.source_faces),) and bool(np.any(self.hull_mask))
                 and bool(np.any(~self.hull_mask)), "full source mask must identify hull and lid")
        _require(int(np.count_nonzero(~self.hull_mask)) == len(self.lid_faces), "lid/source face counts disagree")
        z, radius = _number(self.lid_z, "lid_z"), _number(self.faces_max_radius, "faces_max_radius")
        _require(math.isfinite(z) and z < 0.0 and math.isfinite(radius) and radius > 0.0,
                 "lid z must be negative and lid face radius positive")
        _require(bool(np.all(self.lid_vertices[:, 2] == z)), "lid must be exactly planar at its declared z")
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
        if self.prony_mse_tolerance != 1.e-4 or self.prony_max_terms != 30:
            out.update(prony_mse_tolerance=self.prony_mse_tolerance, prony_max_terms=self.prony_max_terms,
                       prony_recipe=self.prony_recipe)
        return out

    def content_hash(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()


def profile_from_request(request: Mapping[str, Any]):
    record = request.get(PROFILE_KEY)
    if record is None:
        return None
    _require(isinstance(record, Mapping), "source profile must be a neutral mapping")
    data = dict(record)
    expected = data.pop("profile_hash", None)
    # Check dimensions before constructing private array copies.
    for name in ("lid_faces", "source_faces"):
        _require(isinstance(data.get(name), (list, tuple)) and 0 < len(data[name]) <= MAX_SOURCE_PANELS,
                 f"{name} exceeds source limits")
    for name in ("lid_vertices", "source_vertices"):
        _require(isinstance(data.get(name), (list, tuple)) and 0 < len(data[name]) <= 8 * MAX_SOURCE_PANELS,
                 f"{name} exceeds source limits")
    try:
        profile = TransientSourceProfile(**data)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TransientSourceProfileError(f"invalid source profile: {exc}") from exc
    _require(expected == profile.content_hash(), "source profile hash differs from retained geometry/settings")
    _require(profile.hull_hash == request.get("geometry"), "source profile belongs to a different hull")
    return profile


def bind_profile(request, profile):
    _require(isinstance(profile, TransientSourceProfile), "explicit TransientSourceProfile required")
    _require(profile.hull_hash == request.get("geometry"), "source profile belongs to a different hull")
    out = dict(request)
    out[PROFILE_KEY] = dict(profile.to_dict(), profile_hash=profile.content_hash())
    return out


def create_source_profile(vertices, faces, surface_ids, *, lid_z, faces_max_radius=None,
                          water_depth, source_panel_limit=MAX_SOURCE_PANELS,
                          max_work_bytes=256 * 1024 * 1024, prony_mse_tolerance=1.e-4, prony_max_terms=30):
    """Explicit native lid generation; no physical hull clipping, repair or fallback."""
    import capytaine as cpt
    from .floating_radiation_fields import family_request
    from ._pressure_body import prepare_body
    _require(type(source_panel_limit) is int and 1 <= source_panel_limit <= MAX_SOURCE_PANELS,
             "full source panel limit may only tighten 600")
    _require(type(max_work_bytes) is int and 0 < max_work_bytes <= 256 * 1024 * 1024,
             "working-memory ceiling may only tighten 256 MiB")
    _require(len(faces) <= source_panel_limit and len(vertices) <= 4 * source_panel_limit,
             "physical source manifest exceeds bounded panel/vertex counts")
    _require(str(cpt.__version__).split(".")[:2] == ["3", "0"], "profile requires Capytaine 3.0")
    v, f, labels, mesh, _ = prepare_body(cpt, vertices, faces, surface_ids, period=math.tau,
                                       heading=0., water_depth=water_depth, rho=1025., max_panels=source_panel_limit)
    _require(not bool(np.any(np.all(v[f, 2] == 0., axis=1))),
             "physical hull includes a free-surface cap; provide the open wetted hull")
    z = _number(lid_z, "lid_z")
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
    return TransientSourceProfile(req["geometry"], lid.vertices, lid.faces,
                                  body.mesh_including_lid.vertices, body.mesh_including_lid.faces,
                                  np.asarray(body.hull_mask, dtype=bool), z, radius,
                                  source_panel_limit=source_panel_limit,
                                  prony_mse_tolerance=prony_mse_tolerance, prony_max_terms=prony_max_terms)


def body_for_profile(cpt, vertices, faces, labels, request, reference_omega):
    """Rebuild exact declared source geometry; no solve and no Green-function fit."""
    from ._pressure_body import prepare_body
    profile = profile_from_request(request)
    _require(profile is not None, "transient source profile is missing")
    v, f, labels, mesh, _ = prepare_body(cpt, vertices, faces, labels, period=math.tau / float(reference_omega),
                                       heading=request["heading"], water_depth=request["water_depth"],
                                       rho=request["rho"], max_panels=profile.source_panel_limit)
    from .floating_radiation_fields import family_request, DOF_NAMES
    actual = family_request(v, f, labels, heading=request["heading"], water_depth=request["water_depth"],
                            rho=request["rho"], gravity=request["gravity"], Q0=request["Q0"], cg_world=request["cg_world"])
    _require(actual["geometry"] == profile.hull_hash, "profile hull geometry changed")
    _require(profile.lid_z > -float(request["water_depth"]), "lid intersects seabed")
    lid = cpt.Mesh(vertices=profile.lid_vertices.copy(), faces=profile.lid_faces.copy(), auto_check=False)
    body = cpt.FloatingBody(mesh=mesh, lid_mesh=lid, center_of_mass=np.asarray(request["cg_world"]))
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
    return profile


def solve_profile_snapshot(vertices, faces, surface_ids, omega, *, profile, **kwargs):
    """Explicit Prepare operation, sharing canonical native solve/retention code."""
    from .floating_radiation_fields import solve_family_snapshot
    return solve_family_snapshot(vertices, faces, surface_ids, omega, source_profile=profile, **kwargs)


def _prony_target(kh, values):
    """Capytaine 3.0's finite-depth target, with its removable +kh pole evaluated stably."""
    kh = float(kh)
    _require(kh > .1, "finite-depth kh <= 0.1 remains unsupported by this profile")
    x = np.asarray(values, dtype=float)
    if kh >= 1.e5:
        return 1. - np.tanh(x)
    t = math.tanh(kh)
    c = (1. + t)**2 / (1. - t*t + t / kh)
    delta = x - kh
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ((x + kh*t) * np.exp(x)) / (x*np.sinh(x) - kh*t*np.cosh(x)) - c / delta - 2.
    near = np.abs(delta) < 1.e-3
    if bool(np.any(near)):
        d = delta[near]
        d2 = d*d
        v = 1. - d2/3. + 2.*d2*d2/15. - 17.*d2*d2*d2/315.
        w = d/3. - 2.*d*d2/15. + 17.*d*d2*d2/315.
        h = t + kh*(1. - t*t)
        b = kh*(1. + t)
        a = t + v*(kh*(1. - t*t) + d)
        out[near] = (1. + t) * (b*kh*(1. - t*t)*w + h + b*v*(h - 1.) + d*v*h) / (a*h) - 2.
    _require(bool(np.all(np.isfinite(out))), "native Prony target is non-finite")
    return out


def solver_for_profile(cpt, profile):
    """Explicit accuracy settings; the public native fitter retains its own domain-noise RNG.

    Actual fitted coefficients, not just a tolerance or seed, are retained by
    every snapshot. No vendor globals or parent random generators are changed.
    """
    if profile.prony_mse_tolerance == 1.e-4 and profile.prony_max_terms == 30:
        return cpt.BEMSolver()
    default = cpt.BEMSolver().engine.green_function
    tolerance, maximum = profile.prony_mse_tolerance, profile.prony_max_terms

    class TransientDelhommeau(type(default)):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            _require(self.finite_depth_prony_decomposition_method.lower() == "python", "profile requires native Python Prony")
            self.exportable_settings.update(transient_prony_mse_tolerance=tolerance,
                                            transient_prony_max_terms=maximum, transient_prony_recipe=PRONY_RECIPE)

        @lru_cache(maxsize=256)
        def find_best_exponential_decomposition(self, dimensionless_wavenumber, *, method=None):
            from capytaine.tools.prony_decomposition import find_best_exponential_decomposition, PronyDecompositionFailure
            _require(method is None or str(method).lower() == "python", "unsupported Prony method")
            kh = float(dimensionless_wavenumber)
            try:
                a, rates = find_best_exponential_decomposition(lambda x: _prony_target(kh, x), -.1, 20.,
                                                               range(4, maximum + 1, 2), tol=tolerance)
            except PronyDecompositionFailure as exc:
                raise TransientSourceProfileError(f"native Prony accuracy target was not reached: {exc}") from exc
            return np.stack([rates, a])

    return cpt.BEMSolver(green_function=TransientDelhommeau())


def replay_profile_green_function(cpt, snapshot, problem):
    """Restore the captured profile's native kernel and fit; no fit or BEM solve."""
    default = cpt.BEMSolver().engine.green_function
    settings = dict(snapshot.gf_settings or {})
    try:
        tolerance = settings.pop("transient_prony_mse_tolerance")
        maximum = settings.pop("transient_prony_max_terms")
        recipe = settings.pop("transient_prony_recipe")
    except KeyError as exc:
        raise TransientSourceProfileError("retained accuracy profile is incomplete") from exc
    _require(recipe == PRONY_RECIPE and 1.e-14 <= tolerance <= 1.e-4
             and type(maximum) is int and 4 <= maximum <= 160 and maximum % 2 == 0,
             "unsupported retained Prony accuracy settings")
    effective = {k: v for k, v in default.exportable_settings.items() if k != "green_function"}
    from .wave_surface_fields import _json
    _require(_json(settings) == _json(effective), "installed native Green-function settings differ from the captured profile")
    kh = float(problem.wavenumber * problem.water_depth)
    _require(kh == snapshot.prony_kh and snapshot.prony is not None, "retained Prony fit belongs to another kh")
    retained = np.array(snapshot.prony, copy=True)

    class Replay(type(default)):
        def find_best_exponential_decomposition(self, dimensionless_wavenumber, *, method=None):
            requested = self.finite_depth_prony_decomposition_method if method is None else str(method).lower()
            _require(requested == snapshot.prony_method and float(dimensionless_wavenumber) == kh,
                     "replay requested a Prony fit that was not captured")
            return retained.copy()
        __hash__ = type(default).__hash__

    Replay.__name__ = TIGHT_GREEN_NAME
    out = Replay(**effective)
    out.exportable_settings.update(transient_prony_mse_tolerance=tolerance,
                                   transient_prony_max_terms=maximum, transient_prony_recipe=recipe)
    return out
