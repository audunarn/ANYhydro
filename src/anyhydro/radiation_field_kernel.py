"""Time-domain radiation field kernels: physical panel pressure and waterline elevation.

Companion of :mod:`anyhydro.radiation`.  For each rigid coordinate ``j`` (six mean-body DOFs) a
physical field ``F`` (panel pressure or waterline elevation) radiated by the body motion is

    F(t) = P_a q_ddot(t) + integral_0^t P_v(tau) q_dot(t - tau) d tau

with phasor ``x(t) = Re{X exp(-i w t)}``.  For a native transfer ``H(w)`` (field per unit
displacement amplitude)

    H(w) = -w^2 P_a - i w C(w) + w S(w),    C = int P_v cos(w t) dt,  S = int P_v sin(w t) dt

so ``C(w) = -Im H / w`` and the kernel is the cosine transform ``P_v(t) = (2/pi) int C(w) cos(w t) dw``
(the same construction, quadrature, low/high tails and time window as the generalized kernel in
``radiation.py``, whose private core routines are reused).  ``P_a`` is the computed infinite-frequency
coefficient ``-Re(H)/w^2`` at ``w -> inf`` supplied from the native symbolic ``omega = inf`` solve;
it is never extrapolated from the last finite-frequency sample and never inferred from a 6x6 matrix.

Signed physical-pressure projection: with ``Proj`` (6 x n_panel) the work-conjugate map from panel
pressure to generalized force, ``Proj @ P_a = -A_inf`` and ``Proj @ P_v(t) = -K_r(t)`` (to the
caller tolerance, default the existing 1e-8 source tolerance).  Elevation kernels are not projected.

Offline evaluation: after the rigid trajectory is known, field histories are formed by the SAME
discrete trapezoidal rule as the online global memory,

    F_n = P_a a_n + dt (0.5 P_v[n] v_0 + sum_{j=1}^{n-1} P_v[n-j] v_j + 0.5 P_v[0] v_n),

either by a direct sum (independent oracle) or by zero-padded block FFT convolution in panel chunks.
No SVD/Prony/ROM approximation.  Artifacts are development-only and always ``unqualified``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Optional, Tuple

import numpy as np

from .errors import HydroError
from .radiation import (
    HIGH_TAIL_TRUNCATE,
    A_INF_COMPUTED,
    KERNEL_SYMMETRISATION_NONE,
    LOW_TAIL_EXPLICIT_ZERO,
    QUADRATURE_PIECEWISE_LINEAR,
    TIME_WINDOW_BARTLETT,
    RadiationKernel,
    _frozen_float_array as _radiation_frozen_float_array,
    _piecewise_linear_cosine_integral,
    _trapezoid_weights,
)

#: documented numeric-array digest protocol shared (by specification, not by import) with ANYsolver's
#: ``numeric_array_digest``: sha256(b"numeric-array-v1|" + repr(shape tuple of ints) + b"|" + C-order '<f8' bytes)
DIGEST_PROTOCOL = "numeric-array-v1"


def numeric_array_digest(array: Any) -> str:
    """Digest of a numeric array under the ``numeric-array-v1`` protocol (matches the solver result digests)."""
    raw = _real_array(array, "array")
    h = hashlib.sha256()
    h.update(DIGEST_PROTOCOL.encode() + b"|" + repr(tuple(int(x) for x in raw.shape)).encode() + b"|")
    # C-order buffered iteration in chunks of at most 65536 elements: strided / transposed / non-native-dtype input is
    # never materialised as a full contiguous copy and only a 512 KiB chunk is converted and hashed at a time
    if raw.size:
        it = np.nditer(raw, flags=["external_loop", "buffered", "zerosize_ok"], op_dtypes=[np.dtype("<f8")],
                       casting="safe", order="C", buffersize=1 << 16)
        for chunk in it:
            h.update(np.ascontiguousarray(chunk, dtype="<f8").tobytes())
    return h.hexdigest()


RADIATION_FIELD_FORMAT = "anyhydro-radiation-field-kernel-v1"
RADIATION_FIELD_VERSION = "1.0.0"
N_DOF = 6
PROJECTION_REL_TOL = 1.0e-8
MAX_PANELS = 600
MAX_QUERY_POINTS = 2000
DEFAULT_WORK_BYTES = 256 * 1024 * 1024
DEFAULT_ARTIFACT_BYTES = 2 * 1024**3

__all__ = [
    "RadiationFieldError",
    "RadiationFieldLimitError",
    "RadiationFieldSerializationError",
    "FieldResourceLimits",
    "FieldReconstructionCriteria",
    "RadiationFieldKernel",
    "build_radiation_field_kernel",
    "build_radiation_field_kernel_from_native",
    "native_binding_record",
    "BOUNDARY_POLICY_NATIVE_STRICT",
    "BOUNDARY_POLICY_EXACT_DIRICHLET_Z0",
    "BOUNDARY_POLICIES",
    "NativeBoundaryError",
    "field_history_direct",
    "field_history_fft",
    "iter_field_history",
    "load_radiation_field_kernel",
    "NativeFieldCoefficients",
    "field_history_from_transient",
    "numeric_array_digest",
    "recover_waterline_coefficients",
    "load_native_field_coefficients",
    "native_infinite_frequency_field_coefficients",
    "physical_force_projection",
]


class RadiationFieldError(HydroError):
    """A radiation field kernel input, result or artifact is invalid."""


class RadiationFieldLimitError(RadiationFieldError):
    """A declared storage / memory / panel limit would be exceeded (refused before allocation)."""


class RadiationFieldSerializationError(RadiationFieldError):
    """A field-kernel artifact could not be saved, loaded or verified."""


class NativeBoundaryError(RadiationFieldError):
    """The native omega = inf free-surface boundary check FAILED (raw residual above the declared roundoff scale).

    ``diagnostic`` is the complete (unadmitted) :class:`NativeFieldCoefficients` snapshot with the raw residual,
    limit and ``boundary_status == 'failed'``; it can be saved for inspection but is refused by every production
    path (builder binding, recovery, ``builder_arrays``).  Nothing is clamped, zeroed or loosened."""

    def __init__(self, message: str, diagnostic: Any = None):
        super().__init__(message)
        self.diagnostic = diagnostic


def _require(cond: bool, message: str, error=RadiationFieldError) -> None:
    if not cond:
        raise error(message)


def _real_array(value: Any, name: str) -> np.ndarray:
    """Float64 REAL data; complex input (even with zero imaginary part) is refused BEFORE any cast.  Used at every
    real-valued boundary of this module; the explicitly complex interfaces (finite-frequency transfers, retained
    normalised native sources / potential) do not call it."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise RadiationFieldError(f"{name}: not a numeric array ({exc})") from exc
    if raw.dtype.kind == "c":
        raise RadiationFieldError(f"{name}: complex-valued input is refused (real coefficients required; "
                                  "no imaginary part is dropped)")
    try:
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RadiationFieldError(f"{name}: not a numeric array ({exc})") from exc


def _real_array_any(value: Any, name: str) -> np.ndarray:
    """Array of unchanged dtype (bool / int / float) after refusing complex input."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise RadiationFieldError(f"{name}: not a valid array ({exc})") from exc
    if raw.dtype.kind == "c":
        raise RadiationFieldError(f"{name}: complex-valued input is refused for a real-valued quantity")
    return raw


def _frozen_float_array(value, name: str, ndim: Optional[int] = None) -> np.ndarray:
    """Owned immutable real array: the complex guard runs here, then the shared (unmodified) radiation helper."""
    arr = _real_array(value, name)
    try:
        return _radiation_frozen_float_array(arr, name, ndim=ndim)
    except HydroError as exc:
        raise RadiationFieldError(str(exc)) from exc


@dataclass(frozen=True)
class FieldResourceLimits:
    """Owner limits.  Exceeding one refuses before the corresponding allocation."""

    max_panels: int = MAX_PANELS
    max_query_points: int = MAX_QUERY_POINTS
    max_work_bytes: int = DEFAULT_WORK_BYTES
    max_artifact_bytes: int = DEFAULT_ARTIFACT_BYTES

    def __post_init__(self) -> None:
        # an explicit configuration may only TIGHTEN the approved ceilings
        ceilings = {"max_panels": MAX_PANELS, "max_query_points": MAX_QUERY_POINTS,
                    "max_work_bytes": DEFAULT_WORK_BYTES, "max_artifact_bytes": DEFAULT_ARTIFACT_BYTES}
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            _require(isinstance(value, (int, np.integer)) and not isinstance(value, bool) and 0 < value <= ceiling,
                     f"{name} must be a positive integer not above the approved ceiling {ceiling}",
                     RadiationFieldLimitError)

    def to_dict(self) -> dict:
        return {k: int(getattr(self, k)) for k in
                ("max_panels", "max_query_points", "max_work_bytes", "max_artifact_bytes")}


@dataclass(frozen=True)
class FieldReconstructionCriteria:
    """Caller criteria for the full-field transfer reconstruction (all required, no blanket default).

    Per window frequency: ``||H_rec - H||_F <= rel_tol * ||H||_F + abs_tol`` over the whole field block.
    """

    omega_min: float
    omega_max: float
    rel_tol: float
    abs_tol: float

    def __post_init__(self) -> None:
        for name in ("omega_min", "omega_max", "rel_tol", "abs_tol"):
            v = float(getattr(self, name))
            _require(np.isfinite(v) and v >= 0.0, f"{name} must be finite and >= 0")
            object.__setattr__(self, name, v)
        _require(0.0 < self.omega_min <= self.omega_max, "need 0 < omega_min <= omega_max")

    def to_dict(self) -> dict:
        return {"omega_min": self.omega_min, "omega_max": self.omega_max,
                "rel_tol": self.rel_tol, "abs_tol": self.abs_tol}


def _canonical(value: Any) -> str:
    def conv(v):
        if isinstance(v, Mapping):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v
    return json.dumps(conv(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _freeze_meta(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_meta(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_meta(v) for v in value)
    return value


def _chunk_bounds(n: int, chunk: int) -> list:
    return [(s, min(s + chunk, n)) for s in range(0, n, chunk)]


@dataclass(frozen=True)
class RadiationFieldKernel:
    """Immutable chunked field kernel with one coherent reference to the generalized kernel.

    ``pressure_a``/``elevation_a`` are the instantaneous coefficients (n_field, 6); the memory
    kernels are tuples of chunks ``(n_time, n_chunk, 6)`` over the field axis.  The distinct field
    dimension is not an extra rigid DOF and is not subject to the 12-DOF kernel limit.
    """

    kernel_hash: str
    settings: Mapping[str, Any]
    dof_keys: Tuple[str, ...]
    time: np.ndarray
    panel_ids: Tuple[str, ...]
    query_point_ids: Tuple[str, ...]
    pressure_a: np.ndarray
    pressure_chunks: Tuple[np.ndarray, ...]
    elevation_a: np.ndarray
    elevation_chunks: Tuple[np.ndarray, ...]
    projection: np.ndarray
    added_mass_infinite: np.ndarray
    diagnostics: Mapping[str, Any]
    identity: Mapping[str, Any]
    chunk_size: int
    # numeric_array_digest (protocol numeric-array-v1) of the generalized kernel samples / A_inf this field kernel
    # was built from; empty = UNBOUND (development fixture): the bound wrapper field_history_from_transient refuses it
    kernel_digest: str = ""
    ainf_digest: str = ""

    def __post_init__(self) -> None:
        _require(isinstance(self.chunk_size, (int, np.integer)) and not isinstance(self.chunk_size, bool)
                 and int(self.chunk_size) >= 1, "chunk_size must be a positive integer")
        object.__setattr__(self, "chunk_size", int(self.chunk_size))
        for dname in ("kernel_digest", "ainf_digest"):
            dv = getattr(self, dname)
            _require(isinstance(dv, str) and (dv == "" or (len(dv) == 64 and all(c in "0123456789abcdef" for c in dv))),
                     f"{dname} must be empty (unbound) or a sha256 hex digest")
        _require(isinstance(self.kernel_hash, str) and len(self.kernel_hash) == 64, "kernel_hash must be a sha256 hex digest")
        time = _frozen_float_array(self.time, "time", ndim=1)
        n_t = time.size
        _require(n_t >= 2 and time[0] == 0.0, "time must start at zero with >= 2 samples")
        dt = float(time[1] - time[0])
        _require(dt > 0.0 and np.allclose(np.diff(time), dt, rtol=1e-9, atol=0.0), "time must be uniform")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "dof_keys", tuple(str(k) for k in self.dof_keys))
        _require(len(self.dof_keys) == N_DOF, f"exactly {N_DOF} rigid DOF keys are required")
        pids = tuple(str(p) for p in self.panel_ids)
        qids = tuple(str(q) for q in self.query_point_ids)
        _require(len(set(pids)) == len(pids) and len(set(qids)) == len(qids), "panel / query-point ids must be unique")
        object.__setattr__(self, "panel_ids", pids)
        object.__setattr__(self, "query_point_ids", qids)
        for name, ids, a, chunks in (("pressure", pids, self.pressure_a, self.pressure_chunks),
                                     ("elevation", qids, self.elevation_a, self.elevation_chunks)):
            a_f = _frozen_float_array(a, f"{name}_a", ndim=2)
            _require(a_f.shape == (len(ids), N_DOF), f"{name}_a must have shape {(len(ids), N_DOF)}")
            frozen = tuple(_frozen_float_array(c, f"{name}_chunks", ndim=3) for c in chunks)
            bounds = _chunk_bounds(len(ids), int(self.chunk_size))
            _require(len(frozen) == len(bounds), f"{name}: chunk count does not match chunk_size")
            for c, (s, e) in zip(frozen, bounds):
                _require(c.shape == (n_t, e - s, N_DOF), f"{name}: chunk shape {c.shape} != {(n_t, e - s, N_DOF)}")
            object.__setattr__(self, f"{name}_a", a_f)
            object.__setattr__(self, f"{name}_chunks", frozen)
        proj = _frozen_float_array(self.projection, "projection", ndim=2)
        _require(proj.shape == (N_DOF, len(pids)), "projection must have shape (6, n_panel)")
        object.__setattr__(self, "projection", proj)
        a_inf = _frozen_float_array(self.added_mass_infinite, "added_mass_infinite", ndim=2)
        _require(a_inf.shape == (N_DOF, N_DOF), "added_mass_infinite must be 6x6")
        object.__setattr__(self, "added_mass_infinite", a_inf)
        object.__setattr__(self, "settings", _freeze_meta(dict(self.settings)))
        object.__setattr__(self, "diagnostics", _freeze_meta(dict(self.diagnostics)))
        object.__setattr__(self, "identity", _freeze_meta(dict(self.identity)))

    # ------------------------------------------------------------------ views
    @property
    def dt(self) -> float:
        return float(self.time[1] - self.time[0])

    @property
    def n_time(self) -> int:
        return int(self.time.size)

    @property
    def qualification_status(self) -> str:
        return "unqualified"

    @property
    def stored_bytes(self) -> int:
        n = sum(c.nbytes for c in self.pressure_chunks) + sum(c.nbytes for c in self.elevation_chunks)
        return int(n + self.pressure_a.nbytes + self.elevation_a.nbytes + self.projection.nbytes)

    def _parts(self, kind: str):
        if kind == "pressure":
            return self.pressure_a, self.pressure_chunks
        if kind == "elevation":
            return self.elevation_a, self.elevation_chunks
        raise RadiationFieldError("kind must be 'pressure' or 'elevation'")

    def kernel_dense(self, kind: str, limits: Optional[FieldResourceLimits] = None) -> np.ndarray:
        """Collect the (n_time, n_field, 6) kernel; refuses above the working-memory limit."""
        a, chunks = self._parts(kind)
        lim = limits or FieldResourceLimits()
        need = self.n_time * a.shape[0] * N_DOF * 8
        _require(need <= lim.max_work_bytes, f"dense {kind} kernel needs {need} B > work limit {lim.max_work_bytes} B",
                 RadiationFieldLimitError)
        return np.concatenate(chunks, axis=1) if chunks else np.zeros((self.n_time, 0, N_DOF))

    # ------------------------------------------------------------------ checks
    def reconstruct(self, kind: str, omegas: Any) -> np.ndarray:
        """H_rec(w) = -w^2 P_a - i w C_rec + w S_rec, shape (n_w, n_field, 6)."""
        w = np.atleast_1d(_real_array(omegas, "omegas"))
        _require(w.ndim == 1 and np.all(np.isfinite(w)) and np.all(w > 0.0), "omegas must be finite and > 0")
        a, chunks = self._parts(kind)
        tw = _trapezoid_weights(self.time)
        cos_wt = np.cos(np.outer(w, self.time))
        sin_wt = np.sin(np.outer(w, self.time))
        out = np.empty((w.size, a.shape[0], N_DOF), dtype=np.complex128)
        for (s, e), c in zip(_chunk_bounds(a.shape[0], self.chunk_size), chunks):
            wk = (tw[:, None] * c.reshape(self.n_time, -1))
            cc = (cos_wt @ wk).reshape(w.size, e - s, N_DOF)
            ss = (sin_wt @ wk).reshape(w.size, e - s, N_DOF)
            out[:, s:e] = (-(w**2)[:, None, None] * a[None, s:e] - 1j * w[:, None, None] * cc
                           + w[:, None, None] * ss)
        return out

    def reconstruction_report(self, kind: str, omegas: Any, transfer: Any, criteria: FieldReconstructionCriteria) -> dict:
        w = _real_array(omegas, "omegas")
        sel = (w >= criteria.omega_min) & (w <= criteria.omega_max)
        _require(int(np.count_nonzero(sel)) >= 1, "no source frequency lies inside the criteria window")
        ref = np.asarray(transfer, dtype=np.complex128)[sel]
        rec = self.reconstruct(kind, w[sel])
        err = np.sqrt(np.sum(np.abs(rec - ref) ** 2, axis=(1, 2)))
        nrm = np.sqrt(np.sum(np.abs(ref) ** 2, axis=(1, 2)))
        allowed = criteria.rel_tol * nrm + criteria.abs_tol
        return {"kind": kind, "n_frequencies": int(ref.shape[0]), "max_abs_error": float(err.max()),
                "max_rel_error": float(np.max(err / np.where(nrm > 0.0, nrm, 1.0))),
                "worst_error_to_allowed": float(np.max(err / np.where(allowed > 0.0, allowed, 1.0))),
                "passed": bool(np.all(err <= allowed)), "criteria": criteria.to_dict()}

    def projection_errors(self) -> dict:
        """Max |Proj P_a + A_inf| and max |Proj P_v(t) + K_r(t)| are stored at build; recomputed here for P_a only."""
        return {"instantaneous": float(np.max(np.abs(self.projection @ self.pressure_a + self.added_mass_infinite)))}

    # ------------------------------------------------------------------ integrity
    def _meta(self) -> dict:
        return {
            "format": RADIATION_FIELD_FORMAT, "version": RADIATION_FIELD_VERSION, "kernel_hash": self.kernel_hash,
            "settings": self.settings, "dof_keys": list(self.dof_keys), "panel_ids": list(self.panel_ids),
            "query_point_ids": list(self.query_point_ids), "diagnostics": self.diagnostics,
            "identity": self.identity, "chunk_size": int(self.chunk_size),
            "n_pressure_chunks": len(self.pressure_chunks), "n_elevation_chunks": len(self.elevation_chunks),
            "kernel_digest": self.kernel_digest, "ainf_digest": self.ainf_digest,
            "digest_protocol": DIGEST_PROTOCOL,
            "qualification_status": "unqualified",
        }

    def content_hash(self) -> str:
        h = hashlib.sha256()
        meta = _canonical(self._meta()).encode("utf-8")
        h.update(len(meta).to_bytes(8, "little"))
        h.update(meta)
        arrays = [("time", self.time), ("pressure_a", self.pressure_a), ("elevation_a", self.elevation_a),
                  ("projection", self.projection), ("added_mass_infinite", self.added_mass_infinite)]
        arrays += [(f"pressure_chunk_{i}", c) for i, c in enumerate(self.pressure_chunks)]
        arrays += [(f"elevation_chunk_{i}", c) for i, c in enumerate(self.elevation_chunks)]
        for name, arr in arrays:
            data = np.ascontiguousarray(arr, dtype="<f8").tobytes()
            h.update(name.encode("utf-8"))
            h.update(len(data).to_bytes(8, "little"))
            h.update(data)
        return h.hexdigest()

    def save(self, path: str, limits: Optional[FieldResourceLimits] = None) -> str:
        """Atomic checksummed ``.npz``; refuses above the artifact ceiling before writing."""
        lim = limits or FieldResourceLimits()
        _require(2 * self.stored_bytes <= lim.max_artifact_bytes,
                 f"artifact needs {self.stored_bytes} B (2x on reload) > ceiling {lim.max_artifact_bytes} B",
                 RadiationFieldLimitError)
        digest = self.content_hash()
        payload = {"time": self.time, "pressure_a": self.pressure_a, "elevation_a": self.elevation_a,
                   "projection": self.projection, "added_mass_infinite": self.added_mass_infinite}
        payload.update({f"pressure_chunk_{i}": c for i, c in enumerate(self.pressure_chunks)})
        payload.update({f"elevation_chunk_{i}": c for i, c in enumerate(self.elevation_chunks)})
        header = _canonical({"meta": self._meta(), "content_hash": digest}).encode("utf-8")
        payload["header"] = np.frombuffer(header, dtype=np.uint8)
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(handle, **payload)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return digest


MAX_NPZ_MEMBERS = 4096
MAX_HEADER_BYTES = 1 << 20


def _inspect_npz_artifact(path: str, err) -> tuple:
    """Inspect an ``.npz`` WITHOUT materialising any payload array.

    Reads the zip directory and each member's ``.npy`` header only (shape, dtype, uncompressed size), refuses
    object dtypes, unsupported header versions, declared-size inconsistencies, too many members and an oversize
    metadata member, and parses only the bounded ``header`` member (via ``read_array``, not ``np.load``).
    Returns ``({key: (shape, dtype, uncompressed_bytes)}, header_dict)`` for the payload members.
    """
    import io
    import zipfile
    members: dict = {}
    header_raw = None
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            _require(0 < len(infos) <= MAX_NPZ_MEMBERS, "artifact has no members or too many members", err)
            for info in infos:
                name = info.filename
                _require(name.endswith(".npy") and "/" not in name and "\\" not in name and len(name) > 4,
                         f"unsafe or non-array member {name!r}", err)
                key = name[:-4]
                _require(key not in members and not (key == "header" and header_raw is not None),
                         f"duplicate member {name!r}", err)
                with zf.open(info) as fh:
                    version = np.lib.format.read_magic(fh)
                    if version == (1, 0):
                        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(fh)
                    elif version == (2, 0):
                        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(fh)
                    else:
                        raise err(f"member {name!r}: unsupported .npy version {version}")
                _require(not dtype.hasobject and dtype.kind in "fiuc", f"member {name!r}: unsafe dtype {dtype}", err)
                count = 1
                for d in shape:
                    _require(isinstance(d, int) and d >= 0, f"member {name!r}: invalid shape {shape}", err)
                    count *= int(d)
                payload = count * dtype.itemsize
                _require(payload <= info.file_size <= payload + 65536,
                         f"member {name!r}: declared size {info.file_size} B inconsistent with its header", err)
                if key == "header":
                    _require(info.file_size <= MAX_HEADER_BYTES and dtype == np.dtype("u1") and len(shape) == 1,
                             "metadata header member is oversize or not a uint8 vector", err)
                    header_raw = zf.read(info)
                members[key] = (tuple(int(d) for d in shape), dtype, int(info.file_size))
            _require(header_raw is not None, "artifact has no metadata header", err)
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as exc:
        raise err(f"artifact is corrupt or unreadable: {exc}") from exc
    try:
        arr = np.lib.format.read_array(io.BytesIO(header_raw), allow_pickle=False)
        header = json.loads(bytes(arr.tobytes()).decode("utf-8"))
    except Exception as exc:
        raise err(f"artifact metadata header is malformed: {exc}") from exc
    _require(isinstance(header, dict) and isinstance(header.get("meta"), dict), "metadata header lacks 'meta'", err)
    del members["header"]
    return members, header


def load_radiation_field_kernel(path: str, *, expected_kernel_hash: Optional[str] = None,
                                limits: Optional[FieldResourceLimits] = None) -> RadiationFieldKernel:
    """Load and verify (format, structure, kernel reference and content hash); no pickle."""
    lim = limits or FieldResourceLimits()
    err = RadiationFieldSerializationError
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise err(f"cannot read field-kernel artifact: {exc}") from exc
    # reading + freezing holds two copies of every array
    _require(2 * size <= lim.max_artifact_bytes,
             f"artifact file {size} B (2x while loading) exceeds ceiling {lim.max_artifact_bytes} B",
             RadiationFieldLimitError)
    # ---- admission BEFORE any array is materialised: zip directory, .npy headers, exact key/shape/dtype set
    members, header = _inspect_npz_artifact(path, err)
    try:
        meta = header["meta"]
        _require(meta.get("format") == RADIATION_FIELD_FORMAT, "unknown field-kernel format", err)
        n_p, n_e = int(meta["n_pressure_chunks"]), int(meta["n_elevation_chunks"])
        n_panel, n_query, cs = len(meta["panel_ids"]), len(meta["query_point_ids"]), int(meta["chunk_size"])
        _require(0 < n_panel <= lim.max_panels and 0 < n_query <= lim.max_query_points and cs >= 1,
                 "declared panel / query-point counts or chunk size outside the limits", RadiationFieldLimitError)
        _require("time" in members and len(members["time"][0]) == 1, "time member missing or not 1-D", err)
        n_t = int(members["time"][0][0])
        expected = {"time": (n_t,), "pressure_a": (n_panel, N_DOF), "elevation_a": (n_query, N_DOF),
                    "projection": (N_DOF, n_panel), "added_mass_infinite": (N_DOF, N_DOF)}
        for tag, count, n_f in (("pressure", n_p, n_panel), ("elevation", n_e, n_query)):
            bounds = _chunk_bounds(n_f, cs)
            _require(count == len(bounds), f"{tag} chunk count does not match chunk_size", err)
            for i, (s, e) in enumerate(bounds):
                expected[f"{tag}_chunk_{i}"] = (n_t, e - s, N_DOF)
        _require(set(members) == set(expected), f"unexpected or missing arrays: {sorted(set(members) ^ set(expected))}", err)
        total = 0
        for key, shape in expected.items():
            got_shape, got_dtype, _size = members[key]
            _require(got_shape == shape and got_dtype == np.dtype("<f8"),
                     f"array {key!r} has shape/dtype {got_shape}/{got_dtype}, expected {shape}/float64", err)
            count = 1
            for d in shape:
                count *= int(d)
            total += count * 8
        _require(2 * total <= lim.max_artifact_bytes,
                 f"expanded arrays {total} B (2x while loading) exceed ceiling {lim.max_artifact_bytes} B",
                 RadiationFieldLimitError)
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in expected}
        kernel = RadiationFieldKernel(
            kernel_hash=meta["kernel_hash"], settings=meta["settings"], dof_keys=tuple(meta["dof_keys"]),
            time=arrays["time"], panel_ids=tuple(meta["panel_ids"]), query_point_ids=tuple(meta["query_point_ids"]),
            pressure_a=arrays["pressure_a"], pressure_chunks=tuple(arrays[f"pressure_chunk_{i}"] for i in range(n_p)),
            elevation_a=arrays["elevation_a"], elevation_chunks=tuple(arrays[f"elevation_chunk_{i}"] for i in range(n_e)),
            projection=arrays["projection"], added_mass_infinite=arrays["added_mass_infinite"],
            diagnostics=meta["diagnostics"], identity=meta["identity"], chunk_size=int(meta["chunk_size"]),
            kernel_digest=str(meta.get("kernel_digest", "")), ainf_digest=str(meta.get("ainf_digest", "")),
        )
    except RadiationFieldError:
        raise
    except Exception as exc:  # corrupt zip, missing key, bad shapes
        raise err(f"field-kernel artifact is corrupt or malformed: {exc}") from exc
    if kernel.content_hash() != header.get("content_hash"):
        raise err("content hash mismatch: the artifact was modified or corrupted")
    if expected_kernel_hash is not None and kernel.kernel_hash != expected_kernel_hash:
        raise err("artifact references a different generalized RadiationKernel than expected")
    return kernel


# ----------------------------------------------------------------------------- builder

def _field_kernel_chunk(settings, omega_ext: np.ndarray, c_ext: np.ndarray, t: np.ndarray) -> np.ndarray:
    """P_v chunk = (2/pi) int C cos(w t) dw with the generalized-kernel quadrature, then the time window."""
    if settings.quadrature_method == QUADRATURE_PIECEWISE_LINEAR:
        k = (2.0 / np.pi) * _piecewise_linear_cosine_integral(omega_ext, c_ext, t)
    else:
        weights = _trapezoid_weights(omega_ext)
        flat = c_ext.reshape(omega_ext.size, -1)
        k = ((2.0 / np.pi) * (np.cos(np.outer(t, omega_ext)) @ (weights[:, None] * flat))).reshape((t.size,) + c_ext.shape[1:])
    if settings.time_window is not None:
        if settings.time_window_shape == TIME_WINDOW_BARTLETT:
            k *= np.clip(1.0 - t / settings.time_window, 0.0, None)[:, None, None]
        k[t > settings.time_window] = 0.0
    _require(bool(np.all(np.isfinite(k))), "computed field kernel contains non-finite values")
    return k


def _check_transfer(name: str, omegas: np.ndarray, transfer: Any, inf: Any, zero: Any, n_field: int,
                    explicit_zero: bool):
    h = np.array(transfer, dtype=np.complex128, copy=True)
    _require(h.shape == (omegas.size, n_field, N_DOF), f"{name}_transfer must have shape {(omegas.size, n_field, N_DOF)}")
    _require(bool(np.all(np.isfinite(h))), f"{name}_transfer contains non-finite values")
    a = np.array(_real_array(inf, f"{name}_inf"), dtype=np.float64, copy=True)
    _require(a.shape == (n_field, N_DOF) and bool(np.all(np.isfinite(a))),
             f"{name}_inf must be a finite {(n_field, N_DOF)} array from the native omega=inf solve")
    if explicit_zero:
        _require(zero is not None, f"{name}_zero_frequency is required by the explicit zero-frequency tail policy")
        z = np.array(_real_array(zero, f"{name}_zero_frequency"), dtype=np.float64, copy=True)
        _require(z.shape == (n_field, N_DOF) and bool(np.all(np.isfinite(z))), f"{name}_zero_frequency must be finite {(n_field, N_DOF)}")
    else:
        _require(zero is None, f"{name}_zero_frequency must not be supplied with the linear-to-zero extension")
        z = np.zeros((n_field, N_DOF))
    return h, a, z


def build_radiation_field_kernel(
    *,
    kernel: RadiationKernel,
    omegas: Any,
    pressure_transfer: Any,
    pressure_inf: Any,
    projection: Any,
    panel_ids: Any,
    elevation_transfer: Any,
    elevation_inf: Any,
    query_point_ids: Any,
    pressure_zero_frequency: Any = None,
    elevation_zero_frequency: Any = None,
    pressure_criteria: FieldReconstructionCriteria,
    elevation_criteria: FieldReconstructionCriteria,
    projection_rel_tol: float = PROJECTION_REL_TOL,
    limits: Optional[FieldResourceLimits] = None,
    chunk_size: int = 64,
    identity: Optional[Mapping[str, Any]] = None,
    _native_binding: Optional[Mapping[str, Any]] = None,
    _development: bool = False,
) -> RadiationFieldKernel:
    """UNBOUND development primitive: build pressure and elevation memory kernels from RAW caller arrays.

    ``_development`` (internal; only the explicit development factory sets it): a failed transfer reconstruction is
    RETAINED in the diagnostics (``unpassed_gates``) instead of refusing and the hashed identity carries
    ``development='unverified_hydrodynamics'``.  The signed projection, limits and every other check still refuse.

    Nothing here proves that ``pressure_inf`` / ``elevation_inf`` / ``projection`` came from an admitted native
    omega = inf solve.  Production code must use :func:`build_radiation_field_kernel_from_native`.

    Build matching pressure and elevation memory kernels from native field transfers.

    ``kernel`` is the generalized :class:`RadiationKernel` of the SAME posed family (its settings, window,
    tails and quadrature are reused; settings that this v1 cannot replicate for fields are refused).
    ``pressure_transfer``/``elevation_transfer`` are complex phasor fields per unit rigid displacement at
    ``omegas`` (exactly the kernel's source frequencies); ``*_inf`` are the native ``omega = inf`` coefficients
    ``P_a`` (never extrapolated).  Reconstruction of BOTH fields, the signed projection checks
    (``Proj P_a = -A_inf``, ``Proj P_v = -K_r``) and all limits must pass or the build refuses.
    """
    _require(isinstance(kernel, RadiationKernel), "kernel must be a RadiationKernel")
    st = kernel.settings
    _require(kernel.n_dof == N_DOF, f"field kernels require the six rigid DOFs (kernel has {kernel.n_dof})")
    _require(st.high_tail_policy == HIGH_TAIL_TRUNCATE and st.a_inf_policy == A_INF_COMPUTED
             and st.kernel_symmetrisation == KERNEL_SYMMETRISATION_NONE,
             "v1 field kernels support only truncated high tails, computed A_inf and no kernel symmetrisation "
             "(the matching global kernel must use the same policies)")
    lim = limits or FieldResourceLimits()
    _require(isinstance(lim, FieldResourceLimits), "limits must be a FieldResourceLimits")
    # machine binding fields are RESERVED: a caller cannot make raw arrays claim an admitted-native binding; only the
    # admitted wrapper (after its gate) passes ``_native_binding``.  The binding (or the explicit 'unbound' marker)
    # becomes part of the hashed identity.  Source binding is NOT proof of finite-frequency transfer provenance.
    ident = dict(identity or {})
    clash = sorted(set(ident) & RESERVED_IDENTITY_KEYS)
    _require(not clash, f"identity keys {clash} are reserved for the admitted-native binding")
    if _native_binding is None:
        ident["binding"] = "unbound_raw_arrays"
    else:
        b = dict(_native_binding)
        _require(_CORE_BINDING_KEYS <= set(b) <= RESERVED_IDENTITY_KEYS, "invalid internal native binding record")
        ident.update(b)
    if _development:
        ident["development"] = "unverified_hydrodynamics"
    _require(isinstance(chunk_size, (int, np.integer)) and not isinstance(chunk_size, bool) and chunk_size >= 1,
             "chunk_size must be a positive integer")
    # the source projection tolerance may only tighten the existing 1e-8 gate
    _require(isinstance(projection_rel_tol, (int, float)) and not isinstance(projection_rel_tol, bool)
             and np.isfinite(projection_rel_tol) and 0.0 < projection_rel_tol <= PROJECTION_REL_TOL,
             f"projection_rel_tol must lie in (0, {PROJECTION_REL_TOL:g}]; the source gate can only tighten")
    pids = tuple(str(p) for p in panel_ids)
    qids = tuple(str(q) for q in query_point_ids)
    n_p, n_e = len(pids), len(qids)
    _require(0 < n_p <= lim.max_panels, f"{n_p} panels outside (0, {lim.max_panels}]", RadiationFieldLimitError)
    _require(0 < n_e <= lim.max_query_points, f"{n_e} query points outside (0, {lim.max_query_points}]",
             RadiationFieldLimitError)
    n_t = int(st.n_time)
    stored = n_t * (n_p + n_e) * N_DOF * 8
    # the validated artifact holds a second (immutable) copy of every chunk while the first is released
    _require(2 * stored <= lim.max_artifact_bytes,
             f"field kernel storage {stored} B (n_time={n_t}, {n_p}+{n_e} fields x 6; peak 2x while freezing) "
             f"exceeds the artifact ceiling {lim.max_artifact_bytes} B", RadiationFieldLimitError)
    n_w_in = int(np.asarray(omegas).size)
    ch = min(int(chunk_size), max(n_p, n_e))
    work = 8 * (
        2 * n_w_in * n_t                       # cos / sin matrices of the reconstruction check
        + 8 * 512 * n_w_in                     # blocked cosine-integral temporaries
        + 4 * n_t * ch * N_DOF                 # kernel chunk, weighted copy, frozen copy
        + 6 * n_w_in * ch * N_DOF              # extended C(w), reconstructed C/S/H chunks (H complex = 2 doubles)
        + 2 * n_w_in * (n_p + n_e) * N_DOF     # private float copies of the native transfers' imaginary parts
    ) + 16 * n_w_in * (n_p + n_e) * N_DOF      # private complex copies of the native transfers
    _require(work <= lim.max_work_bytes,
             f"field kernel work arrays {work} B exceed the working-memory ceiling {lim.max_work_bytes} B; "
             "reduce chunk_size", RadiationFieldLimitError)

    w = np.array(_real_array(omegas, "omegas"), dtype=np.float64, copy=True)
    _require(w.ndim == 1 and w.size >= 3 and bool(np.all(np.isfinite(w))) and bool(np.all(np.diff(w) > 0.0)) and w[0] > 0.0,
             "omegas must be a strictly increasing positive 1-D grid")
    grid = np.asarray(kernel.diagnostics.frequency_grid)
    _require(grid.size == w.size + 1 and bool(np.array_equal(grid[1:], w)) and grid[0] == 0.0,
             "omegas must equal the generalized kernel's source frequencies exactly (same posed family)")

    explicit_zero = st.low_tail_policy == LOW_TAIL_EXPLICIT_ZERO
    hp, ap, zp = _check_transfer("pressure", w, pressure_transfer, pressure_inf, pressure_zero_frequency, n_p, explicit_zero)
    he, ae, ze = _check_transfer("elevation", w, elevation_transfer, elevation_inf, elevation_zero_frequency, n_e, explicit_zero)
    proj = np.array(_real_array(projection, "projection"), dtype=np.float64, copy=True)
    _require(proj.shape == (N_DOF, n_p) and bool(np.all(np.isfinite(proj))), f"projection must be finite {(N_DOF, n_p)}")

    t = np.arange(st.n_time, dtype=np.float64) * st.dt
    omega_ext = np.concatenate(([0.0], w))
    chunks_out = {}
    proj_k = np.zeros((st.n_time, N_DOF, N_DOF))
    for name, h, zero, n_f in (("pressure", hp, zp, n_p), ("elevation", he, ze, n_e)):
        c_native = -h.imag / w[:, None, None]                    # C(w) = -Im H / w
        chunks = []
        for s, e in _chunk_bounds(n_f, int(chunk_size)):
            c_ext = np.concatenate((zero[None, s:e], c_native[:, s:e]), axis=0)
            ck = _field_kernel_chunk(st, omega_ext, c_ext, t)
            chunks.append(ck)
            if name == "pressure":
                proj_k += np.einsum("ip,tpj->tij", proj[:, s:e], ck)
        chunks_out[name] = tuple(chunks)

    # signed projection of the physical pressure kernels: -A_inf and -K_r (source tolerance)
    a_inf = np.asarray(kernel.added_mass_infinite)
    k_ref = np.asarray(kernel.kernel)
    a_scale = max(float(np.max(np.abs(a_inf))), np.finfo(float).tiny)
    k_scale = max(float(np.max(np.abs(k_ref))), np.finfo(float).tiny)
    err_a = float(np.max(np.abs(proj @ ap + a_inf)))
    err_k = float(np.max(np.abs(proj_k + k_ref)))
    _require(err_a <= projection_rel_tol * a_scale,
             f"signed projection of P_a differs from -A_inf by {err_a:.3e} (> {projection_rel_tol:g} x {a_scale:.3e})")
    _require(err_k <= projection_rel_tol * k_scale,
             f"signed projection of P_v differs from -K_r by {err_k:.3e} (> {projection_rel_tol:g} x {k_scale:.3e})")
    if explicit_zero:
        b0 = np.asarray(kernel.diagnostics.zero_frequency_damping)
        b_scale = max(float(np.max(np.abs(b0))), np.finfo(float).tiny)
        err_b = float(np.max(np.abs(proj @ zp + b0)))
        _require(err_b <= projection_rel_tol * b_scale, f"projection of C(0) differs from -B(0) by {err_b:.3e}")

    result = RadiationFieldKernel(
        kernel_hash=kernel.content_hash(),
        settings=st.to_dict(),
        dof_keys=kernel.dof_keys,
        time=t,
        panel_ids=pids,
        query_point_ids=qids,
        pressure_a=ap,
        pressure_chunks=chunks_out["pressure"],
        elevation_a=ae,
        elevation_chunks=chunks_out["elevation"],
        projection=proj,
        added_mass_infinite=a_inf,
        diagnostics={},
        identity=ident,
        chunk_size=int(chunk_size),
        kernel_digest=numeric_array_digest(kernel.kernel),
        ainf_digest=numeric_array_digest(kernel.added_mass_infinite),
    )
    reports = {
        "pressure": result.reconstruction_report("pressure", w, hp, pressure_criteria),
        "elevation": result.reconstruction_report("elevation", w, he, elevation_criteria),
    }
    unpassed = []
    for name in ("pressure", "elevation"):
        if not reports[name]["passed"]:
            if not _development:
                label = "panel-pressure" if name == "pressure" else "waterline-elevation"
                _require(False, f"{label} transfer reconstruction failed: {reports[name]}")
            unpassed.append(f"{name}_transfer_reconstruction")
    diagnostics = {
        "unpassed_gates": unpassed, "development": bool(_development),
        "projection_a_inf_error": err_a, "projection_kernel_error": err_k,
        "projection_rel_tol": float(projection_rel_tol), "reconstruction": reports,
        "limits": lim.to_dict(), "stored_bytes": int(result.stored_bytes),
        "a_inf_source": "native_symbolic_omega_inf (caller supplied pressure_inf / elevation_inf)",
    }
    # one frozen copy of the chunk data only: attach the diagnostics to the already-validated artifact
    object.__setattr__(result, "diagnostics", _freeze_meta(json.loads(_canonical(diagnostics))))
    return result


def build_radiation_field_kernel_from_native(
    native: "NativeFieldCoefficients",
    *,
    kernel: RadiationKernel,
    omegas: Any,
    pressure_transfer: Any,
    elevation_transfer: Any,
    pressure_criteria: FieldReconstructionCriteria,
    elevation_criteria: FieldReconstructionCriteria,
    pressure_zero_frequency: Any = None,
    elevation_zero_frequency: Any = None,
    projection_rel_tol: float = PROJECTION_REL_TOL,
    limits: Optional[FieldResourceLimits] = None,
    chunk_size: int = 64,
    identity: Optional[Mapping[str, Any]] = None,
) -> RadiationFieldKernel:
    """Admitted-native binding of the builder: the infinite-frequency coefficients, projection and ids come ONLY from
    an admitted :class:`NativeFieldCoefficients` (boundary status recomputed ``'admitted'``); a failed or unchecked
    object raises :class:`NativeBoundaryError`.  The native content hash and admission status are written into the
    field-kernel identity.  This is a source-level binding only: it does not make the finite-frequency transfers
    (which remain caller-supplied) native-verified, and no production capability is claimed."""
    _require(isinstance(native, NativeFieldCoefficients), "native must be a NativeFieldCoefficients")
    native.require_effective_admitted()      # strict: raw gate; exact_dirichlet_z0: recomputed policy evaluation
    ident = dict(identity or {})
    binding = native_binding_record(native, "admitted_native_source")
    return build_radiation_field_kernel(_native_binding=binding,
        kernel=kernel, omegas=omegas, pressure_transfer=pressure_transfer, elevation_transfer=elevation_transfer,
        pressure_zero_frequency=pressure_zero_frequency, elevation_zero_frequency=elevation_zero_frequency,
        pressure_criteria=pressure_criteria, elevation_criteria=elevation_criteria,
        projection_rel_tol=projection_rel_tol, limits=limits, chunk_size=chunk_size, identity=ident,
        **native.effective_builder_arrays())


def native_binding_record(native: "NativeFieldCoefficients", binding: str, **extra: Any) -> dict:
    """Machine binding fields written into the HASHED field-kernel identity by production paths only (after their
    admission gate): native content hash, RAW boundary status, selected policy and its recomputed status, and the
    digests of the raw E_inf and of the pointwise bound (raw values stay in the native artifact)."""
    bound = native.pointwise_bound
    rec = dict(binding=binding, native_hash=native.content_hash(), native_boundary_status=native.boundary_status,
               native_helper=NATIVE_FIELD_HELPER_VERSION, boundary_policy=native.boundary_policy,
               policy_status=native.policy_status, raw_elevation_digest=numeric_array_digest(native.elevation_inf),
               pointwise_bound_digest="" if bound is None else numeric_array_digest(bound),
               bound_algorithm="" if bound is None else POINTWISE_BOUND_ALGORITHM)
    _require(set(extra) <= RESERVED_IDENTITY_KEYS - set(rec), "invalid extra binding keys")
    rec.update(extra)
    return rec


# ----------------------------------------------------------------------------- offline evaluation

def _check_history(fk: RadiationFieldKernel, accelerations: Any, velocities: Any):
    a = _real_array(accelerations, "accelerations")
    v = _real_array(velocities, "velocities")
    _require(a.ndim == 2 and a.shape[1] == N_DOF and a.shape == v.shape, "accelerations / velocities must be (n, 6)")
    _require(bool(np.all(np.isfinite(a))) and bool(np.all(np.isfinite(v))), "non-finite trajectory")
    return a, v


def _admit_history(fk: RadiationFieldKernel, kind: str, n_s: int, limits: FieldResourceLimits, chunk: int,
                   block: Optional[int], collect: bool) -> int:
    a, _ = fk._parts(kind)
    n_f = a.shape[0]
    if collect:
        need = n_s * n_f * 8
        _require(need <= limits.max_work_bytes, f"accepted field history {need} B exceeds the working-memory limit "
                 f"{limits.max_work_bytes} B; iterate in chunks", RadiationFieldLimitError)
    blk = n_s if block is None else int(block)
    _require(blk >= 1, "block_length must be a positive integer")
    nfft = 1 << int(np.ceil(np.log2(max(2, blk + fk.n_time - 1))))
    chunk = min(int(chunk), n_f) if n_f else int(chunk)
    # rFFT of the kernel chunk, product, inverse transform, velocity spectrum, outputs and the per-chunk
    # zero-padded endpoint correction (all counted before the first allocation)
    work = 8 * (nfft * chunk * (N_DOF + 2) + nfft * N_DOF + n_s * chunk * (3 + 2 * N_DOF))
    _require(work <= limits.max_work_bytes,
             f"FFT work arrays {work} B exceed the working-memory limit {limits.max_work_bytes} B; "
             "reduce chunk or block_length", RadiationFieldLimitError)
    return nfft


def field_history_direct(fk: RadiationFieldKernel, kind: str, accelerations: Any, velocities: Any,
                         limits: Optional[FieldResourceLimits] = None) -> np.ndarray:
    """Independent direct-sum oracle: F_n = P_a a_n + dt * trapezoid_j(P_v[n-j] v_j), shape (n, n_field)."""
    lim = limits or FieldResourceLimits()
    acc, vel = _check_history(fk, accelerations, velocities)
    n_s = acc.shape[0]
    a0, _ = fk._parts(kind)
    # dense kernel + the per-step gathered copy + weights/products, before the dense kernel is built
    dense_need = 3 * fk.n_time * a0.shape[0] * N_DOF * 8 + n_s * a0.shape[0] * 8
    _require(dense_need <= lim.max_work_bytes,
             f"direct oracle needs about {dense_need} B > working-memory limit {lim.max_work_bytes} B",
             RadiationFieldLimitError)
    kern = fk.kernel_dense(kind, lim)
    dt = fk.dt
    out = acc @ a0.T
    _require(n_s <= fk.n_time or float(np.max(np.abs(kern[-1]))) == 0.0,
             "history longer than the kernel requires an exactly-zero windowed endpoint")
    for n in range(n_s):
        m_max = min(n, fk.n_time - 1)
        if n == 0:
            continue
        lags = np.arange(0, m_max + 1)
        js = n - lags
        weights = np.ones(lags.size)
        # trapezoid weights on j = 0 (oldest) and j = n (current)
        weights[js == 0] *= 0.5
        weights[lags == 0] *= 0.5
        out[n] += dt * np.einsum("m,mfj,mj->f", weights, kern[lags], vel[js])
    return out


def _fft_conv_chunk(kern: np.ndarray, vel: np.ndarray, nfft_block: Optional[int]) -> np.ndarray:
    """Linear convolution y[n] = sum_{j<=n} K[n-j] v[j] per (field, dof), summed over dof: (n_s, n_chunk).

    Zero-padded rFFT; when ``nfft_block`` is given the velocity is cut into blocks and overlap-added.
    """
    n_k, n_c, _ = kern.shape
    n_s = vel.shape[0]
    block = n_s if nfft_block is None else int(nfft_block)
    out = np.zeros((n_s, n_c))
    size = 1 << int(np.ceil(np.log2(max(2, block + n_k - 1))))
    kf = np.fft.rfft(kern, n=size, axis=0)                       # (size/2+1, n_c, 6)
    for start in range(0, n_s, block):
        vb = vel[start:start + block]
        vf = np.fft.rfft(vb, n=size, axis=0)                     # (.., 6)
        y = np.fft.irfft(np.einsum("fcj,fj->fc", kf, vf), n=size, axis=0)
        seg = y[: min(vb.shape[0] + n_k - 1, n_s - start)]
        out[start:start + seg.shape[0]] += seg
    return out


def iter_field_history(fk: RadiationFieldKernel, kind: str, accelerations: Any, velocities: Any, *,
                       block_length: Optional[int] = None, limits: Optional[FieldResourceLimits] = None
                       ) -> Iterator[Tuple[slice, np.ndarray]]:
    """Yield ``(field_slice, history[n_s, n_chunk])`` using zero-padded block FFT convolution per chunk.

    Implements exactly the discrete trapezoidal operator of the online global memory:
    ``dt (sum_{j=0}^{n} K[n-j] v_j - 0.5 K[n] v_0 - 0.5 K[0] v_n) + P_a a_n``.
    Memory is checked before the first allocation.
    """
    lim = limits or FieldResourceLimits()
    acc, vel = _check_history(fk, accelerations, velocities)
    n_s = acc.shape[0]
    _require(n_s >= 1, "empty trajectory")
    a0, chunks = fk._parts(kind)
    _admit_history(fk, kind, n_s, lim, fk.chunk_size, block_length, False)
    if n_s > fk.n_time:
        _require(all(float(np.max(np.abs(c[-1]))) == 0.0 for c in chunks),
                 "history longer than the kernel requires an exactly-zero windowed endpoint")
    dt = fk.dt
    for (s, e), c in zip(_chunk_bounds(a0.shape[0], fk.chunk_size), chunks):
        y = _fft_conv_chunk(c, vel, block_length)
        # remove the end-point halves of the trapezoid: 0.5 K[n] v_0 (when n < n_time) and 0.5 K[0] v_n
        k_n = np.zeros((n_s, e - s, N_DOF))
        m = min(n_s, fk.n_time)
        k_n[:m] = c[:m]
        y -= 0.5 * np.einsum("nfj,j->nf", k_n, vel[0])
        y -= 0.5 * np.einsum("fj,nj->nf", c[0], vel)
        out = dt * y + acc @ a0[s:e].T
        out[0] = acc[0] @ a0[s:e].T              # memory_0 = 0 exactly
        yield slice(s, e), out


def field_history_fft(fk: RadiationFieldKernel, kind: str, accelerations: Any, velocities: Any, *,
                      block_length: Optional[int] = None, limits: Optional[FieldResourceLimits] = None) -> np.ndarray:
    """Collect :func:`iter_field_history` into ``(n_s, n_field)``; refuses above the working-memory limit."""
    lim = limits or FieldResourceLimits()
    acc, _ = _check_history(fk, accelerations, velocities)
    a0, _ = fk._parts(kind)
    _admit_history(fk, kind, acc.shape[0], lim, fk.chunk_size, block_length, True)
    out = np.empty((acc.shape[0], a0.shape[0]))
    for sl, part in iter_field_history(fk, kind, accelerations, velocities, block_length=block_length, limits=lim):
        out[:, sl] = part
    return out


SUPPORTED_RESULT_FRAME = "body_attached_first_order"
_GRID_TOL = 1.0e-9


def field_history_from_transient(fk: RadiationFieldKernel, result: Any, kind: str, *, method: str = "fft",
                                 block_length: Optional[int] = None,
                                 limits: Optional[FieldResourceLimits] = None) -> np.ndarray:
    """BOUND field evaluation for a transient result (duck-typed: no ANYsolver import).

    Everything is checked BEFORE any evaluation: the result must be complete (``status == 'complete'``), carry the
    supported frame, and its computed ``kernel_digest`` / ``ainf_digest`` (``numeric-array-v1`` of the sampled
    kernel and A_inf actually used by the run) must equal the digests this field kernel was built from; ``dt``
    must agree to the existing grid tolerance.  A field kernel without digests (unbound fixture) or a result
    from a different kernel / A_inf is refused.  The raw ``field_history_direct`` / ``field_history_fft`` /
    ``iter_field_history`` functions remain UNBOUND development primitives.
    """
    _require(method in ("fft", "direct"), "method must be 'fft' or 'direct'")
    for attr in ("status", "complete", "frame", "kernel_digest", "ainf_digest", "digest_protocol", "dt",
                 "accelerations", "velocities", "times", "n_accepted"):
        _require(hasattr(result, attr), f"result lacks {attr!r}: not a bound transient result")
    _require(result.status == "complete" and bool(result.complete),
             f"transient status {result.status!r} is not complete; non-success histories are never evaluated")
    _require(result.frame == SUPPORTED_RESULT_FRAME,
             f"unsupported frame {result.frame!r}; only {SUPPORTED_RESULT_FRAME!r} is evaluated")
    _require(result.digest_protocol == DIGEST_PROTOCOL == "numeric-array-v1", "unsupported digest protocol")
    _require(bool(fk.kernel_digest) and bool(fk.ainf_digest),
             "this field kernel is UNBOUND (no generalized-kernel / A_inf digests): refused by the bound entry point")
    _require(result.kernel_digest == fk.kernel_digest,
             "the transient result was computed with a DIFFERENT radiation kernel than this field kernel")
    _require(result.ainf_digest == fk.ainf_digest,
             "the transient result was computed with a DIFFERENT A_inf than this field kernel")
    # structure: derive dt from the ACTUAL real times and require every array to agree with the accepted count
    times = _real_array(result.times, "times")
    acc, vel = _check_history(fk, result.accelerations, result.velocities)
    n = times.size
    _require(times.ndim == 1 and n >= 2 and bool(np.all(np.isfinite(times))),
             "result.times must be a finite 1-D vector with at least two samples")
    try:                                # None / NaN / inf / non-numeric counts become typed refusals
        count_ok = (isinstance(result.n_accepted, (int, np.integer, float, np.floating))
                    and not isinstance(result.n_accepted, (bool, np.bool_))
                    and float(result.n_accepted) == int(result.n_accepted)
                    and int(result.n_accepted) == n)
    except (TypeError, ValueError, OverflowError):
        count_ok = False
    _require(count_ok and acc.shape[0] == n and vel.shape[0] == n,
             f"inconsistent accepted history: n_accepted={result.n_accepted!r}, times={n}, "
             f"accelerations={acc.shape[0]}, velocities={vel.shape[0]}")
    steps = np.diff(times)
    dt_actual = float(steps[0])
    _require(dt_actual > 0.0 and float(np.max(np.abs(steps - dt_actual))) <= _GRID_TOL * dt_actual,
             "result.times are not a strictly increasing uniform grid")
    try:
        declared = float(result.dt)
    except (TypeError, ValueError, OverflowError):
        declared = float("nan")
    _require(np.isfinite(declared) and abs(declared - dt_actual) <= _GRID_TOL * dt_actual,
             f"declared result dt {declared!r} differs from the dt {dt_actual!r} derived from result.times")
    _require(abs(dt_actual - fk.dt) <= _GRID_TOL * fk.dt,
             f"result time step {dt_actual!r} differs from the field-kernel dt {fk.dt!r}")
    # NOTE: the digest comparison above is a CONSISTENCY check between two numeric fingerprints; it is not a
    # provenance proof (a caller able to forge a result can copy a digest).
    if method == "direct":
        return field_history_direct(fk, kind, result.accelerations, result.velocities, limits)
    return field_history_fft(fk, kind, result.accelerations, result.velocities, block_length=block_length, limits=limits)


# ----------------------------------------------------------------------------- native omega = infinity helper

_DOF_NAMES = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")
_CORE_BINDING_KEYS = frozenset({"binding", "native_hash", "native_boundary_status", "native_helper"})
RESERVED_IDENTITY_KEYS = _CORE_BINDING_KEYS | frozenset({
    "boundary_policy", "policy_status", "pointwise_bound_digest", "raw_elevation_digest", "source_snapshot_ids",
    "family_radiation_key", "bound_algorithm", "development"})
NATIVE_FIELD_HELPER_VERSION = "native_inf_frequency_field_v2"
NATIVE_COEFFICIENTS_FORMAT = "anyhydro-native-inf-field-coefficients-v2"
_NATIVE_ARRAYS = ("pressure_inf", "elevation_inf", "added_mass_infinite", "projection", "centers", "normals", "areas",
                  "query_points", "vertices", "faces", "hull_mask", "sources_over_omega", "potential_over_omega",
                  "prony", "boundary_residual", "boundary_limit", "abs_s_query")
#: explicit boundary policies.  native_strict (DEFAULT): the raw 4096 eps gate decides.  exact_dirichlet_z0
#: (user-approved, conditional on verification): at query points with z EXACTLY 0 the effective instantaneous
#: radiated-elevation coefficient is the continuum Dirichlet limit 0, provided the raw value is consistent with the
#: pointwise triangle bound; raw values, raw status and bound are retained.  Never applied at z != 0.
BOUNDARY_POLICY_NATIVE_STRICT = "native_strict"
BOUNDARY_POLICY_EXACT_DIRICHLET_Z0 = "exact_dirichlet_z0"
BOUNDARY_POLICIES = (BOUNDARY_POLICY_NATIVE_STRICT, BOUNDARY_POLICY_EXACT_DIRICHLET_Z0)


#: identity of the pointwise-bound ALGORITHM; recorded in the hashed metadata of every artifact that carries |S(z0)|,
#: so one source hash can never denote two different numerical bounds.  v1 (BLAS matmul, layout/alignment dependent
#: summation) is superseded and refused on load.
POINTWISE_BOUND_ALGORITHM = "pointwise-bound-fsum-hypot-v2"


def _source_magnitudes(sources_over_omega: Any) -> np.ndarray:
    """|sigma_jp| (6, n_src) with math.hypot per element (scalar libm call; no SIMD-path dependence)."""
    s = np.asarray(sources_over_omega, dtype=np.complex128)
    out = np.empty(s.shape, dtype=np.float64)
    re, im = s.real.tolist(), s.imag.tolist()
    for j in range(s.shape[0]):
        out[j] = [math.hypot(x, y) for x, y in zip(re[j], im[j])]
    return out


def _abs_influence(S: np.ndarray) -> np.ndarray:
    """|S| deterministically: exact |x| for real S; per-element math.hypot for complex S."""
    if S.dtype.kind != "c":
        return np.abs(S).astype(np.float64)
    out = np.empty(S.shape, dtype=np.float64)
    re, im = S.real.tolist(), S.imag.tolist()
    for i in range(S.shape[0]):
        out[i] = [math.hypot(x, y) for x, y in zip(re[i], im[i])]
    return out


def _pointwise_bound(abs_s: np.ndarray, sources_over_omega: np.ndarray, gravity: float) -> np.ndarray:
    """B (n_query, 6) = (1 + gamma_m) * [sum_p |S_ip| |sigma_jp|] / g with the declared floating-point margin
    gamma_m = m eps / (1 - m eps), m = n_src + 4.  Real, nonnegative; independent of the signed potential.

    Deterministic across memory layout, alignment, BLAS and process (algorithm POINTWISE_BOUND_ALGORITHM): every
    product |S_ip| |sigma_jp| is one IEEE multiplication (exact rounding, order-free), the sum over p is math.fsum
    (correctly rounded, order independent) and the scaling is a fixed scalar sequence ``(factor * sum) / g``.  The
    correctly rounded sum is within the same gamma_m margin as any recursive summation, so the rigorous triangle
    bound is preserved.  Memory: one (6, n_src) product row per query point, no n_q * n_src * 6 temporary."""
    a = np.ascontiguousarray(np.asarray(abs_s, dtype=np.float64))
    mags = _source_magnitudes(sources_over_omega)
    n_q, n_src = a.shape
    m = float(n_src + 4) * np.finfo(float).eps
    factor = 1.0 + m / (1.0 - m)
    g = float(gravity)
    out = np.empty((n_q, mags.shape[0]), dtype=np.float64)
    for i in range(n_q):
        prod = mags * a[i][None, :]                 # elementwise IEEE products, (6, n_src)
        for j in range(mags.shape[0]):
            out[i, j] = (factor * math.fsum(prod[j].tolist())) / g
    return out


def _frozen_any(value: Any, dtype, name: str) -> np.ndarray:
    """Defensive copy with a given dtype, backed by an immutable bytes buffer (``setflags(write=True)`` fails)."""
    if np.dtype(dtype).kind != "c":
        value = _real_array_any(value, name)
    try:
        arr = np.array(value, dtype=dtype, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise RadiationFieldError(f"{name}: not a valid {np.dtype(dtype)} array ({exc})") from exc
    if arr.size == 0:
        arr.setflags(write=False)
        return arr
    return np.frombuffer(arr.tobytes(), dtype=arr.dtype).reshape(arr.shape)


def _plain_complex(value: Any, name: str) -> np.ndarray:
    """Finite plain COMPLEX array (both parts preserved) from a symbolically normalised quantity."""
    if hasattr(value, "symbol"):
        raise RadiationFieldError(
            f"{name}: a symbolic {value.symbol!r} factor remained after normalisation; no numeric infinity is substituted")
    raw = np.array(value, copy=True)
    _require(raw.dtype != object, f"{name}: object array of symbolic values (not a finite coefficient)")
    arr = np.array(raw, dtype=np.complex128)
    _require(bool(np.all(np.isfinite(arr))), f"{name}: non-finite coefficient")
    return arr


def _waterline_coefficient(enc: Any, potential: Any, gravity: float, name: str) -> np.ndarray:
    """E_a = -Re(eta)/omega^2 with eta = i*omega*potential/g, normalised on the symbolic objects first."""
    eta = 1j * enc * potential / gravity
    return _plain_limit(-np.real(eta) / (enc * enc), name)


def _plain_limit(value: Any, name: str) -> np.ndarray:
    """Convert a symbolically normalised Capytaine quantity to a finite plain real array.

    ``-Re(p) / encounter_omega**2`` (and the elevation analogue) must have been formed on the symbolic
    objects BEFORE this call so the infinite frequency factors cancel.  A remaining symbolic wrapper means
    the limit is not a finite coefficient and is refused (a numeric infinity is never multiplied through).
    """
    if hasattr(value, "symbol"):
        raise RadiationFieldError(
            f"{name}: a symbolic {value.symbol!r} factor remained after normalisation; the omega = inf limit "
            "is not a finite coefficient and no numeric infinity is substituted")
    arr = np.array(value, copy=True)
    _require(arr.dtype != object, f"{name}: object array of symbolic values (not a finite coefficient)")
    if arr.dtype.kind == "c":
        # A normalised physical coefficient must be real.  Exactly-zero imaginary parts (e.g. a complex dtype produced
        # by an exact real expression) are dropped; ANY nonzero imaginary component is refused, never discarded.
        _require(bool(np.all(arr.imag == 0.0)) and bool(np.all(np.isfinite(arr.real))),
                 f"{name}: nonzero or non-finite imaginary component in a coefficient that must be real")
        arr = arr.real
    arr = np.asarray(arr, dtype=np.float64)
    _require(bool(np.all(np.isfinite(arr))), f"{name}: non-finite coefficient")
    return arr


def physical_force_projection(centers: Any, normals: Any, areas: Any, Q0: Any, cg_world: Any) -> np.ndarray:
    """Signed physical-pressure projection (6, n_panel) in mean-body generalized coordinates.

    Force on the body from pressure ``p`` is ``-p n dS`` with ``n`` the outward (hull-into-fluid) unit normal;
    the generalized force of DOF ``j`` is its work-conjugate ``-p (n . u_j(x)) dS`` with ``u_j`` the rigid
    displacement field (translation: ``Q0[:, j]``; rotation: ``Q0[:, j] x (x - cg)``).  Hence
    ``Proj @ P_a = -A_inf`` and ``Proj @ P_v(t) = -K_r(t)`` for physical pressure kernels.
    """
    c = _real_array(centers, "centers")
    n = _real_array(normals, "normals")
    a = _real_array(areas, "areas")
    Q = _real_array(Q0, "Q0")
    cg = _real_array(cg_world, "cg_world").reshape(3)
    _require(c.ndim == 2 and c.shape[1] == 3 and n.shape == c.shape and a.shape == (c.shape[0],),
             "centers / normals / areas must be (n, 3), (n, 3), (n,)")
    nn = np.linalg.norm(n, axis=1)
    _require(bool(np.all(nn > 0.0)), "zero-length panel normal")
    n = n / nn[:, None]
    proj = np.empty((N_DOF, c.shape[0]))
    for j in range(3):
        proj[j] = -a * (n @ Q[:, j])
    for j in range(3):
        u = np.cross(Q[:, j][None, :], c - cg[None, :])
        proj[3 + j] = -a * np.einsum("pi,pi->p", n, u)
    return proj


def _derived_boundary_limit(vertices: Any, request: Any) -> Optional[np.ndarray]:
    """The immutable declared ceiling 4096 eps * [L/g, L/g, L/g, L R/g, L R/g, L R/g] derived ONLY from the retained
    vertices (L = largest bounding-box extent, R = max vertex distance from the CG) and the retained request
    (cg_world, gravity).  Returns None when the geometry / environment is incomplete or invalid (status then stays
    'unchecked').  This is the single definition used by the helper, the constructor and the recovery check."""
    try:
        v = np.asarray(vertices, dtype=np.float64)
        cg = np.asarray(request["cg_world"], dtype=np.float64).reshape(3)
        g = float(request["gravity"])
    except (TypeError, ValueError, KeyError):
        return None
    if (v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 1 or not np.all(np.isfinite(v)) or not np.all(np.isfinite(cg))
            or not np.isfinite(g) or g <= 0.0):
        return None
    extent = float(np.max(np.ptp(v, axis=0)))
    reach = float(np.max(np.linalg.norm(v - cg.reshape(1, 3), axis=1)))
    unit = 4096.0 * np.finfo(float).eps
    return np.array([unit * extent / g] * 3 + [unit * extent * reach / g] * 3)


@dataclass(frozen=True)
class NativeFieldCoefficients:
    """Named immutable infinite-frequency field coefficients from ONE posed native family.

    ``pressure_inf`` (n_panel, 6) and ``elevation_inf`` (n_query, 6) are the finite normalised coefficients
    ``P_a = -Re(p)/w^2`` and ``E_a = -Re(eta)/w^2`` at ``omega = inf`` (per unit rigid displacement, physical
    sign); ``added_mass_infinite`` is the native ``A_inf`` of the SAME six solves.  They feed
    :func:`build_radiation_field_kernel` (``pressure_inf``, ``elevation_inf``, ``projection``, ids).
    Always ``unqualified``.
    """

    pressure_inf: np.ndarray
    elevation_inf: np.ndarray
    added_mass_infinite: np.ndarray
    projection: np.ndarray
    centers: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    query_points: np.ndarray
    panel_ids: Tuple[str, ...]
    query_point_ids: Tuple[str, ...]
    dof_names: Tuple[str, ...]
    projection_error: float
    provenance: Mapping[str, Any]
    # ---- complete recoverable source of the SAME six solves (see recover_waterline_coefficients)
    vertices: Any = None                  # (nv, 3) manifest vertices
    faces: Any = None                     # (nf, 4) int64 manifest faces
    hull_mask: Any = None                 # (nf,) bool physical-hull mask
    surface_ids: Tuple[str, ...] = ()     # one label per face
    sources_over_omega: Any = None        # (6, n_src) COMPLEX sources / omega (symbolic limit taken first)
    potential_over_omega: Any = None      # (6, n_pot) COMPLEX potential / omega
    request: Any = None                   # family request (Q0, cg, environment, geometry hash)
    provider: Any = None                  # backend, version, module path, method, engine, green_function
    gf_settings: Any = None               # exported Green-function constructor settings
    prony: Any = None                     # (2, m) realised finite-depth decomposition actually used
    prony_kh: float = 0.0                 # kh of that decomposition (inf at omega = inf; stored as the string 'inf')
    prony_method: str = ""
    reference_omega: Any = None           # finite frequency used only to prepare the body
    boundary_residual: Any = None         # (6,) max |E_a| per DOF actually observed on the waterline points
    boundary_limit: Any = None            # (6,) declared roundoff scale 4096 eps [L/g, L R/g]
    boundary_status: str = ""             # 'admitted' | 'failed' | 'unchecked'; hashed, recomputed and verified on load
    # ---- explicit boundary POLICY (user-approved, conditional on verification); the raw gate above is unchanged
    boundary_policy: str = BOUNDARY_POLICY_NATIVE_STRICT
    abs_s_query: Any = None               # (n_query, n_src) |S(z0)| of the SAME captured GF at the actual points
    policy_status: str = ""               # recomputed: 'admitted' | 'refused' (exact_dirichlet_z0) or raw status (strict)

    def __post_init__(self) -> None:
        for name, ndim in (("pressure_inf", 2), ("elevation_inf", 2), ("added_mass_infinite", 2), ("projection", 2),
                           ("centers", 2), ("normals", 2), ("areas", 1), ("query_points", 2)):
            object.__setattr__(self, name, _frozen_float_array(getattr(self, name), name, ndim=ndim))
        object.__setattr__(self, "panel_ids", tuple(str(p) for p in self.panel_ids))
        object.__setattr__(self, "query_point_ids", tuple(str(p) for p in self.query_point_ids))
        object.__setattr__(self, "dof_names", tuple(self.dof_names))
        object.__setattr__(self, "provenance", _freeze_meta(json.loads(_canonical(dict(self.provenance)))))
        n_panel, n_query = self.pressure_inf.shape[0], self.elevation_inf.shape[0]
        _require(len(set(self.panel_ids)) == len(self.panel_ids) == n_panel and
                 len(set(self.query_point_ids)) == len(self.query_point_ids) == n_query,
                 "one unique id per panel and per query point is required")
        if self.faces is not None:                # exact integer indices only: a float array is never truncated
            raw_faces = np.asarray(self.faces)
            _require(raw_faces.dtype.kind in "iu", f"faces: integer dtype required (got {raw_faces.dtype}); no truncation")
        for name, dtype, shape in (("vertices", np.float64, None), ("faces", np.int64, (n_panel, 4)),
                                   ("hull_mask", np.bool_, None), ("sources_over_omega", np.complex128, None),
                                   ("potential_over_omega", np.complex128, None), ("prony", np.float64, None),
                                   ("boundary_residual", np.float64, (N_DOF,)), ("boundary_limit", np.float64, (N_DOF,)),
                                   ("abs_s_query", np.float64, None)):
            value = getattr(self, name)
            if value is None:
                continue
            arr = _frozen_any(value, dtype, name)
            _require(shape is None or arr.shape == shape, f"{name}: expected shape {shape}, got {arr.shape}")
            if arr.dtype.kind in "fc":
                _require(bool(np.all(np.isfinite(arr))), f"{name}: non-finite values are not admitted")
            object.__setattr__(self, name, arr)
        for name in ("sources_over_omega", "potential_over_omega"):
            arr = getattr(self, name)
            _require(arr is None or (arr.ndim == 2 and arr.shape[0] == N_DOF and arr.shape[1] >= n_panel),
                     f"{name} must be (6, n >= n_panel) complex")
        if self.request is not None:
            object.__setattr__(self, "request", _freeze_meta(json.loads(_canonical(dict(self.request)))))
        if self.request is not None and 'transient_source_profile' in self.request:
            from .transient_native_profile import profile_from_request
            profile = profile_from_request(self.request)
            _require(self.hull_mask is not None and np.array_equal(self.hull_mask, profile.hull_mask),
                     "native hull mask differs from the complete transient source profile")
            _require(self.sources_over_omega is not None
                     and self.sources_over_omega.shape[1] == len(profile.source_faces),
                     "native source vector differs from the complete transient source profile")
        else:
            _require(self.hull_mask is None or self.hull_mask.shape == (n_panel,), "hull_mask must match hull panels")
        for name in ("provider", "gf_settings"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _freeze_meta(json.loads(_canonical(dict(getattr(self, name))))))
        if self.faces is not None and self.vertices is not None:
            _require(self.vertices.ndim == 2 and self.vertices.shape[1] == 3 and self.vertices.shape[0] >= 1,
                     "vertices must be a non-empty (n, 3) array")
            _require(self.faces.ndim == 2 and self.faces.shape[0] >= 1 and self.faces.shape[1] == 4,
                     "faces must be a non-empty (n, 4) array")
            _require(int(self.faces.min()) >= 0 and int(self.faces.max()) < self.vertices.shape[0],
                     "faces index outside the vertex array")
        object.__setattr__(self, "surface_ids", tuple(str(s) for s in self.surface_ids))
        _require(not self.surface_ids or len(self.surface_ids) == n_panel, "one surface id per panel is required")
        # boundary admission is a recomputed property of the raw residual and limit, never a caller assertion
        # ... and from the ACTUAL elevation coefficients: reported metadata may never contradict the raw data
        _require(n_query >= 1 and n_panel >= 1 and self.elevation_inf.shape[1] == N_DOF,
                 "at least one panel and one waterline query point (with six DOF columns) are required")
        observed = np.max(np.abs(self.elevation_inf), axis=0)
        # the limit is an IMMUTABLE ceiling derived from the retained geometry / environment; a caller may only tighten
        ceiling = _derived_boundary_limit(self.vertices, self.request)
        if self.boundary_limit is not None:
            _require(bool(np.all(self.boundary_limit >= 0.0)), "boundary_limit must be nonnegative and finite")
            if ceiling is not None:
                _require(bool(np.all(self.boundary_limit <= ceiling)),
                         "boundary_limit widens the derived ceiling 4096 eps [L/g, L R/g]; only tightening is permitted")
        if self.boundary_residual is not None:
            _require(bool(np.all(self.boundary_residual >= 0.0)), "boundary_residual must be nonnegative and finite")
            _require(bool(np.array_equal(self.boundary_residual, observed)),
                     "boundary_residual contradicts the raw elevation_inf (must equal max |E_a| per DOF)")
        if ceiling is None:
            recomputed = "unchecked"              # no complete valid geometry / environment: nothing can be admitted
        else:
            effective = ceiling if self.boundary_limit is None else self.boundary_limit
            if self.boundary_limit is None:
                object.__setattr__(self, "boundary_limit", _frozen_any(ceiling, np.float64, "boundary_limit"))
            recomputed = "admitted" if bool(np.all(observed <= effective)) else "failed"
        _require(self.boundary_status in ("", recomputed),
                 f"boundary_status {self.boundary_status!r} contradicts the raw residual / limit ({recomputed!r})")
        object.__setattr__(self, "boundary_status", recomputed)
        # ---- policy evaluation: recomputed from raw data, never asserted by the caller
        _require(self.boundary_policy in BOUNDARY_POLICIES,
                 f"boundary_policy must be one of {BOUNDARY_POLICIES} (got {self.boundary_policy!r})")
        if self.abs_s_query is not None:
            _require(self.abs_s_query.ndim == 2 and self.abs_s_query.shape[0] == n_query
                     and self.sources_over_omega is not None
                     and self.abs_s_query.shape[1] == self.sources_over_omega.shape[1]
                     and bool(np.all(self.abs_s_query >= 0.0)),
                     "abs_s_query must be a nonnegative (n_query, n_src) array matching sources_over_omega")
        pstatus, reason = self._evaluate_policy()
        # native_strict: the policy status IS the recomputed raw status (any carried value is simply recomputed);
        # an explicit policy's status must agree with its recomputed evaluation (never asserted by a caller)
        if self.boundary_policy != BOUNDARY_POLICY_NATIVE_STRICT:
            _require(self.policy_status in ("", pstatus),
                     f"policy_status {self.policy_status!r} contradicts the recomputed policy evaluation ({pstatus!r}: {reason})")
        object.__setattr__(self, "policy_status", pstatus)
        kh = float(self.prony_kh)
        _require(not np.isnan(kh), "prony_kh must not be NaN")
        object.__setattr__(self, "prony_kh", kh)
        if self.reference_omega is not None:
            object.__setattr__(self, "reference_omega", float(self.reference_omega))

    @property
    def pointwise_bound(self) -> Optional[np.ndarray]:
        """(n_query, 6) rigorous triangle bound B_ij = (1+gamma) sum_p |S(z0)_ip| |sigma_jp| / g (None if unavailable).

        Exact continuum S(z0) = 0 at omega = inf, so |raw E_ij| = |Im(S sigma_j)_i| / g <= B_ij bounds the discrete
        approximation; computed from the retained |S| and normalised sources, independently of the signed potential."""
        if self.abs_s_query is None or self.sources_over_omega is None or self.request is None:
            return None
        try:
            g = float(self.request["gravity"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (np.isfinite(g) and g > 0.0):
            return None
        return _pointwise_bound(self.abs_s_query, self.sources_over_omega, g)

    def _evaluate_policy(self) -> tuple:
        if self.boundary_policy == BOUNDARY_POLICY_NATIVE_STRICT:
            return self.boundary_status, "native_strict uses the raw 4096 eps gate"
        # exact_dirichlet_z0: ONLY exactly z == 0 query points, a complete bound and raw data consistent with it
        if not bool(np.all(self.query_points[:, 2] == 0.0)):
            return "refused", "exact_dirichlet_z0 applies only at query points with z exactly 0"
        bound = self.pointwise_bound
        if bound is None:
            return "refused", "no |S(z0)| / sources / gravity retained: the pointwise bound is unavailable"
        if not bool(np.all(np.abs(self.elevation_inf) <= bound)):
            return "refused", "raw E_inf exceeds the pointwise triangle bound (inconsistent source data)"
        return "admitted", "exact continuum Dirichlet limit at z = 0; raw values and bound retained"

    @property
    def policy_reason(self) -> str:
        return self._evaluate_policy()[1]

    @property
    def effective_admitted(self) -> bool:
        """Admission under the SELECTED policy (strict: the raw gate; exact_dirichlet_z0: the policy evaluation)."""
        return self.policy_status == "admitted"

    @property
    def effective_elevation_inf(self) -> np.ndarray:
        """Coefficient fed to field builders: exactly 0 ONLY under an admitted exact_dirichlet_z0 policy; otherwise the
        raw native E_inf.  The raw ``elevation_inf`` itself is never overwritten."""
        if self.boundary_policy == BOUNDARY_POLICY_EXACT_DIRICHLET_Z0 and self.effective_admitted:
            return _frozen_any(np.zeros_like(self.elevation_inf), np.float64, "effective_elevation_inf")
        return self.elevation_inf

    def require_effective_admitted(self) -> "NativeFieldCoefficients":
        if not self.effective_admitted:
            raise NativeBoundaryError(
                f"native coefficients are not admitted under policy {self.boundary_policy!r} (policy status "
                f"{self.policy_status!r}: {self.policy_reason}; raw boundary status {self.boundary_status!r})", self)
        return self

    def effective_builder_arrays(self) -> dict:
        """Builder arguments under the selected policy (raw P_inf / projection / ids unchanged; E_inf effective)."""
        self.require_effective_admitted()
        return dict(pressure_inf=self.pressure_inf, elevation_inf=self.effective_elevation_inf, projection=self.projection,
                    panel_ids=self.panel_ids, query_point_ids=self.query_point_ids)

    @property
    def is_admitted(self) -> bool:
        """True only when the free-surface boundary residual is within the declared scale (recomputed, hashed).
        This is the RAW native gate; it is not changed by any policy."""
        return self.boundary_status == "admitted"

    def require_admitted(self) -> "NativeFieldCoefficients":
        if not self.is_admitted:
            observed = "unknown" if self.elevation_inf is None else f"{float(np.max(np.abs(self.elevation_inf))):.3e}"
            raise NativeBoundaryError(
                f"native coefficients are not admitted (boundary status {self.boundary_status!r}; max |E_a| "
                f"{observed}); diagnostic only", self)
        return self

    @property
    def is_recoverable(self) -> bool:
        """True when the retained source suffices to evaluate new waterline points with zero solves."""
        return all(x is not None for x in (self.vertices, self.faces, self.hull_mask, self.sources_over_omega,
                                           self.request, self.provider, self.gf_settings, self.prony))

    def _meta(self) -> dict:
        """JSON-safe metadata: infinity is the explicit string 'inf'; every identity and id is included."""
        kh = self.prony_kh
        extra = {}
        if self.abs_s_query is not None:          # the bound algorithm is part of the hashed identity of |S| evidence
            extra["bound_algorithm"] = POINTWISE_BOUND_ALGORITHM
        return dict(
            **extra,
            format=NATIVE_COEFFICIENTS_FORMAT, helper=NATIVE_FIELD_HELPER_VERSION, provenance=self.provenance,
            panel_ids=list(self.panel_ids), query_point_ids=list(self.query_point_ids), dof_names=list(self.dof_names),
            surface_ids=list(self.surface_ids), request=self.request, provider=self.provider, gf_settings=self.gf_settings,
            prony_kh="inf" if np.isinf(kh) and kh > 0 else float(kh), prony_method=self.prony_method,
            reference_omega=self.reference_omega, projection_error=float(self.projection_error),
            boundary_status=self.boundary_status, boundary_policy=self.boundary_policy, policy_status=self.policy_status,
            present=[n for n in _NATIVE_ARRAYS if getattr(self, n) is not None], qualification_status="unqualified")

    @property
    def qualification_status(self) -> str:
        return "unqualified"

    def builder_arrays(self) -> dict:
        """Keyword arguments for the raw builder; refused unless the boundary check was admitted."""
        self.require_admitted()
        return dict(pressure_inf=self.pressure_inf, elevation_inf=self.elevation_inf, projection=self.projection,
                    panel_ids=self.panel_ids, query_point_ids=self.query_point_ids)

    def content_hash(self) -> str:
        """SHA-256 over ALL identities and ids (panel / query / surface), provider and Green-function state, geometry,
        normals / areas and every retained array (name, dtype, shape and bytes)."""
        h = hashlib.sha256()
        meta = _canonical(self._meta()).encode("utf-8")
        h.update(len(meta).to_bytes(8, "little"))
        h.update(meta)
        for name in _NATIVE_ARRAYS:
            arr = getattr(self, name)
            if arr is None:
                continue
            a = np.ascontiguousarray(arr)
            h.update(_canonical([name, list(a.shape), a.dtype.str]).encode("utf-8"))
            h.update(a.tobytes())
        return h.hexdigest()

    def arrays(self) -> dict:
        return {n: getattr(self, n) for n in _NATIVE_ARRAYS if getattr(self, n) is not None}

    def save(self, path: str, limits: Optional[FieldResourceLimits] = None) -> str:
        """Atomic versioned ``.npz`` (no pickle, no private objects); refuses above the artifact ceiling."""
        lim = limits or FieldResourceLimits()
        stored = sum(int(a.nbytes) for a in self.arrays().values())
        _require(2 * stored <= lim.max_artifact_bytes,
                 f"native coefficients need {stored} B (2x on reload) > ceiling {lim.max_artifact_bytes} B",
                 RadiationFieldLimitError)
        digest = self.content_hash()
        payload = {k: np.asarray(v) for k, v in self.arrays().items()}
        payload["hull_mask"] = payload["hull_mask"].astype(np.uint8) if "hull_mask" in payload else None
        if payload["hull_mask"] is None:
            del payload["hull_mask"]
        header = _canonical({"meta": self._meta(), "content_hash": digest}).encode("utf-8")
        payload["header"] = np.frombuffer(header, dtype=np.uint8)
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(handle, **payload)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return digest


def native_infinite_frequency_field_coefficients(
    *,
    vertices: Any,
    faces: Any,
    surface_ids: Any,
    query_points: Any,
    water_depth: float,
    rho: float = 1025.0,
    gravity: float = 9.81,
    Q0: Any,
    cg_world: Any,
    heading: float = 0.0,
    reference_omega: Optional[float] = None,
    body: Any = None,
    query_point_ids: Any = None,
    limits: Optional[FieldResourceLimits] = None,
    projection_rel_tol: float = PROJECTION_REL_TOL,
    cpt: Any = None,
    solver: Any = None,
    boundary_policy: str = BOUNDARY_POLICY_NATIVE_STRICT,
    source_profile: Any = None,
) -> NativeFieldCoefficients:
    """Six native ``omega = inf`` radiation solves of the SAME posed family -> finite field coefficients.

    ``boundary_policy``: ``'native_strict'`` (default; the raw 4096 eps gate must pass) or the explicit, user-approved
    ``'exact_dirichlet_z0'`` (query points must have z EXACTLY 0; the raw E_inf must be consistent with the pointwise
    triangle bound built from |S(z0)| of the SAME captured Green function and the normalised sources; the effective
    coefficient is then the continuum limit 0 while every raw value, the raw status and the bound are retained).
    ``|S(z0)|`` is captured with ``solver.engine.build_S_matrix`` at the actual points BEFORE the signed potential
    is formed, for either policy (evidence); the selected policy alone decides the non-success.

    * ``body``: an already prepared same-family body (mean-body DOFs, exact manifest mesh) OR
      ``reference_omega``, a declared positive FINITE frequency used only to prepare exactly that body
      through the family's ``_body_for_family`` (``omega = inf`` cannot be passed there: period 0).  No finite
      solve is performed.  Mesh, order, Q0, CG, environment and the no-lid / no-reordering guards are those of
      the floating family.
    * Each DOF: ``RadiationProblem(body, omega=np.inf, radiating_dof=name, water_depth, rho, g)`` solved by
      ``BEMSolver().solve(..., method='indirect', keep_details=True, n_threads=1)``.  ``P_a`` is
      ``-Re(pressure)/(encounter_omega*encounter_omega)`` formed on the symbolic objects and only then converted.
    * ``query_points`` are the actual world waterline points (z = 0); the potential comes from
      ``solver.compute_potential(points, result)``, ``eta = i*encounter_omega*potential/g`` and the same
      symbolic normalisation ``-Re(eta)/omega^2``.  No panel-centroid approximation.
    * The signed integration ``Proj @ P_a`` must equal ``-A_inf`` of the same solves to ``projection_rel_tol``
      (can only tighten the 1e-8 gate).

    Capytaine is imported lazily here only (or injected through ``cpt`` for tests); never qualified.
    """
    lim = limits or FieldResourceLimits()
    _require(isinstance(lim, FieldResourceLimits), "limits must be a FieldResourceLimits")
    _require(isinstance(projection_rel_tol, (int, float)) and not isinstance(projection_rel_tol, bool)
             and np.isfinite(projection_rel_tol) and 0.0 < projection_rel_tol <= PROJECTION_REL_TOL,
             f"projection_rel_tol must lie in (0, {PROJECTION_REL_TOL:g}]")
    _require(boundary_policy in BOUNDARY_POLICIES, f"boundary_policy must be one of {BOUNDARY_POLICIES}")
    v = _real_array(vertices, "vertices")
    f = np.asarray(faces)
    _require(f.dtype.kind in "iu", f"faces: exact integer indices required (got {f.dtype}); nothing is truncated")
    _require(v.ndim == 2 and v.shape[1] == 3 and f.ndim == 2 and f.shape[1] == 4 and f.shape[0] >= 1
             and int(f.min()) >= 0 and int(f.max()) < v.shape[0],
             "faces must be (n, 4) indices inside the (n_vertices, 3) vertex array")
    pts = np.array(_real_array(query_points, "query_points"), dtype=np.float64, copy=True)
    _require(pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] >= 1 and bool(np.all(np.isfinite(pts))),
             "query_points must be finite (n, 3) world waterline points")
    _require(bool(np.all(np.abs(pts[:, 2]) <= 1e-12)), "query_points must lie on the waterline z = 0")
    _require(pts.shape[0] <= lim.max_query_points, f"{pts.shape[0]} query points exceed {lim.max_query_points}",
             RadiationFieldLimitError)
    _require(f.ndim == 2 and f.shape[0] <= lim.max_panels, f"{f.shape[0]} panels exceed the {lim.max_panels}-panel bound",
             RadiationFieldLimitError)
    qids = tuple(f"q{i}" for i in range(pts.shape[0])) if query_point_ids is None else tuple(str(q) for q in query_point_ids)
    _require(len(qids) == pts.shape[0] and len(set(qids)) == len(qids), "one unique id per query point is required")
    _require((body is None) != (reference_omega is None),
             "give exactly one of: an already prepared same-family body, or a positive finite reference_omega "
             "(omega = inf cannot prepare the body)")
    if reference_omega is not None:
        _require(np.isfinite(reference_omega) and reference_omega > 0.0, "reference_omega must be positive and finite")
    if cpt is None:
        try:
            import capytaine as cpt  # lazy: the base import never touches the native dependency
        except ImportError as exc:
            raise RadiationFieldError("optional dependency capytaine is not installed") from exc

    from .floating_radiation_fields import family_request, _body_for_family
    request = family_request(v, f, surface_ids, heading=heading, water_depth=water_depth, rho=rho, gravity=gravity,
                             Q0=Q0, cg_world=cg_world)
    if source_profile is not None:
        from .transient_native_profile import bind_profile
        request = bind_profile(request, source_profile)
    if body is None:
        v_b, f_b, labels, _mesh, body = _body_for_family(cpt, v, f, surface_ids, request, float(reference_omega))
    else:
        v_b, f_b, labels = v, f, tuple(str(s) for s in surface_ids)
    _require(np.array_equal(np.asarray(body.mesh.vertices), v_b) and np.array_equal(np.asarray(body.mesh.faces), f_b),
             "native body geometry/order differs from the retained manifest (no clipping or reordering is allowed)")
    n_panels = int(np.asarray(body.mesh.faces).shape[0])
    _require(n_panels <= lim.max_panels, f"{n_panels} panels exceed the {lim.max_panels}-panel bound",
             RadiationFieldLimitError)
    _require(body.mesh_including_lid.nb_faces <= lim.max_panels,
             "full native source unknowns (hull plus lid) exceed the panel limit", RadiationFieldLimitError)
    kw = dict(body=body, omega=np.inf, water_depth=water_depth, rho=rho, g=gravity)
    if solver is None:
        if source_profile is not None:
            from .transient_native_profile import solver_for_profile
            solver = solver_for_profile(cpt, source_profile)
        else:
            solver = cpt.BEMSolver()

    hull = np.asarray(body.hull_mask, dtype=bool)
    p_inf = np.zeros((n_panels, N_DOF))
    e_inf = np.zeros((pts.shape[0], N_DOF))
    a_inf = np.zeros((N_DOF, N_DOF))
    src_rows: list = []
    pot_rows: list = []
    problem = None
    for j, name in enumerate(_DOF_NAMES):
        problem = cpt.RadiationProblem(radiating_dof=name, **kw)
        pm = problem.body.mesh
        _require(np.array_equal(np.asarray(pm.vertices), v_b) and np.array_equal(np.asarray(pm.faces), f_b),
                 "solved problem geometry/order differs from the manifest (native clipping/reordering)")
        if 'transient_source_profile' in request:
            from .transient_native_profile import verify_profile_body
            verify_profile_body(problem.body, request)
        else:
            _require(problem.body.mesh_including_lid.nb_faces == pm.nb_faces,
                     "lid profiles are unsupported: the source manifest carries no lid geometry/mask")
        res = solver.solve(problem, method="indirect", keep_details=True, n_threads=1)
        _require(getattr(res, "sources", None) is not None and getattr(res, "potential", None) is not None
                 and getattr(res, "pressure", None) is not None,
                 "backend did not retain radiation sources / potential / pressure")
        enc = getattr(res, "encounter_omega", problem.encounter_omega)
        # retained source: COMPLEX sources/omega and potential/omega (symbolic limit first, both parts kept)
        src_rows.append(_plain_complex(res.sources / enc, f"sources/omega[{name}]"))
        pot_rows.append(_plain_complex(res.potential / enc, f"potential/omega[{name}]"))
        # symbolic frequency factors cancel here, BEFORE any numeric conversion
        p_coef = -np.real(res.pressure) / (enc * enc)
        p_arr = _plain_limit(p_coef, f"P_a[{name}]")
        _require(p_arr.ndim == 1, "pressure coefficient must be 1-D over panels")
        if p_arr.shape[0] != n_panels and p_arr.shape[0] == hull.shape[0]:
            p_arr = p_arr[hull]               # physical hull panels only (lid sources stay out of the output)
        _require(p_arr.shape == (n_panels,), f"pressure coefficient has {p_arr.shape[0]} entries for {n_panels} panels")
        p_inf[:, j] = p_arr
        if j == 0:
            # |S(z0)| of the SAME captured GF at the ACTUAL points, before (and independently of) the signed potential
            abs_s = _capture_abs_s(solver, pts, problem, res, src_rows[0].shape[0], lim, boundary_policy)
        potential = solver.compute_potential(pts, res)
        e_arr = _waterline_coefficient(enc, potential, gravity, f"E_a[{name}]")
        _require(e_arr.shape == (pts.shape[0],), "elevation coefficient must have one entry per query point")
        e_inf[:, j] = e_arr
        for i, dof in enumerate(_DOF_NAMES):
            a_inf[i, j] = float(res.added_mass[dof])

    centers = np.asarray(body.mesh.faces_centers, dtype=np.float64)
    normals = np.asarray(body.mesh.faces_normals, dtype=np.float64)
    areas = np.asarray(body.mesh.faces_areas, dtype=np.float64)
    proj = physical_force_projection(centers, normals, areas, Q0, cg_world)
    err = float(np.max(np.abs(proj @ p_inf + a_inf)))
    scale = max(float(np.max(np.abs(a_inf))), np.finfo(float).tiny)
    _require(err <= projection_rel_tol * scale,
             f"signed integration of P_a differs from -A_inf of the same solves by {err:.3e} "
             f"(> {projection_rel_tol:g} x {scale:.3e})")
    # ---- Green-function state of the SAME solver: exported settings and the decomposition it actually realised
    import math
    from pathlib import Path
    gf = solver.engine.green_function
    gf_settings = {k: val for k, val in gf.exportable_settings.items() if k != "green_function"}
    provider = dict(backend="capytaine", version=str(cpt.__version__), path=str(Path(cpt.__file__).resolve()),
                    method="indirect", recipe=NATIVE_FIELD_HELPER_VERSION, engine=type(solver.engine).__name__,
                    green_function=gf.exportable_settings["green_function"])
    kh = float(problem.wavenumber * water_depth)            # inf at omega = inf (evaluated by the solver all the same)
    prony_method = str(gf_settings.get("finite_depth_prony_decomposition_method", "python")).lower()
    cached = type(gf).find_best_exponential_decomposition
    before = cached.cache_info().hits if hasattr(cached, "cache_info") else None
    prony = np.array(gf.find_best_exponential_decomposition(kh), dtype=np.float64)
    _require(before is not None and cached.cache_info().hits > before,
             "could not capture the realised finite-depth Green-function decomposition (cache miss)")
    # ---- free-surface acceleration coefficient at z = 0 is theoretically ZERO: report, never clamp
    boundary_residual = np.max(np.abs(e_inf), axis=0)
    boundary_limit = _derived_boundary_limit(v_b, request)       # the single shared definition (same as the constructor)
    _require(boundary_limit is not None, "the boundary ceiling cannot be derived from the retained geometry / environment")
    provenance = dict(
        helper=NATIVE_FIELD_HELPER_VERSION, native_dependency="capytaine", capytaine_version=str(getattr(cpt, "__version__", "unknown")),
        method="indirect", n_threads=1, keep_details=True, omega="inf (symbolic SymbolicMultiplication)",
        reference_omega=None if reference_omega is None else float(reference_omega),
        normalization=("P_a = -Re(pressure)/(encounter_omega*encounter_omega) and E_a = -Re(i*encounter_omega*"
                       "potential/g)/(encounter_omega*encounter_omega), both formed on the symbolic objects and "
                       "converted to numeric arrays only afterwards; numeric infinity is never multiplied"),
        query_points="actual world waterline points z=0 via solver.compute_potential (no centroid approximation)",
        projection=("Proj[j,p] = -area_p * n_p . u_j(c_p); physical force on the body is -p n dS; "
                    "Proj @ P_a = -A_inf checked against the same solves"),
        projection_rel_tol=float(projection_rel_tol), projection_error=err,
        request=request, surface_ids=[str(s) for s in labels],
        n_panels=n_panels, n_query_points=int(pts.shape[0]), limits=lim.to_dict(), qualification_status="unqualified",
    )
    result = NativeFieldCoefficients(
        pressure_inf=p_inf, elevation_inf=e_inf, added_mass_infinite=a_inf, projection=proj, centers=centers,
        normals=normals, areas=areas, query_points=pts,
        panel_ids=tuple(f"{i}:{s}" for i, s in enumerate(labels)), query_point_ids=qids,
        dof_names=_DOF_NAMES, projection_error=err, provenance=provenance,
        vertices=v_b, faces=np.asarray(f_b, dtype=np.int64), hull_mask=hull, surface_ids=tuple(str(s) for s in labels),
        sources_over_omega=np.stack(src_rows), potential_over_omega=np.stack(pot_rows), request=request,
        provider=provider, gf_settings=gf_settings, prony=prony, prony_kh=kh, prony_method=prony_method,
        reference_omega=None if reference_omega is None else float(reference_omega),
        boundary_residual=boundary_residual, boundary_limit=boundary_limit,
        boundary_policy=boundary_policy, abs_s_query=abs_s)
    if not result.effective_admitted:
        # FAIL CLOSED: raw residual / limit / bound / sources stay in the diagnostic snapshot; nothing clamped or zeroed
        raise NativeBoundaryError(
            f"native free-surface boundary check FAILED under policy {boundary_policy!r} ({result.policy_reason}): "
            f"per-DOF max |E_a| {[float(x) for x in boundary_residual]}, raw scale {[float(x) for x in boundary_limit]}, "
            f"raw status {result.boundary_status!r}; the coefficients are NOT admitted (diagnostic snapshot attached)",
            result)
    return result


def _capture_abs_s(solver: Any, points: np.ndarray, problem: Any, result: Any, n_src: int,
                   lim: FieldResourceLimits, boundary_policy: str) -> Optional[np.ndarray]:
    """|S| (n_points, n_src) from ``solver.engine.build_S_matrix`` with the result's own Green-function parameters
    (exactly as ``compute_potential`` uses them).  Bounded before allocation.  Under exact_dirichlet_z0 any failure
    is a typed refusal; under native_strict the evidence is optional (None)."""
    need = 3 * int(points.shape[0]) * int(n_src) * 16
    _require(need <= lim.max_work_bytes, f"|S(z0)| capture needs {need} B > working-memory limit {lim.max_work_bytes} B",
             RadiationFieldLimitError)
    try:
        params = dict(free_surface=getattr(result, "free_surface", getattr(problem, "free_surface", 0.0)),
                      water_depth=getattr(result, "water_depth", problem.water_depth),
                      wavenumber=getattr(result, "encounter_wavenumber", getattr(problem, "encounter_wavenumber", None)))
        S = solver.engine.build_S_matrix(points, problem.body.mesh_including_lid, **params)
        _require(not hasattr(S, "symbol"), "the influence matrix S(z0) is symbolic; no numeric bound is available")
        raw = np.asarray(S)
        _require(raw.dtype != object and raw.shape == (points.shape[0], n_src),
                 f"S(z0) has shape {raw.shape}, expected {(points.shape[0], n_src)}")
        _require(raw.dtype.kind in "fc", f"S(z0) has a non-numeric dtype {raw.dtype}")
        a = _abs_influence(raw)                     # deterministic |S| (exact for real, math.hypot for complex)
        _require(bool(np.all(np.isfinite(a))), "S(z0) contains non-finite values")
        return a
    except (RadiationFieldError, AttributeError, TypeError, ValueError) as exc:
        if boundary_policy == BOUNDARY_POLICY_EXACT_DIRICHLET_Z0:
            raise RadiationFieldError(f"exact_dirichlet_z0 needs |S(z0)| from the same captured GF: {exc}") from exc
        return None


class RecoveredWaterline:
    """Waterline coefficients re-evaluated at NEW points from retained sources (zero solves)."""

    policy = ("same derived ceiling 4096 eps [L/g, L R/g] as the source; raw residual kept, nothing clamped; "
              "a fake-backend measurement only, real-backend replay unverified")

    def __init__(self, elevation_inf, query_points, query_point_ids, native_hash, solve_calls,
                 boundary_residual=None, boundary_limit=None, boundary_status="unchecked",
                 boundary_policy=BOUNDARY_POLICY_NATIVE_STRICT, policy_status=None, pointwise_bound=None):
        object.__setattr__(self, "boundary_policy", str(boundary_policy))
        object.__setattr__(self, "policy_status", str(boundary_status if policy_status is None else policy_status))
        object.__setattr__(self, "pointwise_bound",
                           None if pointwise_bound is None else _frozen_any(pointwise_bound, np.float64, "pointwise_bound"))
        object.__setattr__(self, "boundary_residual",
                           None if boundary_residual is None else _frozen_any(boundary_residual, np.float64, "boundary_residual"))
        object.__setattr__(self, "boundary_limit",
                           None if boundary_limit is None else _frozen_any(boundary_limit, np.float64, "boundary_limit"))
        object.__setattr__(self, "boundary_status", str(boundary_status))
        object.__setattr__(self, "elevation_inf", _frozen_any(elevation_inf, np.float64, "elevation_inf"))
        object.__setattr__(self, "query_points", _frozen_any(query_points, np.float64, "query_points"))
        object.__setattr__(self, "query_point_ids", tuple(query_point_ids))
        object.__setattr__(self, "native_hash", native_hash)
        object.__setattr__(self, "solve_calls", int(solve_calls))

    def __setattr__(self, name, value):
        raise AttributeError("RecoveredWaterline is immutable")

    qualification_status = "unqualified"

    @property
    def is_admitted(self) -> bool:
        """RAW native gate at the new points (unchanged by any policy)."""
        return self.boundary_status == "admitted"

    @property
    def effective_admitted(self) -> bool:
        return self.policy_status == "admitted"

    @property
    def effective_elevation_inf(self) -> np.ndarray:
        """0 only under an admitted exact_dirichlet_z0 evaluation; otherwise the raw recovered coefficients."""
        if self.boundary_policy == BOUNDARY_POLICY_EXACT_DIRICHLET_Z0 and self.effective_admitted:
            return _frozen_any(np.zeros_like(self.elevation_inf), np.float64, "effective_elevation_inf")
        return self.elevation_inf


def recover_waterline_coefficients(native: NativeFieldCoefficients, query_points: Any, *, query_point_ids: Any = None,
                                   limits: Optional[FieldResourceLimits] = None, cpt: Any = None,
                                   replay: Any = None) -> RecoveredWaterline:
    """Evaluate E_a at new z = 0 points from the RETAINED sources and replayed Green function: ``compute_potential``
    only, ZERO ``solver.solve`` calls, no new Prony fit.  Provider version / module path, retained settings, exact
    manifest mesh (no clipping / reordering / lid) and the mean-body DOFs of the family are guarded."""
    _require(isinstance(native, NativeFieldCoefficients) and native.is_recoverable,
             "the coefficients do not retain a complete recoverable source")
    native.require_effective_admitted()    # a source not admitted under ITS policy is never recovered from
    policy = native.boundary_policy
    _require(native.reference_omega is not None, "no finite reference_omega was retained: the body cannot be rebuilt")
    lim = limits or FieldResourceLimits()
    pts = np.array(_real_array(query_points, "query_points"), dtype=np.float64, copy=True)
    _require(pts.ndim == 2 and pts.shape[1] == 3 and 1 <= pts.shape[0] <= lim.max_query_points
             and bool(np.all(np.isfinite(pts))) and bool(np.all(np.abs(pts[:, 2]) <= 1e-12)),
             "query_points must be finite (n, 3) waterline points with z = 0 within the point limit")
    if policy == BOUNDARY_POLICY_EXACT_DIRICHLET_Z0:
        _require(bool(np.all(pts[:, 2] == 0.0)),
                 "exact_dirichlet_z0 applies only at z EXACTLY 0; no projection or tolerance maps other points to 0")
    qids = tuple(f"q{i}" for i in range(pts.shape[0])) if query_point_ids is None else tuple(str(q) for q in query_point_ids)
    _require(len(qids) == pts.shape[0] and len(set(qids)) == len(qids), "one unique id per query point is required")
    if cpt is None:
        try:
            import capytaine as cpt
        except ImportError as exc:
            raise RadiationFieldError("optional dependency capytaine is not installed") from exc
    from pathlib import Path
    prov = native.provider
    _require(prov.get("backend") == "capytaine" and prov.get("version") == str(cpt.__version__)
             and prov.get("path") == str(Path(cpt.__file__).resolve()),
             "retained sources were produced by a different Capytaine provider; prepare again")
    if replay is None:
        from .wave_surface_fields import _replay_green_function as replay
    from types import SimpleNamespace
    from .floating_radiation_fields import _body_for_family
    request = json.loads(_canonical(dict(native.request)))
    v, f, _labels, _mesh, body = _body_for_family(cpt, native.vertices, native.faces, native.surface_ids, request,
                                                  native.reference_omega)
    _require(np.array_equal(np.asarray(body.mesh.vertices), native.vertices)
             and np.array_equal(np.asarray(body.mesh.faces), native.faces)
             and np.array_equal(np.asarray(body.hull_mask, dtype=bool), native.hull_mask),
             "rebuilt body differs from the retained manifest (clipping / reordering / mask)")
    view = SimpleNamespace(prony=np.array(native.prony), gf_settings=json.loads(_canonical(dict(native.gf_settings))),
                           prony_kh=native.prony_kh, prony_method=native.prony_method,
                           provider=json.loads(_canonical(dict(prov))))
    out = np.zeros((pts.shape[0], N_DOF))
    for j, name in enumerate(_DOF_NAMES):
        problem = cpt.RadiationProblem(body=body, omega=np.inf, radiating_dof=name, water_depth=request["water_depth"],
                                       rho=request["rho"], g=request["gravity"])
        if 'transient_source_profile' in request:
            from .transient_native_profile import verify_profile_body
            verify_profile_body(problem.body, request)
        else:
            _require(problem.body.mesh_including_lid.nb_faces == problem.body.mesh.nb_faces, "lid profiles are unsupported")
        enc = problem.encounter_omega
        # the retained decomposition belongs to this exact dimensionless wavenumber; never refit on recovery
        kh_now = float(problem.wavenumber * request["water_depth"])
        _require(kh_now == native.prony_kh,
                 f"retained decomposition belongs to kh={native.prony_kh!r}, the rebuilt problem has kh={kh_now!r}")
        res = problem.make_results_container(sources=enc * native.sources_over_omega[j])
        try:
            replayed = replay(cpt, view, problem)      # verifies installed GF settings == retained settings and kh
        except ValueError as exc:                      # WaveFieldError and friends
            raise RadiationFieldError(f"Green-function replay refused: {exc}") from exc
        solver = cpt.BEMSolver(green_function=replayed)
        if j == 0:   # |S(z0)| from the REPLAYED retained GF at the new points, before the signed potential
            abs_s = _capture_abs_s(solver, pts, problem, res, native.sources_over_omega.shape[1], lim, policy)
        out[:, j] = _waterline_coefficient(enc, solver.compute_potential(pts, res), request["gravity"], f"E_a[{name}]")
    # an admitted SOURCE does not admit a new field: the criterion of the SAME policy is applied to the new points
    limit = _derived_boundary_limit(native.vertices, native.request)
    _require(limit is not None, "the boundary ceiling cannot be derived from the retained geometry / environment")
    limit = np.minimum(limit, native.boundary_limit)
    residual = np.max(np.abs(out), axis=0)
    status = "admitted" if bool(np.all(residual <= limit)) else "failed"          # RAW gate, always reported
    bound = None if abs_s is None else _pointwise_bound(abs_s, native.sources_over_omega, float(request["gravity"]))
    if policy == BOUNDARY_POLICY_EXACT_DIRICHLET_Z0:
        pstatus = "admitted" if bool(np.all(np.abs(out) <= bound)) else "refused"
        reason = "raw E_inf exceeds the pointwise triangle bound at the new points (inconsistent)"
    else:
        pstatus, reason = status, "raw 4096 eps gate at the new points"
    rec = RecoveredWaterline(out, pts, qids, native.content_hash(), 0, residual, limit, status,
                             boundary_policy=policy, policy_status=pstatus, pointwise_bound=bound)
    if pstatus != "admitted":
        raise NativeBoundaryError(
            f"recovered waterline coefficients FAILED under policy {policy!r} ({reason}): per-DOF max |E_a| "
            f"{[float(x) for x in residual]}, raw scale {[float(x) for x in limit]} (raw values and bound retained in "
            "the diagnostic; nothing clamped)", rec)
    return rec


def load_native_field_coefficients(path: str, *, expected_hash: Optional[str] = None,
                                   limits: Optional[FieldResourceLimits] = None) -> NativeFieldCoefficients:
    """Safe neutral load (no pickle / private objects): admission from the zip directory and ``.npy`` headers BEFORE
    any array is materialised, exact key / shape / dtype set from the metadata, 2x expanded-size bound, then the full
    content hash (ids, provider, Green-function state, geometry and every array) is verified."""
    lim = limits or FieldResourceLimits()
    err = RadiationFieldSerializationError
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise err(f"cannot read native coefficients: {exc}") from exc
    _require(2 * size <= lim.max_artifact_bytes, f"artifact file {size} B (2x while loading) exceeds ceiling "
             f"{lim.max_artifact_bytes} B", RadiationFieldLimitError)
    members, header = _inspect_npz_artifact(path, err)
    try:
        meta = header["meta"]
        _require(meta.get("format") == NATIVE_COEFFICIENTS_FORMAT, "unknown native-coefficients format", err)
        present = list(meta["present"])
        if "abs_s_query" in present:
            # artifacts written with the superseded (layout-dependent) bound algorithm are refused, never migrated
            _require(meta.get("bound_algorithm") == POINTWISE_BOUND_ALGORITHM,
                     f"artifact records pointwise-bound algorithm {meta.get('bound_algorithm', 'pointwise-bound-matmul-v1')!r}; "
                     f"only {POINTWISE_BOUND_ALGORITHM!r} is admitted - recapture required (no silent migration)", err)
        _require(set(present) <= set(_NATIVE_ARRAYS) and len(set(present)) == len(present), "invalid array list", err)
        n_f, n_q = len(meta["panel_ids"]), len(meta["query_point_ids"])
        n_source = n_f
        if meta.get('request') is not None and 'transient_source_profile' in meta['request']:
            from .transient_native_profile import profile_from_request
            n_source = len(profile_from_request(meta['request']).source_faces)
            _require(n_source <= lim.max_panels, "full source count exceeds the panel limit", RadiationFieldLimitError)
        _require(0 < n_f <= lim.max_panels and 0 < n_q <= lim.max_query_points, "declared counts outside the limits",
                 RadiationFieldLimitError)
        _require(set(members) == set(present), f"unexpected or missing arrays: {sorted(set(members) ^ set(present))}", err)
        free = {"vertices": 3, "sources_over_omega": 6, "potential_over_omega": 6, "prony": 2,   # leading-fixed, free tail
                "abs_s_query": n_q}
        fixed = {"pressure_inf": (n_f, 6), "elevation_inf": (n_q, 6), "added_mass_infinite": (6, 6), "projection": (6, n_f),
                 "centers": (n_f, 3), "normals": (n_f, 3), "areas": (n_f,), "query_points": (n_q, 3),
                 "faces": (n_f, 4), "hull_mask": (n_source,), "boundary_residual": (6,), "boundary_limit": (6,)}
        dtypes = {"faces": "<i8", "hull_mask": "u1", "sources_over_omega": "<c16", "potential_over_omega": "<c16"}
        total = 0
        for key in present:
            shape, dtype, _size = members[key]
            want_dtype = np.dtype(dtypes.get(key, "<f8"))
            _require(dtype == want_dtype, f"array {key!r} has dtype {dtype}, expected {want_dtype}", err)
            if key in fixed:
                _require(shape == fixed[key], f"array {key!r} has shape {shape}, expected {fixed[key]}", err)
            else:
                _require(len(shape) == 2 and shape[0 if key == "vertices" else 0] == (
                    free[key] if key != "vertices" else shape[0]) and (key != "vertices" or shape[1] == 3),
                    f"array {key!r} has an invalid shape {shape}", err)
            count = 1
            for d in shape:
                count *= int(d)
            total += count * want_dtype.itemsize
        _require(2 * total <= lim.max_artifact_bytes, f"expanded arrays {total} B (2x while loading) exceed the ceiling "
                 f"{lim.max_artifact_bytes} B", RadiationFieldLimitError)
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in present}
        kh = meta["prony_kh"]
        kh = float("inf") if kh == "inf" else float(kh)
        fields = {k: arrays.get(k) for k in _NATIVE_ARRAYS}
        if fields["hull_mask"] is not None:
            fields["hull_mask"] = fields["hull_mask"].astype(bool)
        obj = NativeFieldCoefficients(
            pressure_inf=fields.pop("pressure_inf"), elevation_inf=fields.pop("elevation_inf"),
            added_mass_infinite=fields.pop("added_mass_infinite"), projection=fields.pop("projection"),
            centers=fields.pop("centers"), normals=fields.pop("normals"), areas=fields.pop("areas"),
            query_points=fields.pop("query_points"), panel_ids=tuple(meta["panel_ids"]),
            query_point_ids=tuple(meta["query_point_ids"]), dof_names=tuple(meta["dof_names"]),
            projection_error=float(meta["projection_error"]), provenance=meta["provenance"],
            surface_ids=tuple(meta["surface_ids"]), request=meta["request"], provider=meta["provider"],
            gf_settings=meta["gf_settings"], prony_kh=kh, prony_method=meta["prony_method"],
            reference_omega=meta["reference_omega"], boundary_status=str(meta.get("boundary_status", "")),
            boundary_policy=str(meta.get("boundary_policy", BOUNDARY_POLICY_NATIVE_STRICT)),
            policy_status=str(meta.get("policy_status", "")), **fields)
    except RadiationFieldError:
        raise
    except Exception as exc:
        raise err(f"native coefficients artifact is corrupt or malformed: {exc}") from exc
    if obj.content_hash() != header.get("content_hash"):
        raise err("content hash mismatch: the artifact was modified or corrupted")
    if expected_hash is not None and obj.content_hash() != expected_hash:
        raise err("artifact content differs from the expected (stale) hash")
    return obj
