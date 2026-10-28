"""Immutable fixed-body incident + diffraction response banks (Pa/m, SI).

Data and persistence import no BEM engine. Frequencies are exact float identities;
phasors multiply exp(-i omega t), positive pressure acts inward.
"""
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
import time
import numpy as np

VERSION = 'pressure_bank_v1'
CONVENTIONS = dict(units='Pa/m', phasor='exp(-i omega t)', normals='outward',
                   pressure='positive inward', physics='fixed incident+diffraction')


class PressureBankError(ValueError):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _array(value, dtype):
    raw = np.asarray(value)
    if dtype == np.float64 and raw.dtype.kind not in 'fiu':
        raise PressureBankError('real numeric bank arrays required, excluding bool/complex')
    a = np.asarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class PressureResponseBank:
    vertices: np.ndarray
    faces: np.ndarray
    surface_ids: tuple
    centers: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    omegas: np.ndarray
    responses: np.ndarray  # frequency, source panel
    heading: float = 0.
    water_depth: float = 200.
    rho: float = 1025.
    gravity: float = 9.81
    phase_origin: tuple = (0., 0., 0.)
    panel_ids: tuple = ()
    provenance_json: str = '{}'
    bank_id: str = field(default='')

    @property
    def conventions(self):
        return dict(CONVENTIONS)

    def __post_init__(self):
        original_frequencies = tuple(self.omegas)
        if any(isinstance(x, (bool, np.bool_, complex, np.complexfloating)) or
                not isinstance(x, (int, float, np.integer, np.floating)) for x in original_frequencies):
            raise PressureBankError('real numeric frequencies required, excluding bool/complex')
        for name in ('vertices', 'centers', 'normals', 'areas', 'omegas'):
            object.__setattr__(self, name, _array(getattr(self, name), np.float64))
        raw = np.asarray(self.faces)
        if raw.dtype.kind not in 'iu':
            raise PressureBankError('integer faces required')
        object.__setattr__(self, 'faces', _array(raw, np.int64))
        object.__setattr__(self, 'responses', _array(self.responses, np.complex128))
        object.__setattr__(self, 'surface_ids', tuple(self.surface_ids))
        object.__setattr__(self, 'panel_ids', tuple(self.panel_ids) or tuple(range(len(self.faces))))
        object.__setattr__(self, 'phase_origin', tuple(self.phase_origin))
        # Canonical JSON is immutable, unlike a retained caller-owned mapping.
        object.__setattr__(self, 'provenance_json', _json(json.loads(self.provenance_json)))
        self._validate_shapes()
        expected = self._identity()
        if self.bank_id and self.bank_id != expected:
            raise PressureBankError('bank content identity mismatch')
        object.__setattr__(self, 'bank_id', expected)

    def _metadata(self):
        return dict(version=VERSION, conventions=CONVENTIONS, surface_ids=self.surface_ids,
            panel_ids=self.panel_ids, heading=self.heading, water_depth=self.water_depth,
            rho=self.rho, gravity=self.gravity, phase_origin=self.phase_origin,
            provenance=json.loads(self.provenance_json))

    def _identity(self):
        h = hashlib.sha256(_json(self._metadata()).encode())
        for name in ('vertices', 'faces', 'centers', 'normals', 'areas', 'omegas', 'responses'):
            a = getattr(self, name)
            h.update(_json((name, a.shape, a.dtype.str)).encode()); h.update(memoryview(a).cast('B'))
        return h.hexdigest()

    def _validate_shapes(self):
        n, k = len(self.faces), len(self.omegas)
        if (self.vertices.ndim != 2 or self.vertices.shape[1:] != (3,) or
                self.faces.shape != (n, 4) or not n or self.faces.min() < 0 or
                self.faces.max() >= len(self.vertices) or self.centers.shape != (n, 3) or
                self.normals.shape != (n, 3) or self.areas.shape != (n,) or
                self.omegas.shape != (k,) or not k or self.responses.shape != (k, n)):
            raise PressureBankError('bank geometry/response shape mismatch')
        if (len(self.surface_ids) != n or any(type(x) is not str for x in self.surface_ids) or
                len(self.panel_ids) != n or len(set(self.panel_ids)) != n or
                any(type(x) is not int for x in self.panel_ids)):
            raise PressureBankError('stable unique integer panel IDs and surface labels required')
        if (any(not all(np.isfinite(part).all() for part in np.array_split(getattr(self, x).reshape(-1),
                    max(1, getattr(self, x).size//4096+1))) for x in
                ('vertices', 'centers', 'normals', 'areas', 'omegas', 'responses')) or
                np.any(self.areas <= 0) or np.any(self.omegas <= 0) or
                len(set(self.omegas)) != k or not np.allclose(
                    np.linalg.norm(self.normals, axis=1), 1., rtol=0., atol=1e-8)):
            raise PressureBankError('finite bank data, unique positive frequencies and unit normals required')
        if (not math.isfinite(self.heading) or not 0 <= self.heading < math.tau or
                any(not math.isfinite(x) or x <= 0 for x in (self.rho, self.gravity, self.water_depth)) or
                len(self.phase_origin) != 3 or not np.isfinite(self.phase_origin).all()):
            raise PressureBankError('canonical heading and positive physical constants required')

    def validate(self):
        self._validate_shapes()
        if self._identity() != self.bank_id:
            raise PressureBankError('bank changed after identity was issued')

    def save(self, path):
        self.validate()
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._metadata(); metadata['bank_id'] = self.bank_id
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name+'.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez(stream, metadata=_json(metadata), **{name: getattr(self, name) for name in
                    ('vertices', 'faces', 'centers', 'normals', 'areas', 'omegas', 'responses')})
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)

    def to_dict(self):
        """Canonical owner record for self-contained neutral persistence."""
        self.validate()
        return dict(record_type='anyhydro_pressure_response_bank_v1',metadata={**self._metadata(),'bank_id':self.bank_id},
            arrays={**{name:getattr(self,name).tolist() for name in
                ('vertices','faces','centers','normals','areas','omegas')},
                'responses_real':self.responses.real.tolist(),'responses_imag':self.responses.imag.tolist()})

    @classmethod
    def from_dict(cls,data):
        try:
            if (type(data) is not dict or set(data)!={'record_type','metadata','arrays'} or
                    data['record_type']!='anyhydro_pressure_response_bank_v1'):
                raise PressureBankError('unsupported owner pressure bank record')
            metadata=dict(data['metadata']);arrays=dict(data['arrays'])
            if metadata.pop('version')!=VERSION or metadata.pop('conventions')!=CONVENTIONS:
                raise PressureBankError('unsupported pressure bank conventions')
            if set(arrays)!={'vertices','faces','centers','normals','areas','omegas','responses_real','responses_imag'}:
                raise PressureBankError('invalid owner pressure bank array inventory')
            metadata['provenance_json']=_json(metadata.pop('provenance'))
            real=np.asarray(arrays.pop('responses_real'));imag=np.asarray(arrays.pop('responses_imag'))
            if real.dtype.kind not in 'fiu' or imag.dtype.kind not in 'fiu' or real.shape!=imag.shape:
                raise PressureBankError('real matching pressure response parts required')
            responses=np.empty(real.shape,dtype=np.complex128)
            responses.real=real;responses.imag=imag
            return cls(**arrays,responses=responses,**metadata)
        except PressureBankError:raise
        except (KeyError,TypeError,ValueError) as error:
            raise PressureBankError('corrupt owner pressure bank record') from error

    @classmethod
    def load(cls, path):
        try:
            with np.load(path, allow_pickle=False) as z:
                m = json.loads(str(z['metadata']))
                if m.pop('version') != VERSION or m.pop('conventions') != CONVENTIONS:
                    raise PressureBankError('unsupported bank version/conventions')
                m['provenance_json'] = _json(m.pop('provenance'))
                return cls(**{x: z[x] for x in ('vertices', 'faces', 'centers', 'normals',
                    'areas', 'omegas', 'responses')}, **m)
        except Exception as error:
            if isinstance(error, PressureBankError): raise
            raise PressureBankError(f'corrupt pressure bank: {path}') from error


@dataclass(frozen=True)
class PreparationEstimate:
    """Inventory/storage estimates, not elapsed-time or peak-RSS predictions."""
    panels: int
    requested_frequency_count: int
    cache_hits: int
    missing_solve_count: int
    response_storage_bytes: int
    dense_matrix_bytes: int
    max_wall_seconds: float
    max_missing_solves: int
    matrix_estimate_kind: str = 'default-engine three dense complex128 matrices'


@dataclass(frozen=True)
class PreparationProgress:
    completed: tuple
    missing: tuple
    cached_count: int
    solved_count: int
    elapsed_seconds: float
    estimate: PreparationEstimate | None = None


class PressurePreparationError(RuntimeError):
    def __init__(self, message, progress):
        super().__init__(message); self.progress = progress
        self.completed = progress.completed; self.missing = progress.missing


def prepare_pressure_bank(vertices, faces, surface_ids, omegas, *, cache_directory,
        heading=0., water_depth=200., rho=1025., gravity=9.81, phase_origin=(0., 0., 0.),
        max_panels=600, max_missing_solves=32, max_wall_seconds=120., cancel=None,
        progress=None):
    """Serial resumable preparation, default Capytaine 3.0 Green function/indirect.

    Wall budget covers admission, cache reads and solves. A running native solve
    cannot be interrupted in-process; budget/cancel are checked before/after it.
    Completed frequencies are atomically persisted before reporting interruption.
    Callers needing a hard deadline must supervise this call in a child process.
    """
    started = time.perf_counter()
    raw_omegas = tuple(omegas)
    if any(isinstance(x, (bool, np.bool_, complex, np.complexfloating)) or
            not isinstance(x, (int, float, np.integer, np.floating)) for x in raw_omegas):
        raise PressureBankError('real numeric frequencies required, excluding bool/complex')
    requested = tuple(float(x) for x in raw_omegas)
    if (not requested or any(not math.isfinite(x) or x <= 0 for x in requested) or
            len(set(requested)) != len(requested)):
        raise PressureBankError('unique exact positive frequencies required')
    if (type(max_missing_solves) is not int or max_missing_solves < 0 or
            isinstance(max_wall_seconds, bool) or not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0 or
            type(max_panels) is not int or not 0 < max_panels <= 600):
        raise PressureBankError('explicit positive panel/wall limits and nonnegative solve limit required')
    heading = float(heading) % math.tau
    origin = tuple(float(x) for x in phase_origin)
    if origin != (0., 0., 0.):
        raise PressureBankError('v1 preparation requires WORLD0 phase origin')
    if not math.isfinite(gravity) or gravity <= 0:
        raise PressureBankError('positive finite gravity required')
    import capytaine as cpt
    from capytaine.bem.airy_waves import airy_waves_pressure, froude_krylov_force
    from ._pressure_body import prepare_body
    version = importlib.metadata.version('capytaine')
    if version.split('.')[:2] != ['3', '0']:
        raise PressureBankError('pressure adapter requires Capytaine 3.0')
    v, f, labels, mesh, body = prepare_body(cpt, vertices, faces, surface_ids,
        period=2*math.pi/requested[0], heading=heading, water_depth=water_depth,
        rho=rho, max_panels=max_panels)
    solver = cpt.BEMSolver()
    provenance = _json(dict(backend='capytaine', version=version, path=str(Path(cpt.__file__).resolve()),
        method='indirect', green_function=repr(solver.engine.green_function),
        engine=repr(solver.engine), effective_settings=solver.exportable_settings,
        n_threads=1, jobs=1))
    common = dict(vertices=v, faces=f, surface_ids=labels, centers=mesh.faces_centers,
        normals=mesh.faces_normals, areas=mesh.faces_areas, heading=heading,
        water_depth=water_depth, rho=rho, gravity=gravity, phase_origin=origin,
        provenance_json=provenance)
    # Zero response template hashes geometry, ordering, labels, physical settings
    # and the effective backend without conflating it with final content identity.
    PressureResponseBank(omegas=[requested[0]], responses=np.zeros((1, len(f))), **common)
    geometry_key = PressureResponseBank(omegas=[1.], responses=np.zeros((1, len(f))), **common).bank_id
    cache = Path(cache_directory); cache.mkdir(parents=True, exist_ok=True)
    found = {}; cached = 0; solved = 0; inventory_complete = False
    def state():
        estimate = (PreparationEstimate(len(f), len(requested), cached, len(requested)-cached,
            16*len(requested)*len(f), 3*16*len(f)*len(f), float(max_wall_seconds), max_missing_solves)
            if inventory_complete else None)
        return PreparationProgress(tuple(w for w in requested if w in found),
            tuple(w for w in requested if w not in found), cached, solved,
            time.perf_counter()-started, estimate)
    def report():
        if progress is not None: progress(state())
    def stop():
        if (cancel is not None and cancel()) or time.perf_counter()-started >= max_wall_seconds:
            raise PressurePreparationError('preparation cancelled or wall budget reached', state())
    def entry_path(w):
        return cache/(hashlib.sha256((geometry_key+float(w).hex()).encode()).hexdigest()+'.npz')
    try:
        for w in requested:
            stop(); path = entry_path(w)
            if path.exists():
                entry = PressureResponseBank.load(path)
                expected = PressureResponseBank(omegas=[w], responses=entry.responses, **common)
                if expected.bank_id != entry.bank_id:
                    raise PressureBankError('cache entry metadata mismatch')
                found[w] = entry.responses[0]; cached += 1
                stop()
        inventory_complete = True
        report()
        if len(requested)-len(found) > max_missing_solves:
            raise PressurePreparationError('missing solve count exceeds preparation limit', state())
        for w in requested:
            if w in found: continue
            stop()
            problem = cpt.DiffractionProblem(body=body, omega=w, wave_direction=heading,
                water_depth=water_depth, rho=rho, g=gravity)
            result = solver.solve(problem, method='indirect', keep_details=True, n_threads=1)
            response = np.asarray(result.pressure) + airy_waves_pressure(mesh.faces_centers, problem)
            force = -response[:, None]*mesh.faces_areas[:, None]*mesh.faces_normals
            integral = np.r_[force.sum(axis=0), np.cross(mesh.faces_centers, force).sum(axis=0)]
            fk = froude_krylov_force(problem)
            wrench = np.asarray([result.forces[key]+fk[key] for key in
                ('Surge','Sway','Heave','Roll','Pitch','Yaw')])
            if not np.allclose(integral, wrench, rtol=1e-8, atol=1e-6):
                raise PressureBankError('BEM excitation differs from integrated panel pressure')
            entry = PressureResponseBank(omegas=[w], responses=response[None, :], **common)
            entry.save(entry_path(w)); found[w] = entry.responses[0]; solved += 1
            del result, problem
            report(); stop()
        completed_bank = PressureResponseBank(omegas=requested,
            responses=np.stack([found[w] for w in requested]), **common)
        stop()
        return completed_bank
    except PressurePreparationError:
        raise
    except Exception as error:
        raise PressurePreparationError(f'preparation failed: {error}', state()) from error
