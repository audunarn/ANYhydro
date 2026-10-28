"""Retained floating-body field family: 6 radiation + 1 diffraction solution from ONE native solver state.

All of A, B, Fexc and the pressure/elevation recovery come from the SAME Capytaine 3.0 indirect solve family (one
``BEMSolver``, one Green function, one realised finite-depth Prony state, one mesh).  The general ``CapytaineBackend``
uses a different direct/custom-GF setup: its coefficients must never be mixed into this family.

Generalized coordinates are MEAN BODY axes through the mean CG: translation axes are the columns of ``Q0`` and
rotation axes pass through ``cg_world`` along those columns, so A, B, Fexc and ``pR`` are already work-conjugate in
that basis.  Radiation pressure from Capytaine is per unit displacement (RHS = -i w n.dof, p = i w rho phi): no extra
-i w is applied.  Incident wave is the native Airy wave, exp(-i w t), unit amplitude, FIXED WORLD phase origin.

The posed hull must be clipped by the geometry owner BEFORE this module; the solved body's vertices/faces must equal
the manifest exactly (no native auto-clipping or reordering).  Full source vectors (lid sources included when a lid is
present) stay in the record; masks apply only to physical OUTPUT hull pressures/forces.  Native work requires
``ANYFEM_NATIVE_FLOATING_FIELD=1`` and never happens in recovery.
"""
from dataclasses import dataclass
import hashlib, json, math, os
from pathlib import Path
import numpy as np

from .wave_surface_fields import (WaveFieldError, WaveFieldMissing, _json, _numeric_hash, _frozen, _replay_green_function,
                                  _sha)

VERSION = 'floating_field_family_v1'
RECIPE = 'native_indirect_6rad_1diff_mean_body_axes_v1'
NAMESPACE = 'floating-field-families'
DOF_NAMES = ('Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw')
NATIVE_GATE = 'ANYFEM_NATIVE_FLOATING_FIELD'
CONVENTIONS = dict(phasor='exp(-i omega t)', radiation='per unit displacement, p=i*omega*rho*phi, rhs=-i*omega*n.dof',
                   incident='native Airy unit amplitude, fixed world phase origin', axes='mean body through mean CG',
                   normals='outward from hull into fluid')


def _rotation(Q0):
    Q = np.array(Q0, dtype=np.float64)
    if Q.shape != (3, 3) or not np.isfinite(Q).all() or np.abs(Q.T @ Q - np.eye(3)).max() > 1e-10 or np.linalg.det(Q) < 0:
        raise WaveFieldError('Q0 must be a proper body-to-world rotation')
    return Q


def family_request(vertices, faces, surface_ids, *, heading, water_depth, rho, gravity, Q0, cg_world):
    v = np.asarray(vertices, dtype=np.float64); f = np.asarray(faces)
    if f.dtype.kind not in 'iu' or f.ndim != 2 or f.shape[1] != 4:
        raise WaveFieldError('integer quad/degenerate-triangle source faces required')
    cg = np.asarray(cg_world, dtype=np.float64).reshape(3)
    values = (heading, water_depth, rho, gravity, *cg)
    if not all(math.isfinite(float(x)) for x in values) or min(water_depth, rho, gravity) <= 0:
        raise WaveFieldError('invalid physical settings')
    return dict(version=VERSION, recipe=RECIPE, heading=float(heading) % math.tau, water_depth=float(water_depth),
                rho=float(rho), gravity=float(gravity), conventions=CONVENTIONS, Q0=_rotation(Q0).tolist(),
                cg_world=cg.tolist(),
                geometry=_numeric_hash(dict(surface_ids=tuple(str(x) for x in surface_ids)),
                                       dict(vertices=v, faces=f.astype(np.int64))))


@dataclass(frozen=True)
class FloatingFamilySnapshot:
    """One frequency: full source vectors for 6 radiation + diffraction, coefficients, backend state."""
    request: dict
    omega: float
    vertices: np.ndarray
    faces: np.ndarray
    surface_ids: tuple
    radiation_sources: np.ndarray      # (6, n_sources) complex, per unit displacement (lid sources included)
    diffraction_sources: np.ndarray    # (n_sources,)
    added_mass: np.ndarray             # (6,6) raw, unsymmetrised, mean body axes
    damping: np.ndarray                # (6,6)
    excitation: np.ndarray             # (6,) complex FK + diffraction, unit amplitude
    provider: dict
    prony: object = None
    prony_kh: float = 0.
    prony_method: str = ''
    gf_settings: object = None
    snapshot_id: str = ''
    # Native surface pressures at the SAME face centres/normals/areas the native A, B, Fexc integrals used, so source
    # wrenches close against A/B/Fexc to roundoff (no offset-evaluation error).  Hull faces only.
    panel_pressure_radiation: object = None     # (6, nfaces) per unit displacement, i w rho phi_R
    panel_pressure_incident_diffraction: object = None   # (nfaces,) unit amplitude, native Airy + diffraction
    centers: object = None
    normals: object = None
    areas: object = None

    def __post_init__(self):
        for name, dtype in (('vertices', np.float64), ('faces', np.int64), ('radiation_sources', np.complex128),
                            ('diffraction_sources', np.complex128), ('added_mass', np.float64), ('damping', np.float64),
                            ('excitation', np.complex128)):
            object.__setattr__(self, name, _frozen(getattr(self, name), dtype))
        extras = (('panel_pressure_radiation', np.complex128), ('panel_pressure_incident_diffraction', np.complex128),
                  ('centers', np.float64), ('normals', np.float64), ('areas', np.float64))
        present = [getattr(self, n) is not None for n, _ in extras]
        if any(present) and not all(present):
            raise WaveFieldError('panel pressure data must be complete')
        for n, dtype in extras:
            if getattr(self, n) is not None:
                object.__setattr__(self, n, _frozen(getattr(self, n), dtype))
        if all(present):
            nf = len(self.faces)
            if (self.panel_pressure_radiation.shape != (6, nf) or self.panel_pressure_incident_diffraction.shape != (nf,)
                    or self.centers.shape != (nf, 3) or self.normals.shape != (nf, 3) or self.areas.shape != (nf,)
                    or not all(np.isfinite(getattr(self, n)).all() for n, _ in extras)):
                raise WaveFieldError('invalid native panel pressure data')
        if self.prony is not None:
            object.__setattr__(self, 'prony', _frozen(self.prony, np.float64))
        n = self.radiation_sources.shape[-1] if self.radiation_sources.ndim == 2 else -1
        if (not math.isfinite(self.omega) or self.omega <= 0 or self.radiation_sources.shape != (6, n) or n < len(self.faces)
                or self.diffraction_sources.shape != (n,) or self.added_mass.shape != (6, 6) or self.damping.shape != (6, 6)
                or self.excitation.shape != (6,) or self.request.get('version') != VERSION
                or self.request.get('recipe') != RECIPE or len(self.surface_ids) != len(self.faces)):
            raise WaveFieldError('invalid floating field snapshot')
        arrays = (self.radiation_sources, self.diffraction_sources, self.added_mass, self.damping, self.excitation)
        if not all(np.isfinite(a).all() for a in arrays):
            raise WaveFieldError('non-finite floating field snapshot')
        if _numeric_hash(dict(surface_ids=tuple(self.surface_ids)), dict(vertices=self.vertices, faces=self.faces)) \
                != self.request['geometry']:
            raise WaveFieldError('floating snapshot geometry differs from its request')
        if 'transient_source_profile' in self.request:
            from .transient_native_profile import profile_from_request
            profile = profile_from_request(self.request)
            if n != len(profile.source_faces):
                raise WaveFieldError('retained sources do not match the complete transient source profile')
        ident = self._identity()
        if self.snapshot_id and self.snapshot_id != ident:
            raise WaveFieldError('floating snapshot content identity mismatch')
        object.__setattr__(self, 'snapshot_id', ident)

    def _identity(self):
        arrays = dict(vertices=self.vertices, faces=self.faces, rad=self.radiation_sources, dif=self.diffraction_sources,
                      A=self.added_mass, B=self.damping, F=self.excitation)
        meta = dict(request=self.request, omega=float(self.omega).hex(), provider=self.provider,
                    surface_ids=tuple(self.surface_ids))
        if self.prony is not None:
            arrays['prony'] = self.prony
            meta.update(prony_kh=float(self.prony_kh).hex(), prony_method=self.prony_method, gf_settings=self.gf_settings)
        if self.centers is not None:
            arrays.update(ppr=self.panel_pressure_radiation, ppd=self.panel_pressure_incident_diffraction,
                          ctr=self.centers, nrm=self.normals, are=self.areas)
        return _numeric_hash(meta, arrays)

    def validate(self):
        if self._identity() != self.snapshot_id:
            raise WaveFieldError('floating snapshot metadata changed after validation')


def _path(cache, request, omega):
    return Path(cache) / NAMESPACE / (hashlib.sha256(_json(dict(request=request, omega=float(omega).hex())).encode()).hexdigest() + '.npz')


def save_family_snapshot(cache, snap):
    import tempfile
    path = _path(cache, snap.request, snap.omega); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as s:
            np.savez(s, metadata=_json(dict(request=snap.request, omega=float(snap.omega).hex(), surface_ids=snap.surface_ids,
                                            provider=snap.provider, snapshot_id=snap.snapshot_id, prony_kh=snap.prony_kh,
                                            prony_method=snap.prony_method, gf_settings=snap.gf_settings,
                                            has_prony=snap.prony is not None, has_panels=snap.centers is not None)),
                     vertices=snap.vertices, faces=snap.faces, rad=snap.radiation_sources, dif=snap.diffraction_sources,
                     A=snap.added_mass, B=snap.damping, F=snap.excitation,
                     **({} if snap.prony is None else dict(prony=snap.prony)),
                     **({} if snap.centers is None else dict(ppr=snap.panel_pressure_radiation,
                         ppd=snap.panel_pressure_incident_diffraction, ctr=snap.centers, nrm=snap.normals, are=snap.areas)))
            s.flush(); os.fsync(s.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    save_radiation_donor(cache, snap)
    return path


def snapshot_to_payload(snap):
    """(metadata JSON text, arrays dict) - self-contained, for embedding in a saved artifact."""
    meta = _json(dict(request=snap.request, omega=float(snap.omega).hex(), surface_ids=list(snap.surface_ids),
                      provider=snap.provider, snapshot_id=snap.snapshot_id, prony_kh=snap.prony_kh,
                      prony_method=snap.prony_method, gf_settings=snap.gf_settings, has_prony=snap.prony is not None,
                      has_panels=snap.centers is not None))
    arrays = dict(vertices=snap.vertices, faces=snap.faces, rad=snap.radiation_sources, dif=snap.diffraction_sources,
                  A=snap.added_mass, B=snap.damping, F=snap.excitation)
    if snap.prony is not None:
        arrays['prony'] = snap.prony
    if snap.centers is not None:
        arrays.update(ppr=snap.panel_pressure_radiation, ppd=snap.panel_pressure_incident_diffraction, ctr=snap.centers,
                      nrm=snap.normals, are=snap.areas)
    return meta, arrays


def snapshot_from_payload(meta_text, arrays):
    """Validated reconstruction (identity re-verified); never solves."""
    meta = json.loads(meta_text)
    omega = float.fromhex(meta['omega'])
    return FloatingFamilySnapshot(meta['request'], omega, arrays['vertices'], arrays['faces'], tuple(meta['surface_ids']),
                                  arrays['rad'], arrays['dif'], arrays['A'], arrays['B'], arrays['F'], meta['provider'],
                                  arrays['prony'] if meta.get('has_prony') else None, float(meta.get('prony_kh', 0.)),
                                  meta.get('prony_method', ''), meta.get('gf_settings'), meta['snapshot_id'],
                                  *((arrays['ppr'], arrays['ppd'], arrays['ctr'], arrays['nrm'], arrays['are'])
                                    if meta.get('has_panels') else (None,) * 5))


def _donor_path(cache, request, omega):
    key = hashlib.sha256((radiation_key(request) + '|' + float(omega).hex()).encode()).hexdigest()
    return Path(cache) / NAMESPACE / 'radiation-basis' / (key + '.npz')


def save_radiation_donor(cache, snap):
    """Index a snapshot as the reusable radiation basis of (geometry, axes, environment, omega); first writer wins."""
    path = _donor_path(cache, snap.request, snap.omega)
    if path.exists():
        return path
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    meta, arrays = snapshot_to_payload(snap)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as s:
            np.savez(s, metadata=meta, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def recover_radiation_donor(cache, request, omega):
    """A validated snapshot with the SAME radiation basis (any heading), or None; never solves."""
    path = _donor_path(cache, request, omega)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as d:
            snap = snapshot_from_payload(str(d['metadata']), {k: np.array(d[k]) for k in d.files if k != 'metadata'})
        return snap if radiation_key(snap.request) == radiation_key(request) and snap.centers is not None else None
    except WaveFieldError:
        raise
    except Exception as error:
        raise WaveFieldError(f'corrupt radiation basis {path.name}; never silently overwritten') from error


def missing_radiation_rhs(cache, request, omega):
    """Actual native RHS count a Prepare needs for this (omega, heading): 0 cached, 1 (diffraction only) with a donor, else 7."""
    try:
        recover_family_snapshot(cache, request, omega)
        return 0
    except WaveFieldMissing:
        return 1 if recover_radiation_donor(cache, request, omega) is not None else 7


def recover_family_snapshot(cache, request, omega):
    """Load and validate; NEVER solves (reopen/remap of an unchanged hull needs no BEM)."""
    path = _path(cache, request, omega)
    if not path.exists():
        raise WaveFieldMissing(f'floating field family for omega={omega!r} is missing; prepare explicitly')
    try:
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d['metadata']))
            if _json(meta['request']) != _json(request) or meta['omega'] != float(omega).hex() or not _sha(meta['snapshot_id']):
                raise WaveFieldError('floating snapshot request/frequency mismatch')
            return FloatingFamilySnapshot(meta['request'], float(omega), d['vertices'], d['faces'], tuple(meta['surface_ids']),
                                          d['rad'], d['dif'], d['A'], d['B'], d['F'], meta['provider'],
                                          d['prony'] if meta.get('has_prony') else None, float(meta.get('prony_kh', 0.)),
                                          meta.get('prony_method', ''), meta.get('gf_settings'), meta['snapshot_id'],
                                          *((d['ppr'], d['ppd'], d['ctr'], d['nrm'], d['are']) if meta.get('has_panels')
                                            else (None,) * 5))
    except WaveFieldError:
        raise
    except Exception as error:
        raise WaveFieldError(f'corrupt floating snapshot {path.name}; never silently overwritten') from error


def missing_family_frequencies(cache, request, omegas):
    missing = []
    for w in omegas:
        try:
            recover_family_snapshot(cache, request, w)
        except WaveFieldMissing:
            missing.append(float(w))
    return tuple(missing)


def _body_for_family(cpt, vertices, faces, surface_ids, request, omega):
    if 'transient_source_profile' in request:
        from .transient_native_profile import body_for_profile
        return body_for_profile(cpt, vertices, faces, surface_ids, request, omega)
    from ._pressure_body import prepare_body
    Q = np.array(request['Q0']); cg = np.array(request['cg_world'])
    v, f, labels, mesh, body = prepare_body(cpt, vertices, faces, surface_ids, period=math.tau / omega,
                                            heading=request['heading'], water_depth=request['water_depth'],
                                            rho=request['rho'], max_panels=600)
    # The native body must be EXACTLY the posed, pre-clipped manifest: no auto clipping, no reordering.
    if not (np.array_equal(np.asarray(body.mesh.vertices), v) and np.array_equal(np.asarray(body.mesh.faces), f)):
        raise WaveFieldError('solved body geometry/order differs from the retained manifest (native clipping/reordering)')
    body.dofs.clear()
    for i, name in enumerate(DOF_NAMES[:3]):
        body.add_translation_dof(direction=Q[:, i].copy(), name=name)
    for i, name in enumerate(DOF_NAMES[3:]):
        body.add_rotation_dof(rotation_center=cg.copy(), direction=Q[:, i].copy(), name=name)
    return v, f, labels, mesh, body


def radiation_key(request):
    """Identity of the heading-INDEPENDENT radiation basis: geometry, axes, origin, environment (no wave heading)."""
    return _json({k: v for k, v in request.items() if k != 'heading'})


def solve_family_snapshot(vertices, faces, surface_ids, omega, *, heading, water_depth, rho=1025., gravity=9.81, Q0, cg_world,
                          donor=None, source_profile=None):
    """Native 6 radiation + 1 diffraction indirect solves with ONE solver (explicit Prepare).

    ``donor``: a retained snapshot of the SAME radiation basis (``radiation_key`` + omega equal, any heading).  The six
    radiation solutions, A, B and the realised Green-function/Prony state are REUSED verbatim and only the diffraction RHS
    for the new heading is solved, under that replayed state - so A, B, Fexc and pressures still come from one solver state."""
    # No environment gate here: the delivered product runs this on an explicit Prepare.  ``NATIVE_GATE`` only schedules
    # the native pytest nodes (they are skipped unless the parent sets it).
    import importlib.metadata, capytaine as cpt
    from capytaine.bem.airy_waves import froude_krylov_force
    from pathlib import Path as P
    if importlib.metadata.version('capytaine').split('.')[:2] != ['3', '0']:
        raise WaveFieldError('floating family requires Capytaine 3.0')
    request = family_request(vertices, faces, surface_ids, heading=heading, water_depth=water_depth, rho=rho, gravity=gravity,
                             Q0=Q0, cg_world=cg_world)
    if source_profile is not None:
        from .transient_native_profile import bind_profile
        request = bind_profile(request, source_profile)
    v, f, labels, mesh, body = _body_for_family(cpt, vertices, faces, surface_ids, request, omega)
    kw = dict(body=body, omega=omega, water_depth=water_depth, rho=rho, g=gravity)
    A = np.zeros((6, 6)); B = np.zeros((6, 6)); rad = []; prad = []
    hull = np.asarray(body.hull_mask, dtype=bool)
    w = float(omega)
    if donor is not None:
        donor.validate()
        if radiation_key(donor.request) != radiation_key(request) or float(donor.omega) != w or donor.centers is None:
            raise WaveFieldError('donor snapshot is not the same radiation basis (geometry/axes/environment/frequency)')
        probe = cpt.DiffractionProblem(wave_direction=request['heading'], **kw)
        solver = cpt.BEMSolver(green_function=_replay_green_function(cpt, donor, probe))
    else:
        if source_profile is not None:
            from .transient_native_profile import solver_for_profile
            solver = solver_for_profile(cpt, source_profile)
        else:
            solver = cpt.BEMSolver()
    def _verify(problem):
        # native problem initialisation may clip/reorder: the SOLVED body must still equal the manifest; lids unsupported
        pm = problem.body.mesh
        if not (np.array_equal(np.asarray(pm.vertices), v) and np.array_equal(np.asarray(pm.faces), f)):
            raise WaveFieldError('solved problem geometry/order differs from the manifest (native clipping/reordering)')
        if 'transient_source_profile' in request:
            from .transient_native_profile import verify_profile_body
            verify_profile_body(problem.body, request)
        elif problem.body.mesh_including_lid.nb_faces != pm.nb_faces:
            raise WaveFieldError('lid profiles are unsupported: the source manifest carries no lid geometry/mask')
        return problem
    if donor is not None:
        rad = [np.array(s) for s in donor.radiation_sources]
        prad = [np.array(p) for p in donor.panel_pressure_radiation]
        A, B = np.array(donor.added_mass), np.array(donor.damping)
    for j, name in enumerate(DOF_NAMES if donor is None else ()):
        res = solver.solve(_verify(cpt.RadiationProblem(radiating_dof=name, **kw)), method='indirect', keep_details=True,
                           n_threads=1)
        if getattr(res, 'sources', None) is None or getattr(res, 'potential', None) is None:
            raise WaveFieldError('backend did not retain radiation sources/potential')
        rad.append(np.asarray(res.sources))
        pot = np.asarray(res.potential)
        prad.append(1j * w * rho * (pot[hull] if pot.shape[0] == hull.shape[0] else pot))
        for i, dof in enumerate(DOF_NAMES):
            A[i, j] = res.added_mass[dof]; B[i, j] = res.radiation_damping[dof]
    dprob = _verify(cpt.DiffractionProblem(wave_direction=request['heading'], **kw))
    dres = solver.solve(dprob, method='indirect', keep_details=True, n_threads=1)
    fk = froude_krylov_force(dprob)
    from capytaine.bem.airy_waves import airy_waves_pressure
    dpot = np.asarray(dres.potential)
    centers = np.asarray(body.mesh.faces_centers, dtype=float)
    p_id = 1j * w * rho * (dpot[hull] if dpot.shape[0] == hull.shape[0] else dpot) + np.asarray(airy_waves_pressure(centers, dprob))
    F = np.array([fk[d] + dres.forces[d] for d in DOF_NAMES], dtype=np.complex128)
    gf = solver.engine.green_function
    settings = {k: val for k, val in gf.exportable_settings.items() if k != 'green_function'}
    provider = dict(backend='capytaine', version=cpt.__version__, path=str(P(cpt.__file__).resolve()), method='indirect',
                    recipe=RECIPE, engine=type(solver.engine).__name__, green_function=gf.exportable_settings['green_function'])
    prony = None; kh = 0.; method = ''
    if donor is not None:                       # the replayed realised state IS the donor's retained state
        provider, settings, prony, kh, method = donor.provider, donor.gf_settings, donor.prony, donor.prony_kh, donor.prony_method
    elif math.isfinite(water_depth):
        method = str(settings.get('finite_depth_prony_decomposition_method', 'python')).lower()
        kh = float(dprob.wavenumber * water_depth)
        cached = type(gf).find_best_exponential_decomposition
        before = cached.cache_info().hits if hasattr(cached, 'cache_info') else None
        prony = np.array(gf.find_best_exponential_decomposition(kh), dtype=np.float64)
        if before is None or cached.cache_info().hits <= before:
            raise WaveFieldError('could not capture the realised finite-depth Green-function decomposition')
    return FloatingFamilySnapshot(request, float(omega), v, f, labels, np.array(rad), np.asarray(dres.sources), A, B, F,
                                  provider, prony, kh, method, settings, '', np.array(prad), p_id, centers,
                                  np.asarray(body.mesh.faces_normals, dtype=float),
                                  np.asarray(body.mesh.faces_areas, dtype=float))


def _potentials(snap, points, *, radiation_index=None):
    """Public Capytaine recovery from retained sources (never solves). Returns potential at points."""
    import capytaine as cpt
    from capytaine.bem.airy_waves import airy_waves_potential
    snap.validate(); r = snap.request
    if (snap.provider.get('backend') != 'capytaine' or snap.provider.get('version') != cpt.__version__
            or snap.provider.get('path') != str(Path(cpt.__file__).resolve())):
        raise WaveFieldError('retained sources were produced by a different Capytaine provider; prepare again')
    v, f, labels, mesh, body = _body_for_family(cpt, snap.vertices, snap.faces, snap.surface_ids, r, snap.omega)
    kw = dict(body=body, omega=snap.omega, water_depth=r['water_depth'], rho=r['rho'], g=r['gravity'])
    pts = np.asarray(points, dtype=float)
    if radiation_index is None:
        prob = cpt.DiffractionProblem(wave_direction=r['heading'], **kw)
        res = prob.make_results_container(sources=np.asarray(snap.diffraction_sources))
    else:
        prob = cpt.RadiationProblem(radiating_dof=DOF_NAMES[radiation_index], **kw)
        res = prob.make_results_container(sources=np.asarray(snap.radiation_sources[radiation_index]))
    # replay needs the single-source API shape; reuse the fixed-body replay with a duck-typed snapshot
    class _S:  # minimal view for _replay_green_function
        prony = snap.prony; gf_settings = snap.gf_settings; prony_kh = snap.prony_kh; prony_method = snap.prony_method
        provider = snap.provider
    solver = cpt.BEMSolver(green_function=_replay_green_function(cpt, _S, prob))
    phi = np.asarray(solver.compute_potential(pts, res))
    return phi + np.asarray(airy_waves_potential(pts, prob)) if radiation_index is None else phi


class _NativeRecovery:
    """Job-owned native recovery for ONE retained snapshot: body, problems and replayed Green function are built once (no BEM)."""

    def __init__(self, snap):
        import capytaine as cpt
        snap.validate()
        r = snap.request
        if (snap.provider.get('backend') != 'capytaine' or snap.provider.get('version') != cpt.__version__
                or snap.provider.get('path') != str(Path(cpt.__file__).resolve())):
            raise WaveFieldError('retained sources were produced by a different Capytaine provider; prepare again')
        self.cpt, self.snap = cpt, snap
        _, _, _, _, body = _body_for_family(cpt, snap.vertices, snap.faces, snap.surface_ids, r, snap.omega)
        self.kw = dict(body=body, omega=snap.omega, water_depth=r['water_depth'], rho=r['rho'], g=r['gravity'])
        self.diffraction = cpt.DiffractionProblem(wave_direction=r['heading'], **self.kw)

        class _S:  # minimal view for _replay_green_function
            prony = snap.prony; gf_settings = snap.gf_settings; prony_kh = snap.prony_kh; prony_method = snap.prony_method
            provider = snap.provider
        self.solver = cpt.BEMSolver(green_function=_replay_green_function(cpt, _S, self.diffraction))

    def potential(self, points, radiation_index=None):
        from capytaine.bem.airy_waves import airy_waves_potential
        cpt, snap = self.cpt, self.snap
        pts = np.asarray(points, dtype=float)
        if radiation_index is None:
            prob = self.diffraction
            res = prob.make_results_container(sources=np.asarray(snap.diffraction_sources))
        else:
            prob = cpt.RadiationProblem(radiating_dof=DOF_NAMES[radiation_index], **self.kw)
            res = prob.make_results_container(sources=np.asarray(snap.radiation_sources[radiation_index]))
        phi = np.asarray(self.solver.compute_potential(pts, res))
        return phi + np.asarray(airy_waves_potential(pts, prob)) if radiation_index is None else phi


@dataclass(frozen=True)
class FloatingSurfaceElevations:
    omega: float
    snapshot_id: str
    eta_incident_diffraction: np.ndarray        # (n,) unit amplitude at (x,y,0), fluid side
    eta_radiation: np.ndarray                   # (6, n) per unit displacement


def evaluate_floating_elevations(snap, points, fluid_normals, *, offset=1e-3, output_mask=None, evaluator=None):
    """Free-surface elevations ONLY (the waterline admission needs nothing else): the same z = 0 recovery as
    :func:`evaluate_floating_fields`, WITHOUT the hull-pressure potentials that the elevation never uses.

    Native path: the retained body / problems / replayed Green function are built once per call (job-owned, no stale global).
    An arbitrary ``evaluator`` uses the full-field path, preserving its interleaved hull/surface calls and stateful behavior."""
    if evaluator is not None:
        full = evaluate_floating_fields(snap, points, fluid_normals, offset=offset,
                                        output_mask=output_mask, evaluator=evaluator)
        return FloatingSurfaceElevations(full.omega, full.snapshot_id, full.eta_incident_diffraction,
                                          full.eta_radiation)
    snap.validate()
    p = np.asarray(points, float); n = np.asarray(fluid_normals, float)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape != n.shape or not np.isfinite(p).all() or not np.isfinite(n).all():
        raise WaveFieldError('matching finite N by 3 points and normals required')
    nn = np.linalg.norm(n, axis=1)
    if np.any(nn < 1e-12) or offset < 0:
        raise WaveFieldError('unit normals and non-negative offset required')
    n = n / nn[:, None]
    mask = np.ones(len(p), bool) if output_mask is None else np.asarray(output_mask, bool)
    if mask.shape != (len(p),):
        raise WaveFieldError('mask must have one entry per point')
    ev = evaluator
    if evaluator is None:
        native = _NativeRecovery(snap)
        ev = lambda s, q, k: native.potential(q, k)
    r = snap.request; w = snap.omega
    surf = np.array(p); surf[:, 2] = 0.; surf = surf + offset * n * np.array([1., 1., 0.])
    zero = np.zeros(len(p), complex)

    def at(k):
        out = zero.copy()
        if mask.any():
            out[mask] = np.asarray(ev(snap, surf[mask], k), dtype=complex)
        return out
    eid = 1j * w * at(None) / r['gravity']
    er = np.stack([1j * w * at(k) / r['gravity'] for k in range(6)])
    if not all(np.isfinite(a).all() for a in (eid, er)):
        raise WaveFieldError('non-finite recovered floating elevation')
    return FloatingSurfaceElevations(w, snap.snapshot_id, eid, er)


@dataclass(frozen=True)
class FloatingSurfaceFields:
    omega: float
    snapshot_id: str
    incident_diffraction_pressure: np.ndarray   # (n,) unit amplitude, p=i w rho (phi_I+phi_D)
    radiation_pressure: np.ndarray              # (6, n) per unit displacement, p=i w rho phi_R (NO extra -i w)
    eta_incident_diffraction: np.ndarray        # (n,) at supplied points' (x,y,0)
    eta_radiation: np.ndarray                   # (6, n) per unit displacement


def evaluate_floating_fields(snap, points, fluid_normals, *, offset=1e-3, output_mask=None, evaluator=None):
    """Complex pressures/elevations at target points from the SAME retained family; masks apply to OUTPUT only.

    ``evaluator(snap, pts, radiation_index)`` may replace native recovery (tests).  Points outside ``output_mask``
    return exactly zero (they are not physical hull outputs); lid sources stay in the retained vectors.
    """
    snap.validate()
    p = np.asarray(points, float); n = np.asarray(fluid_normals, float)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape != n.shape or not np.isfinite(p).all() or not np.isfinite(n).all():
        raise WaveFieldError('matching finite N by 3 points and normals required')
    nn = np.linalg.norm(n, axis=1)
    if np.any(nn < 1e-12) or offset < 0:
        raise WaveFieldError('unit normals and non-negative offset required')
    n = n / nn[:, None]
    mask = np.ones(len(p), bool) if output_mask is None else np.asarray(output_mask, bool)
    if mask.shape != (len(p),):
        raise WaveFieldError('mask must have one entry per point')
    ev = evaluator or (lambda s, q, k: _potentials(s, q, radiation_index=k))
    r = snap.request; w = snap.omega
    hull = np.array(p); hull[:, 2] = np.minimum(p[:, 2], 0.) ; hull = hull + offset * n
    surf = np.array(p); surf[:, 2] = 0.; surf = surf + offset * n * np.array([1., 1., 0.])
    zero = np.zeros(len(p), complex)
    def at(points_, k):
        out = zero.copy()
        if mask.any():
            out[mask] = np.asarray(ev(snap, points_[mask], k), dtype=complex)
        return out
    pid = 1j * w * r['rho'] * at(hull, None)
    eid = 1j * w * at(surf, None) / r['gravity']
    pr = np.stack([1j * w * r['rho'] * at(hull, k) for k in range(6)])
    er = np.stack([1j * w * at(surf, k) / r['gravity'] for k in range(6)])
    if not all(np.isfinite(a).all() for a in (pid, eid, pr, er)):
        raise WaveFieldError('non-finite recovered floating field')
    return FloatingSurfaceFields(w, snap.snapshot_id, pid, pr, eid, er)
