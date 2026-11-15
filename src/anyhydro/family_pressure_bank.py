"""Pressure banks from a retained floating field family (no solve, no Capytaine, no Qt at import).

Reads the per-frequency hull panel pressures a coherent family cache already holds
(``floating_radiation_fields.recover_family_snapshot``) and composes an ``ImportedPressureBank``.

Conventions (all inherited from the family record, phasor ``exp(-i omega t)``, SI):
- ``panel_pressure_incident_diffraction``: fluid pressure of the unit-amplitude Airy wave plus the fixed-body
  diffraction field, fixed world phase origin.
- ``panel_pressure_radiation[k]``: fluid pressure per unit displacement of mean-body generalized coordinate ``k``
  (``p = i omega rho phi``, no extra ``-i omega``).
- ``fixed_diffraction``: ``p = p_inc+dif``.
- ``total_with_source_motion``: ``p = p_inc+dif + sum_k xi_k * p_rad_k`` where ``xi`` is the complex motion RAO
  (unit wave amplitude, mean body axes through the mean CG, ``x(t)=Re[xi exp(-i omega t)]``) supplied by the caller.
  RAOs come from ANYhydro-side coefficients (see ``linear_rao``) or any other explicit source; nothing is assumed.
Normals are outward from the hull; the pressure is the fluid pressure, so the load on the structure is
``-p n a`` and the generalized force in the mean body axes is ``-W^T p`` (``W`` rows ``[Q^T n, (c-cg)x n Q] a``).
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

COMPOSITIONS = ('fixed_diffraction', 'total_with_source_motion')


class FamilyPressureBankError(ValueError):
    pass


def linear_rao(omega, added_mass, damping, excitation, mass, hydrostatic_stiffness, extra_stiffness=None):
    """Frequency-domain RAO ``xi`` with ``(-w^2 (M+A) - i w B + C) xi = F`` (``exp(-i w t)``, mean body axes)."""
    w = float(omega)
    M, A, B, C = (np.asarray(x, float) for x in (mass, added_mass, damping, hydrostatic_stiffness))
    if any(x.shape != (6, 6) for x in (M, A, B, C)) or not math.isfinite(w) or w <= 0:
        raise FamilyPressureBankError('linear_rao needs 6x6 mass, added mass, damping, stiffness and omega>0')
    if extra_stiffness is not None:
        C = C + np.asarray(extra_stiffness, float)
    xi = np.linalg.solve(-w * w * (M + A) - 1j * w * B + C, np.asarray(excitation, complex))
    if not np.isfinite(xi).all():
        raise FamilyPressureBankError('non-finite RAO')
    return xi


def _rao_matrix(rao_source, omegas):
    if callable(rao_source):
        rao = np.array([np.asarray(rao_source(float(w)), complex) for w in omegas])
    else:
        rao = np.asarray(rao_source, complex)
    if rao.shape != (len(omegas), 6) or not np.isfinite(rao).all():
        raise FamilyPressureBankError('RAO source must give a finite (n_omega, 6) complex array')
    return rao


def snapshot_wrench_operator(snap):
    """``W`` (n_faces, 6): generalized-force weights so that ``-W.T @ p`` is the force/moment in mean body axes."""
    Q = np.asarray(snap.request['Q0'], float)
    cg = np.asarray(snap.request['cg_world'], float)
    return np.column_stack([snap.normals @ Q, np.cross(snap.centers - cg, snap.normals) @ Q]) * snap.areas[:, None]


def floating_family_to_pressure_bank(cache, request, omegas, rao_source=None, *, body_id=0, composition=None,
                                     input_files=None):
    """Compose an ``ImportedPressureBank`` from the retained family at ``omegas`` (strictly increasing).

    ``input_files`` (optional ``{role: {'name':..., 'sha256':...}}``) is recorded verbatim in the provenance.
    ``rao_source=None`` gives ``fixed_diffraction``; otherwise an ``(n_omega, 6)`` complex array or a callable
    ``omega -> (6,)`` gives ``total_with_source_motion``.  ``composition`` may only confirm that choice.
    Missing frequencies are refused (never solved).
    """
    from .floating_radiation_fields import recover_family_snapshot, validate_family_coherence
    from .orcawave_pressure import ImportedPressureBank

    omegas = np.asarray(omegas, float)
    if omegas.ndim != 1 or not len(omegas) or not np.isfinite(omegas).all() or np.any(np.diff(omegas) <= 0):
        raise FamilyPressureBankError('strictly increasing finite omegas required')
    expected = 'fixed_diffraction' if rao_source is None else 'total_with_source_motion'
    if composition not in (None, expected):
        raise FamilyPressureBankError(f'composition {composition!r} contradicts the RAO source ({expected!r})')
    rao = None if rao_source is None else _rao_matrix(rao_source, omegas)
    snaps = [recover_family_snapshot(cache, request, float(w)) for w in omegas]
    for s in snaps:
        s.validate()
        if s.panel_pressure_incident_diffraction is None:
            raise FamilyPressureBankError('family record carries no native hull panel pressures')
    validate_family_coherence(snaps)
    first = snaps[0]
    for s in snaps[1:]:
        for name in ('vertices', 'faces', 'centers', 'normals', 'areas'):
            if not np.array_equal(getattr(s, name), getattr(first, name)):
                raise FamilyPressureBankError(f'family geometry differs between frequencies ({name})')
    rows = []
    for k, s in enumerate(snaps):
        p = np.array(s.panel_pressure_incident_diffraction, complex)
        if rao is not None:
            p = p + rao[k] @ s.panel_pressure_radiation
        rows.append(p)
    n = len(first.faces)
    heading = float(first.request['heading']) % math.tau
    provenance = dict(
        source='anyhydro_floating_family_cache', recipe=first.request['recipe'], composition=expected,
        snapshot_ids=[s.snapshot_id for s in snaps],
        rao_sha256=None if rao is None else hashlib.sha256(np.ascontiguousarray(rao).tobytes()).hexdigest(),
        rao_note=None if rao is None else 'supplied by caller; x(t)=Re[xi exp(-i w t)], mean body axes through mean CG',
        input_files=input_files, composition_formula=('p_inc+dif' if rao is None else 'p_inc+dif + sum_k xi_k p_rad_k'),
        conventions=dict(first.request['conventions']), generalized_axes=dict(Q0=first.request['Q0'], cg_world=first.request['cg_world']))
    return ImportedPressureBank(
        np.array(first.vertices), np.array(first.faces), np.array(first.centers), np.array(first.normals),
        np.array(first.areas), omegas, np.array(rows), tuple(range(n)), tuple(f'body:{body_id}' for _ in range(n)),
        heading, expected, int(body_id), float(first.request['water_depth']), float(first.request['rho']),
        float(first.request['gravity']), json.dumps(provenance, sort_keys=True))
