"""Development coarse-source factory: canonical kernel algebra, retained (never passed) failed reports, unchanged caps.

Reuses the analytic exponential-memory family of test_floating_transient_fields (RETAINED snapshots + native
coefficients, injected elevation evaluator).  That fixture holds 600 frequencies, above the retained 513-frequency
coarse cap, so every test here uses the canonical every-second-sample sub-grid (300 frequencies) consistently for the
snapshots, the reference kernel and the admission grid; the cap itself is checked separately.
Manufactured data only: nothing here is a native qualification.
"""
import dataclasses

import numpy as np
import pytest

from anyhydro.floating_transient_prepare import (
    DEVELOPMENT_MARKER,
    DevelopmentSourceError,
    build_development_coarse_source,
)
from anyhydro.radiation import _kernel_from_arrays
from anyhydro.radiation_field_kernel import FieldReconstructionCriteria, RadiationFieldError, RadiationFieldLimitError

try:
    from tests.test_floating_transient_fields import CRIT, LAM, _zero, fam  # noqa: F401  (fixture)
except ImportError:  # pragma: no cover - rootdir without package
    from test_floating_transient_fields import CRIT, LAM, _zero, fam  # noqa: F401

STRICT = FieldReconstructionCriteria(0.5, 4.0, 1e-14, 0.0)
SUB = slice(None, None, 2)


def _snaps(fam):
    return tuple(fam["snaps"][SUB])


def _dev(fam, snaps=None, criteria=CRIT, settings=None):
    return build_development_coarse_source(
        _snaps(fam) if snaps is None else snaps, fam["native"](), fam["pts"],
        settings=fam["kernel"].settings if settings is None else settings,
        pressure_criteria=criteria, elevation_criteria=criteria,
        pressure_zero_frequency=_zero(fam, "p"), elevation_zero_frequency=_zero(fam, "e"),
        evaluator=fam["evaluator"](), chunk_size=4)


def test_development_source_uses_the_canonical_kernel_and_is_always_unverified(fam):
    src = _dev(fam)
    snaps = _snaps(fam)
    ref = _kernel_from_arrays(omegas=fam["w"][SUB], added_mass=np.stack([s.added_mass for s in snaps]),
                              damping=np.stack([s.damping for s in snaps]), a_inf=fam["A_inf"],
                              dof_keys=fam["kernel"].dof_keys, body_ids=("body",), settings=fam["kernel"].settings,
                              provenance={"source": "reference"})
    # same canonical transform of the same raw damping and the native A_inf
    np.testing.assert_array_equal(src.kernel.kernel, ref.kernel)
    np.testing.assert_array_equal(src.kernel.added_mass_infinite, fam["A_inf"])
    assert src.hydrodynamic_verification == "unverified" and src.marker == DEVELOPMENT_MARKER
    assert src.unpassed_gates[0] == "development_source_not_admitted"
    assert src.field_kernel.identity["development"] == "unverified_hydrodynamics"
    np.testing.assert_array_equal(src.admission_omegas, fam["w"][SUB])
    b = src.solver_binding()
    assert b["hydro_admission"] == "development_unverified" and b["field_kernel_hash"] == src.field_kernel.content_hash()
    # family pose and query points are bound from the ACTUAL retained request / native object
    assert src.identity["family_Q0"] == snaps[0].request["Q0"]
    assert src.identity["query_points_world"] == np.asarray(fam["pts"], float).tolist()
    import json
    json.dumps(src.identity_dict(), allow_nan=False); json.dumps(src.quality_dict(), allow_nan=False)
    json.dumps(src.solver_binding(), allow_nan=False)                 # fully JSON-serialisable provenance
    assert len(src.identity["family_geometry_digest"]) == 64


def test_more_than_513_frequencies_are_refused_not_raised(fam):
    with pytest.raises(RadiationFieldLimitError, match="513"):
        _dev(fam, snaps=fam["snaps"])


def test_failed_reconstruction_is_retained_not_passed_and_accepted_path_still_refuses(fam):
    from anyhydro.floating_transient_fields import build_family_radiation_field_kernel
    snaps = _snaps(fam)
    kernel = _dev(fam).kernel
    with pytest.raises(RadiationFieldError, match="reconstruction failed"):
        build_family_radiation_field_kernel(
            snaps, fam["native"](), fam["pts"], kernel=kernel, pressure_criteria=STRICT,
            elevation_criteria=STRICT, pressure_zero_frequency=_zero(fam, "p"),
            elevation_zero_frequency=_zero(fam, "e"), evaluator=fam["evaluator"](), chunk_size=4)
    src = _dev(fam, criteria=STRICT)
    assert "pressure_transfer_reconstruction" in src.unpassed_gates
    rec = src.source_quality["field_reconstruction"]["pressure"]
    assert rec["passed"] is False


def test_damping_changed_without_its_pressure_is_refused_by_the_unchanged_projection(fam):
    # incoherent manufactured source: B changes, the retained panel pressure does not -> Proj P_v != -K_r (refused,
    # never retained as a development gate: the signed projection is not relaxed)
    skew = np.zeros((6, 6)); skew[0, 1], skew[1, 0] = 50.0, -50.0
    i = 6
    snaps = list(_snaps(fam))
    snaps[i // 2] = fam["snapshot"](i, damping=np.asarray(fam["snaps"][i].damping) + skew)
    with pytest.raises(RadiationFieldError):
        _dev(fam, snaps=snaps)


def test_coherent_asymmetric_source_is_retained_as_an_unpassed_gate(fam):
    # coherent skew family: B += S lam/(lam^2+w^2) at EVERY frequency and the matching panel pressure
    # dMv = -pinv(Proj) S (Proj dMv = -S), with the explicit zero-frequency samples updated consistently
    S =np.zeros((6, 6)); S[0, 1], S[1, 0] = 50.0, -50.0
    dMv = -np.linalg.pinv(fam["proj"]) @ S
    w = fam["w"]
    snaps = []
    for i in range(0, w.size, 2):
        wi = w[i]
        d_hp = dMv * (wi ** 2 - 1j * wi * LAM) / (LAM ** 2 + wi ** 2)
        snaps.append(fam["snapshot"](i, damping=np.asarray(fam["snaps"][i].damping) + S * LAM / (LAM ** 2 + wi ** 2),
                                     panel_pressure_radiation=(fam["hp"][i] + d_hp).T.copy()))
    settings = dataclasses.replace(fam["kernel"].settings, zero_frequency_damping=(fam["D"] + S) / LAM)
    src = build_development_coarse_source(
        snaps, fam["native"](), fam["pts"], settings=settings, pressure_criteria=CRIT, elevation_criteria=CRIT,
        pressure_zero_frequency=(fam["Mv"] + dMv) / LAM, elevation_zero_frequency=_zero(fam, "e"),
        evaluator=fam["evaluator"](), chunk_size=4)
    assert "source_damping_symmetry" in src.unpassed_gates
    assert src.source_quality["damping_symmetry_violations"] >= 1
    # raw data kept: the canonical diagnostics record the violation, nothing symmetrised
    assert src.source_quality["max_damping_asymmetry"] > 0.0


def test_source_tolerances_can_only_tighten(fam):
    loose = dataclasses.replace(fam["kernel"].settings, symmetry_tol=1e-2)
    with pytest.raises(DevelopmentSourceError, match="only tighten"):
        _dev(fam, settings=loose)
