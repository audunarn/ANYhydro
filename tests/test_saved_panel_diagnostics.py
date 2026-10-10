# SPDX-License-Identifier: MPL-2.0
"""Saved-evidence diagnostics contracts: validation and refusal, not BEM truth.

The synthetic evidence set is produced end-to-end by the production caller with
the mock tabulated backend, then pinned byte-for-byte. Corruption, wrong-model
and refusal tests are independent of any numerical truth of the mock values.
"""
import hashlib
import json
import shutil
import sys
import warnings

import numpy as np
import pytest

from numpy.exceptions import ComplexWarning

from anyhydro.errors import CapabilityError
from anyhydro.panel_accuracy_reference import FrozenTabulatedProfile
from anyhydro.saved_panel_diagnostics import (SAVED_DIAGNOSTICS_FORMAT, _signed_stats,
                                              run_saved_panel_diagnostics)
from test_panel_accuracy_reference import build_bundle, build_table_backend
from test_panel_integration_candidate import call

SAVED_NAMES = ('case.json', 'input.npz', 'native-operator-probe.npz', 'accuracy-comparison.json',
               'accuracy-comparison.npz', 'accuracy-input.npz', 'accuracy-native-micro.npz',
               'accuracy-reference.npz', 'accuracy-profile.json', 'frozen-tabulation.npz',
               'tabulation-receipt.json', 'status.json')


def build_saved_evidence(tmp_path, monkeypatch):
    root, manifest, green = build_table_backend(build_bundle(tmp_path, monkeypatch), monkeypatch)
    call(tmp_path/'saved-evidence', accuracy_profile=FrozenTabulatedProfile.load(root), operator_probe_receivers=2)
    candidate = tmp_path/'saved-evidence'/'candidate'
    inputs = tmp_path/'inputs'
    inputs.mkdir()
    for name in SAVED_NAMES:
        shutil.copy(candidate/name, inputs/name)
    bundle_dir = inputs/'reference-bundle'
    bundle_dir.mkdir()
    for name in ('manifest.json', 'input.npz', 'reference.npz'):
        shutil.copy(root/name, bundle_dir/name)
    identities = {}
    for path in sorted(inputs.rglob('*')):
        if path.is_file():
            identities[path.relative_to(inputs).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    identities_path = tmp_path/'input-identities.json'
    identities_path.write_text(json.dumps(identities), encoding='utf-8')
    return inputs, identities_path, green


def reseal(inputs, identities_path, name):
    identities = json.loads(identities_path.read_text())
    identities[name] = hashlib.sha256((inputs/name).read_bytes()).hexdigest()
    identities_path.write_text(json.dumps(identities), encoding='utf-8')


def test_saved_diagnostics_complete_headless_run(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'capytaine', None)
    inputs, identities_path, green = build_saved_evidence(tmp_path, monkeypatch)
    report = run_saved_panel_diagnostics(inputs, tmp_path/'out', identities_path)
    assert report['format'] == SAVED_DIAGNOSTICS_FORMAT and report['state'] == 'completed'
    assert report['native_calls'] == 0 and not report['qualified'] and not report['new_oracle']
    assert report['validation']['status_state'] == 'completed_probe_unqualified'
    imaginary = report['imaginary_pole']
    assert imaginary['acceptance_threshold'] is None and not imaginary['retroactive_imaginary_acceptance']
    assert imaginary['source_quadrature_nodes'] == 4*imaginary['source_panels']
    assert report['fd_defects']['unique_cause_claimed'] is False
    assert report['fd_defects']['original_steps'] == [1e-5, 5e-6]
    assert report['coincident']['receivers'] == 2
    assert report['service_wall_seconds'] == report['process_wall_seconds']
    assert report['process_wall_seconds_is_legacy_service_alias'] is True
    assert report['wall_scope'] == 'service_invocation_only; process_wall_from_supervisor_receipt'
    for name in ('report.json', 'imaginary-pole-comparison.npz', 'fd-signed-defects.npz',
                 'coincident-inventory.json'):
        assert (tmp_path/'out'/name).is_file()
    with np.load(tmp_path/'out'/'imaginary-pole-comparison.npz') as z:
        assert z['difference'].shape == (2, 24) and z['gradient_difference'].shape == (3, 2, 24)
        np.testing.assert_array_equal(z['difference'], z['pole']-z['native_imag'])
    with np.load(tmp_path/'out'/'fd-signed-defects.npz') as z:
        assert z['native_minus_reference_fd'].shape == (7, 3)
        np.testing.assert_array_equal(z['native_minus_reference_fd'], z['native_fd']-z['reference_fd'])
    inventory = json.loads((tmp_path/'out'/'coincident-inventory.json').read_text())
    assert inventory['totals']['receivers'] == 2
    assert not (tmp_path/'out'/'failure-report.json').exists()


def test_missing_pinned_file_refused_with_failure_evidence(tmp_path, monkeypatch):
    inputs, identities_path, green = build_saved_evidence(tmp_path, monkeypatch)
    (inputs/'native-operator-probe.npz').unlink()
    with pytest.raises(CapabilityError, match='bounded regular file'):
        run_saved_panel_diagnostics(inputs, tmp_path/'out', identities_path)
    failure = json.loads((tmp_path/'out'/'failure-report.json').read_text())
    assert failure['state'] == 'failed' and failure['error_type'] == 'CapabilityError'
    assert failure['completed_stages'] == ['identities_loaded']


def test_corrupted_pinned_byte_refused_before_parsing(tmp_path, monkeypatch):
    inputs, identities_path, green = build_saved_evidence(tmp_path, monkeypatch)
    data = bytearray((inputs/'input.npz').read_bytes())
    data[-1] ^= 0xff
    (inputs/'input.npz').write_bytes(bytes(data))
    with pytest.raises(CapabilityError, match='hash mismatch'):
        run_saved_panel_diagnostics(inputs, tmp_path/'out', identities_path)


def test_wrong_model_snapshot_refused_after_reseal(tmp_path, monkeypatch):
    inputs, identities_path, green = build_saved_evidence(tmp_path, monkeypatch)
    case = json.loads((inputs/'case.json').read_text())
    case['snapshot_arrays']['vertices']['sha256'] = '0'*64
    (inputs/'case.json').write_text(json.dumps(case, indent=2), encoding='utf-8')
    reseal(inputs, identities_path, 'case.json')
    with pytest.raises(CapabilityError, match='producing case snapshot'):
        run_saved_panel_diagnostics(inputs, tmp_path/'out', identities_path)


def test_case_hash_tamper_refused(tmp_path, monkeypatch):
    inputs, identities_path, green = build_saved_evidence(tmp_path, monkeypatch)
    case = json.loads((inputs/'case.json').read_text())
    case['case_hash'] = '0'*64
    (inputs/'case.json').write_text(json.dumps(case, indent=2), encoding='utf-8')
    reseal(inputs, identities_path, 'case.json')
    with pytest.raises(CapabilityError, match='case hash mismatch'):
        run_saved_panel_diagnostics(inputs, tmp_path/'out', identities_path)


def test_nonfinite_probe_refused_after_reseal(tmp_path, monkeypatch):
    inputs, identities_path, green = build_saved_evidence(tmp_path, monkeypatch)
    with np.load(inputs/'native-operator-probe.npz') as z:
        arrays = {name: z[name] for name in z.files}
    arrays['s'][0, 0] = np.nan
    np.savez(inputs/'native-operator-probe.npz', **arrays)
    reseal(inputs, identities_path, 'native-operator-probe.npz')
    with pytest.raises(CapabilityError, match='finite'):
        run_saved_panel_diagnostics(inputs, tmp_path/'out', identities_path)


def test_inconsistent_receipt_refused(tmp_path, monkeypatch):
    inputs, identities_path, green = build_saved_evidence(tmp_path, monkeypatch)
    receipt = json.loads((inputs/'tabulation-receipt.json').read_text())
    receipt['actual_tables']['tabulated_r_range']['sha256'] = '0'*64
    (inputs/'tabulation-receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    reseal(inputs, identities_path, 'tabulation-receipt.json')
    with pytest.raises(CapabilityError, match='inconsistent'):
        run_saved_panel_diagnostics(inputs, tmp_path/'out', identities_path)


def test_terminal_status_mismatch_refused(tmp_path, monkeypatch):
    inputs, identities_path, green = build_saved_evidence(tmp_path, monkeypatch)
    status = json.loads((inputs/'status.json').read_text())
    status['state'] = 'completed_unqualified'
    (inputs/'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    reseal(inputs, identities_path, 'status.json')
    with pytest.raises(CapabilityError, match='terminal probe'):
        run_saved_panel_diagnostics(inputs, tmp_path/'out', identities_path)


def test_existing_output_directory_refused(tmp_path, monkeypatch):
    inputs, identities_path, green = build_saved_evidence(tmp_path, monkeypatch)
    (tmp_path/'out').mkdir()
    with pytest.raises(FileExistsError):
        run_saved_panel_diagnostics(inputs, tmp_path/'out', identities_path)


def test_signed_stats_complex_phase_metrics_real_and_componentwise():
    values = np.array([1.0 + 2.0j, -3.0 + 0.5j, 0.0 - 1.5j, 2.0 + 2.0j])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ComplexWarning)
        stats = _signed_stats(values)
    magnitudes = np.abs(values)
    assert isinstance(stats['rms'], float) and isinstance(stats['l2'], float)
    assert stats['rms'] == pytest.approx(float(np.sqrt(np.mean(magnitudes**2))))
    assert stats['l2'] == pytest.approx(float(np.linalg.norm(values)))
    assert stats['max_abs'] == pytest.approx(float(magnitudes.max()))
    assert (stats['signed_min_real'], stats['signed_max_real']) == (-3.0, 2.0)
    assert (stats['signed_min_imag'], stats['signed_max_imag']) == (-1.5, 2.0)
    assert 'signed_min' not in stats and 'signed_max' not in stats
    real_stats = _signed_stats(np.array([-2.0, 1.0, 3.0]))
    assert real_stats['signed_min'] == -2.0 and real_stats['signed_max'] == 3.0
    assert 'signed_min_real' not in real_stats and 'signed_min_imag' not in real_stats
