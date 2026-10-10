from dataclasses import replace
import numpy as np
import pytest
from anyhydro.fixtures import make_two_body_fixture
from anyhydro.radiation import RadiationSettings, radiation_kernel_from_dataset, RadiationError, ReconstructionCriteria, compare_reconstruction

@pytest.fixture
def kernel():
    ds=make_two_body_fixture()
    settings=RadiationSettings(dt=.05,n_time=41,symmetry_tol=1e-10,passivity_tol=1e-10,
                              zero_frequency_damping=np.zeros((12,12)))
    return radiation_kernel_from_dataset(ds,settings)

def test_repeated_zero_queries_preserve_shape(kernel):
    a,b=kernel.reconstruct([0.,0.,.7])
    assert a.shape==(3,12,12) and b.shape==a.shape
    np.testing.assert_array_equal(a[0],a[1])

def test_artifact_time_and_settings_agree(kernel):
    with pytest.raises(RadiationError):
        replace(kernel, settings=replace(kernel.settings,dt=.1))

def test_arrays_cannot_reenable_mutation(kernel):
    for array in [kernel.kernel,kernel.time,kernel.added_mass_infinite,
                  kernel.settings.zero_frequency_damping,
                  kernel.diagnostics.frequency_grid]:
        with pytest.raises(ValueError): array.setflags(write=True)

def test_overflowing_reconstruction_refused(kernel):
    with pytest.raises(RadiationError): kernel.reconstruct([1e308])

def test_settings_refuse_overflowing_duration():
    with pytest.raises(RadiationError):
        RadiationSettings(dt=1e308,n_time=1001,symmetry_tol=0,passivity_tol=0,
                          zero_frequency_damping=np.zeros((12,12)))

def test_reconstruction_decision_arrays_cannot_be_mutated():
    ds=make_two_body_fixture()
    settings=RadiationSettings(dt=.05,n_time=41,symmetry_tol=1e-10,passivity_tol=1e-10,
                              zero_frequency_damping=np.zeros((12,12)))
    artifact=radiation_kernel_from_dataset(ds,settings)
    report=compare_reconstruction(artifact,ds,ReconstructionCriteria(.3,2.2,0,0,0,0))
    for array in [report.added_mass_pass,report.damping_pass,report.added_mass_max_error,
                  report.damping_max_error,report.added_mass_threshold,report.damping_threshold]:
        with pytest.raises(ValueError): array.setflags(write=True)
