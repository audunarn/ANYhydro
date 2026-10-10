# Shared materialized sea states (development)

`RegularWave`, `JonswapSpectrum`, `SpectralBins`, `DirectionalSpectrum` and
`WaveRealization` are public immutable data records. New headings use propagation
radians measured from +X toward +Y. The existing AiryComponent degree field is an
explicit adapter. A realization supplies the same components to
`realization.environment(water_depth)` and ANYhydro excitation synthesis.

The JONSWAP shape follows the [Orcina wave-spectrum definition](https://www.orcina.com/webhelp/OrcaFlex/Content/html/Waves%2CWavespectra.htm)
and [parameter definitions](https://www.orcina.com/webhelp/OrcaFlex/Content/html/Environment%2CDataforJONSWAPandISSCspectra.htm).
Angular density is `Sω=Sf(ω/(2π))/(2π)` in m²/(rad/s). `discretize` explicitly
renormalizes the declared finite, nonoverlapping bins to
`sum(Sω Δω)=Hs²/16`. This does not validate a continuous tail or the chosen grid.
Directional weights are discrete energy fractions summing to one; no directional
spreading function is inferred.

```python
from anyloads import JonswapSpectrum, DirectionalSpectrum, WaveRealization
waves = WaveRealization.from_jonswap(
    JonswapSpectrum(significant_height=2., peak_period=8.),
    omegas=(.7, 1.1, 1.6), bandwidths=(.2, .2, .2), seed=17,
    directions=DirectionalSpectrum((0.,), (1.,)))
saved = waves.to_dict()
restored = WaveRealization.from_dict(saved)
environment = restored.environment(water_depth=20.)
```

Amplitudes are `sqrt(2 Sω Δω direction_weight)`. PCG64 creates phases once.
Saved densities, widths, headings, amplitudes and phases are authoritative;
deserialization validates them and never regenerates phases. The seed records
generation provenance, not permission to replace saved phases. SHA256 covers
the complete named record. Loading accepts data only, with no provider code.

ANYhydro's optional `sea-state` extra supplies `build_excitation_series`:

```python
from anyhydro import build_excitation_series, UNIT_INCIDENT_AMPLITUDE
series = build_excitation_series(
    dataset, restored, amplitude_normalization=UNIT_INCIDENT_AMPLITUDE,
    phase_origin_xy=(0., 0.))
forces = series.evaluate([0., .2, .4])  # rows: time; columns: series.dof_keys
```

The development conversion uses
`F(t)=sum(a_j Re{X_j exp(i phase_j) exp(-i omega_j t)})`. It requires an explicit
unit incident elevation amplitude declaration and global (0,0) phase origin.
These declarations do not establish that a backend produced compatible data.
Only GLOBAL coefficient axes are supported. Complex coefficients interpolate
linearly inside the stored frequency range; extrapolation is refused. Headings
must match modulo 2π within 1e-12 rad conversion roundoff, with no directional
interpolation or ambiguous duplicates. Full body-qualified force DOF identity
is preserved. The output retains source status, dataset hash and wave hash;
`verify_excitation_binding` reconstructs it against the authoritative inputs.

This slice does not qualify a BEM source, large excursions, diffraction, radiation,
current modifications to wave dispersion, infinite depth, or a native transient
integration. Water depth, gravity and phase-origin compatibility with a physical
source require a separate admission contract. Scientific W3/W4 acceptance and
the full ANYmarine demonstrator remain pending.
