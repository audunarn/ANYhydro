"""Development excitation synthesis from explicit saved wave components.

No backend or ANYloads import occurs until the builder is called. Declarations
of normalization and phase origin are caller contracts, not source qualification.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import re
import numpy as np

from .dof import parse_dof_key
from .errors import HydroError
from .frames import require_global_dof_axes
from .hashing import dataset_content_hash
from .schema import validate_hydro_dataset, QUALIFICATION_STATUSES

EXCITATION_SERIES_VERSION = 'hydro_excitation_series_development_v1'
UNIT_INCIDENT_AMPLITUDE = 'unit_incident_elevation_amplitude_m'


class ExcitationSeriesError(HydroError):
    """Invalid or incompatible development excitation declaration."""


def _array(value, name, ndim):
    try:
        if any(isinstance(x,(bool,np.bool_)) for x in np.asarray(value,dtype=object).flat):
            raise ValueError()
        raw = np.asarray(value)
        if raw.ndim != ndim or not raw.size or raw.dtype.kind not in 'iuf':
            raise ValueError()
        a = np.asarray(raw, dtype=float)
        if not np.isfinite(a).all():
            raise ValueError()
        # Bytes-backed storage cannot be made writable through setflags.
        return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)
    except (TypeError, ValueError, OverflowError) as error:
        raise ExcitationSeriesError(f'{name} requires finite real data of rank {ndim}') from error


def _declarations(normalization, origin):
    if normalization != UNIT_INCIDENT_AMPLITUDE:
        raise ExcitationSeriesError('explicit unit incident elevation amplitude normalization required')
    xy = _array(origin, 'phase origin', 1)
    if xy.shape != (2,) or np.any(xy != 0.):
        raise ExcitationSeriesError('only declared global (0,0) phase origin is supported')
    return (0., 0.)


@dataclass(frozen=True, eq=False)
class ExcitationSeries:
    """F(t)=Re{Σ weighted_X[j] exp(-i omega[j] t)} on full stored DOF axes."""
    dof_keys: tuple
    omegas: np.ndarray
    weighted_real: np.ndarray
    weighted_imag: np.ndarray
    wave_realization_hash: str
    hydro_content_hash: str
    source_qualification_status: str
    amplitude_normalization: str
    phase_origin_xy: tuple

    def __post_init__(self):
        try:
            keys = tuple(self.dof_keys)
            if not keys or len(set(keys)) != len(keys):
                raise ValueError()
            for key in keys:
                parse_dof_key(key)
        except (TypeError, ValueError) as error:
            raise ExcitationSeriesError('unique body-qualified DOF keys required') from error
        w = _array(self.omegas, 'omega', 1)
        real = _array(self.weighted_real, 'weighted real excitation', 2)
        imag = _array(self.weighted_imag, 'weighted imaginary excitation', 2)
        if np.any(w <= 0.) or real.shape != (len(w),len(keys)) or imag.shape != real.shape:
            raise ExcitationSeriesError('positive frequencies and matching term/DOF matrices required')
        for key in ('wave_realization_hash','hydro_content_hash'):
            value = getattr(self,key)
            if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None:
                raise ExcitationSeriesError('canonical source and wave SHA256 hashes required')
        if self.source_qualification_status not in QUALIFICATION_STATUSES:
            raise ExcitationSeriesError('unknown source qualification status')
        origin = _declarations(self.amplitude_normalization,self.phase_origin_xy)
        for key,value in [('dof_keys',keys),('omegas',w),('weighted_real',real),
                          ('weighted_imag',imag),('phase_origin_xy',origin)]:
            object.__setattr__(self,key,value)

    def evaluate(self, times):
        times = _array(times,'times',1)
        values = np.empty((len(times),len(self.dof_keys)))
        weights = self.weighted_real+1j*self.weighted_imag
        try:
            with np.errstate(over='raise',invalid='raise'):
                for start in range(0,len(times),2048):
                    phase = -times[start:start+2048,None]*self.omegas
                    values[start:start+2048] = (np.exp(1j*phase)@weights).real
        except FloatingPointError as error:
            raise ExcitationSeriesError('time/force synthesis exceeds the finite supported range') from error
        if not np.isfinite(values).all():
            raise ExcitationSeriesError('nonfinite synthesized force')
        return values

    def to_dict(self):
        if type(self) is not ExcitationSeries:
            raise ExcitationSeriesError('serialization cannot erase subclass semantics')
        return dict(api_version=EXCITATION_SERIES_VERSION, dof_keys=list(self.dof_keys),
            omegas=self.omegas.tolist(),weighted_real=self.weighted_real.tolist(),
            weighted_imag=self.weighted_imag.tolist(),wave_realization_hash=self.wave_realization_hash,
            hydro_content_hash=self.hydro_content_hash,source_qualification_status=self.source_qualification_status,
            amplitude_normalization=self.amplitude_normalization,phase_origin_xy=list(self.phase_origin_xy))

    @classmethod
    def from_dict(cls, data):
        fields = {'api_version','dof_keys','omegas','weighted_real','weighted_imag',
            'wave_realization_hash','hydro_content_hash','source_qualification_status',
            'amplitude_normalization','phase_origin_xy'}
        if type(data) is not dict or set(data) != fields or data['api_version'] != EXCITATION_SERIES_VERSION:
            raise ExcitationSeriesError('excitation series must match its exact named schema')
        return cls(**{k:v for k,v in data.items() if k != 'api_version'})

    @property
    def content_hash(self):
        return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True,separators=(',',':'),
            allow_nan=False).encode('utf-8')).hexdigest()


def build_excitation_series(dataset, realization, *, amplitude_normalization, phase_origin_xy):
    """Interpolate complex coefficients inside the source range; never extrapolate.

    Headings match modulo 2pi within 1e-12 rad conversion roundoff. Directional
    interpolation, reference-origin translation and non-GLOBAL axes are refused.
    """
    origin = _declarations(amplitude_normalization,phase_origin_xy)
    try:
        from anyloads import WaveRealization
    except ImportError as error:
        raise ExcitationSeriesError('ANYloads sea-state extra is required for synthesis') from error
    if type(realization) is not WaveRealization:
        raise ExcitationSeriesError('exact named WaveRealization required')
    validate_hydro_dataset(dataset)
    require_global_dof_axes(dataset)
    w = np.asarray(dataset.omega.values,dtype=float)
    headings = np.asarray(dataset.wave_direction.values,dtype=float)
    distances = np.abs(np.remainder(headings[:,None]-headings[None,:]+math.pi,math.tau)-math.pi)
    if np.any(np.triu(distances <= 1e-12,k=1)):
        raise ExcitationSeriesError('ambiguous duplicate source headings modulo 2pi')
    coefficients = dataset.excitation_force.transpose('omega','wave_direction','influenced_dof').values
    weights = []
    for c in realization.components:
        if not w[0] <= c.omega <= w[-1]:
            raise ExcitationSeriesError('wave frequency outside stored coefficient range')
        heading = math.radians(c.direction_degrees)
        distance = np.abs(np.remainder(headings-heading+math.pi,math.tau)-math.pi)
        matches = np.flatnonzero(distance <= 1e-12)
        if len(matches) != 1:
            raise ExcitationSeriesError('wave heading requires one exact source heading; no interpolation')
        upper = min(int(np.searchsorted(w,c.omega,side='right')),len(w)-1)
        lower = max(upper-1,0)
        f = 0. if upper == lower else (c.omega-w[lower])/(w[upper]-w[lower])
        x = (1.-f)*coefficients[lower,matches[0]]+f*coefficients[upper,matches[0]]
        weights.append(c.amplitude*np.exp(1j*c.phase)*x)
    weights = np.asarray(weights)
    return ExcitationSeries(tuple(str(x) for x in dataset.influenced_dof.values),
        [c.omega for c in realization.components],weights.real,weights.imag,
        realization.content_hash,dataset_content_hash(dataset),dataset.attrs['qualification_status'],
        amplitude_normalization,origin)


def verify_excitation_binding(series, dataset, realization):
    """Reconstruct against authoritative inputs; deserialization alone proves no binding."""
    if type(series) is not ExcitationSeries:
        raise ExcitationSeriesError('named ExcitationSeries required')
    expected = build_excitation_series(dataset,realization,
        amplitude_normalization=series.amplitude_normalization,phase_origin_xy=series.phase_origin_xy)
    if series.content_hash != expected.content_hash:
        raise ExcitationSeriesError('excitation series does not match authoritative source/wave inputs')
    return True
