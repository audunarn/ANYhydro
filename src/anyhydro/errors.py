"""Typed error hierarchy for anyhydro."""

from __future__ import annotations


class HydroError(Exception):
    """Base class for all anyhydro errors."""


class RecordValidationError(HydroError):
    """A hydrodynamic record (mesh, body, request) failed validation."""


class SchemaError(HydroError):
    """A hydrodynamic dataset failed canonical schema validation."""


class CapabilityError(HydroError):
    """A requested capability is not available or not supported."""


class UnsupportedOperationError(CapabilityError):
    """The operation is explicitly unsupported in this slice."""


class TimeDomainAdmissionError(CapabilityError):
    """A dataset is not admissible to the time-domain route."""


class RequestMismatchError(HydroError):
    """A request does not match the stored dataset identity."""


class BodyIdentityMismatchError(HydroError):
    """Body identities or mesh hashes do not match the stored dataset."""


class TransformError(HydroError):
    """Invalid transformation input."""


class PersistenceError(HydroError):
    """A persistence operation failed."""


class GeometryValidationError(HydroError):
    """A body's wetted geometry failed the backend's admission checks."""


class BackendSolveError(HydroError):
    """A backend solve failed or returned invalid data.

    No physical replacement coefficients are ever invented: invalid or
    missing solver output is a typed failure, not a substitution.
    """
