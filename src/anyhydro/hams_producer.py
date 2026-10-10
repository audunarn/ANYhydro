# SPDX-License-Identifier: MPL-2.0
"""Explicit unqualified corrected-HAMS build declarations; never execute."""
from dataclasses import dataclass
import hashlib,json,os,sys
from pathlib import Path
from .errors import CapabilityError,HydroError
from .hams_reference import HAMS_SOURCE_COMMIT

SOURCE_IDENTITIES=frozenset((
    '04e9de2c1b7b20d2c7cb27de635dd86f983bc81d390b7906b64f66089a44cbfd',
    'e795b30b418d7b4eb230c2726c2f2d3d766b2e6e7a0426a6c5830322a426c183',
))

LIMIT_SOURCE_IDENTITY='a2608dc7bbb56176917b1fc13804e77412ef5aefce26740a8366556217ad5940'
GRAVITY_SOURCE_IDENTITY='077b0ec28d35339763cb2d4aaf079ee7da335919fe5f0486aa035faefd602a88'
ALL_SOURCE_IDENTITIES=SOURCE_IDENTITIES | frozenset((LIMIT_SOURCE_IDENTITY,GRAVITY_SOURCE_IDENTITY))

class HamsProducerError(HydroError):pass


@dataclass(frozen=True)
class HamsProducerBuild:
    executable_path: Path
    executable_sha256: str
    source_tree_sha256: str
    platform: str
    build_description: str
    version: str='2.2-phase-corrected-development'
    source_commit: str=HAMS_SOURCE_COMMIT

    def __post_init__(self):
        if not isinstance(self.source_tree_sha256,str) or self.source_commit!=HAMS_SOURCE_COMMIT or self.source_tree_sha256 not in ALL_SOURCE_IDENTITIES:
            raise HamsProducerError('Unknown or uncorrected compiled source profile')
        value=self.executable_sha256
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise HamsProducerError('Explicit executable SHA256 required')
        if self.platform not in ('windows','linux') or any(not isinstance(x,str) or not x.strip() for x in (self.build_description,self.version)):
            raise HamsProducerError('Explicit native platform/version/build description required')
        try:object.__setattr__(self,'executable_path',Path(self.executable_path).resolve())
        except (TypeError,ValueError,OSError) as exc:raise HamsProducerError('Explicit executable path required') from exc

    def identity(self):
        return dict(format='hams-corrected-build-v1',executable_sha256=self.executable_sha256,
            source_tree_sha256=self.source_tree_sha256,source_commit=self.source_commit,
            platform=self.platform,version=self.version,build_description=self.build_description,qualified=False)

    def content_hash(self):
        return hashlib.sha256(json.dumps(self.identity(),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def verify(self,*,require_native=False):
        try:
            raw=self.executable_path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=self.executable_sha256:raise HamsProducerError('Executable bytes changed')
            if self.platform=='windows' and not raw.startswith(b'MZ') or self.platform=='linux' and not raw.startswith(b'\x7fELF'):
                raise HamsProducerError('Executable format differs from declared native platform')
        except OSError as exc:raise HamsProducerError('Producer executable missing or unreadable') from exc
        if require_native and self.platform!=('windows' if os.name=='nt' else 'linux' if sys.platform.startswith('linux') else ''):
            raise CapabilityError('Configured producer requires a different native platform; no implicit WSL/shell launcher')
        return self.identity()

    @classmethod
    def from_identity(cls,identity,*,executable_path):
        if not isinstance(identity,dict) or set(identity)!={'format','executable_sha256','source_tree_sha256','source_commit','platform','version','build_description','qualified'}:
            raise HamsProducerError('Complete build identity required')
        if identity['format']!='hams-corrected-build-v1' or identity['qualified'] is not False:
            raise HamsProducerError('Unknown build identity or qualification assertion')
        return cls(executable_path,identity['executable_sha256'],identity['source_tree_sha256'],identity['platform'],identity['build_description'],identity['version'],identity['source_commit'])
