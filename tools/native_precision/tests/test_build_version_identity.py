# SPDX-License-Identifier: MPL-2.0
"""Focused admission tests for generated backend version metadata."""
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'build_isolated_backend.py'
spec=importlib.util.spec_from_file_location('version_identity',path)
backend=importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)

@pytest.mark.parametrize('source', [
 '__version__ = "3.0.0"\n',
 "__version__ = '3.0.0'\n",
 '__version__ = "3.0.0"\nif __name__ == "__main__":\n    raise RuntimeError("must not execute")\n',
])
def test_generated_version_metadata_parses_without_execution(source):
 assert backend.validate_built_version(source)=='3.0.0'

@pytest.mark.parametrize('source', [
 None,
 '# __version__ = "3.0.0"\n',
 '__version__ = "3.0.1"\n',
 '__version__ = "3.0.0"\n__version__ = "3.0.1"\n',
 '__version__ = str("3.0.0")\n',
])
def test_ambiguous_missing_changed_or_executable_versions_refused(source):
 with pytest.raises((backend.BuildError,ValueError)):
  backend.validate_built_version(source)
