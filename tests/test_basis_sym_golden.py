"""Guard: existing (absolute basis, no symmetrisation) recipes/settings keep their hashes bit-identical."""
import hashlib
import json
import math
from pathlib import Path

from anyhydro.vessel_study import KernelRecipe
from test_vessel_study import _recipe, _kernel

GOLDEN = Path(__file__).parent / "data" / "golden_hashes_basis_sym.json"


def _current():
    k = _kernel()
    s = k.settings()
    return {"kernel_recipe_dict": k.to_dict(),
            "recipe_hash": _recipe().content_hash(),
            "settings_dict_sha256": hashlib.sha256(json.dumps(s.to_dict(), sort_keys=True).encode()).hexdigest(),
            "settings_dict_keys": sorted(s.to_dict())}


def test_golden_hashes_unchanged():
    assert _current() == json.loads(GOLDEN.read_text())


def test_absolute_recipe_has_no_new_keys():
    d = _kernel().to_dict()
    assert "tolerance_basis" not in d and "kernel_symmetrisation" not in d
