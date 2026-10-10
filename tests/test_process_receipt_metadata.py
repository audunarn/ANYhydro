# SPDX-License-Identifier: MPL-2.0
"""Stored-output backend fixture verifies optional supervision metadata plumbing."""
from test_hams_backend import case,run,receipt

def test_optional_supervision_metadata_is_recorded(case,monkeypatch):
    import anyhydro.hams_backend as backend
    original=backend.OwnedProcess
    expected=dict(format='test-private-supervision-receipt',guard_pid=123,producer_pid=456)
    def supervised(*args,**kwargs):
        process=original(*args,**kwargs)
        process.supervision=expected
        return process
    monkeypatch.setattr(backend,'OwnedProcess',supervised)
    run(case)
    assert receipt(case)['supervision']==expected
