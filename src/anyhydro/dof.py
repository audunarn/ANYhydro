"""Body-qualified DOF identity.

Persistent DOF identity is always body-qualified: ``body_id::surge`` etc.
Unqualified names are never used as persistent identity.
"""

from __future__ import annotations

from typing import Tuple

DOF_NAMES: Tuple[str, ...] = ("surge", "sway", "heave", "roll", "pitch", "yaw")
DOF_SEPARATOR = "::"


def qualify_dof(body_id: str, dof: str) -> str:
    """Return the body-qualified DOF key ``body_id::dof``."""
    return f"{body_id}{DOF_SEPARATOR}{dof}"


def parse_dof_key(key: str) -> Tuple[str, str]:
    """Split a qualified DOF key into ``(body_id, dof)``.

    Raises
    ------
    ValueError
        If the key is not a well-formed qualified DOF key.
    """
    if not isinstance(key, str):
        raise ValueError(f"DOF key must be a string, got {type(key)!r}")
    parts = key.split(DOF_SEPARATOR)
    if len(parts) != 2:
        raise ValueError(f"DOF key {key!r} is not body-qualified ('body_id::dof')")
    body_id, dof = parts
    if not body_id:
        raise ValueError(f"DOF key {key!r} has an empty body id")
    if dof not in DOF_NAMES:
        raise ValueError(f"DOF key {key!r} has unknown DOF {dof!r}")
    return body_id, dof


def is_qualified_dof_key(key: str) -> bool:
    try:
        parse_dof_key(key)
    except ValueError:
        return False
    return True


def body_dof_keys(body_id: str) -> Tuple[str, ...]:
    """All six body-qualified DOF keys for ``body_id`` in canonical order."""
    return tuple(qualify_dof(body_id, dof) for dof in DOF_NAMES)


def dof_axis_for_bodies(body_ids) -> Tuple[str, ...]:
    """Full 6N DOF axis for the given bodies, in canonical order."""
    keys: list[str] = []
    for body_id in body_ids:
        keys.extend(body_dof_keys(body_id))
    return tuple(keys)
