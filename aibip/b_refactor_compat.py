"""Narrow read-only compatibility bridge for the B/C move from d3c780e.

No artifact is rewritten and no content checksum is bypassed. Old code hashes
are accepted only for the explicitly recorded before/after source bundle.
Any later edit to that bundle disables legacy acceptance. New caches always
record actual current hashes, including this helper and its registry.
"""
import hashlib
import itertools
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REGISTRY_NAME = 'b_refactor_registry.json'


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def compatibility_fingerprint():
    return _digest((PACKAGE/'b_refactor_compat.py').read_bytes()
                   + (PACKAGE/REGISTRY_NAME).read_bytes())


def _approved_registry():
    """LF/CRLF is irrelevant to the approval gate, not to artifact hashes."""
    try:
        registry = json.loads((PACKAGE/REGISTRY_NAME).read_text(encoding='utf-8'))
        if registry['baseline_commit'] != 'd3c780e':
            return None
        for name, expected in registry['approved_lf_sha256'].items():
            actual = _digest((PACKAGE/name).read_bytes().replace(b'\r\n', b'\n'))
            if actual != expected:
                return None
        return registry
    except (OSError, ValueError, KeyError):
        return None


def compatible_code_hashes(recorded, current):
    if recorded == current:
        return True
    registry = _approved_registry()
    if registry is None or 'b_refactor_compat' in recorded:
        return False
    if current.get('b_refactor_compat') != compatibility_fingerprint():
        return False
    if set(recorded) != set(current)-{'b_refactor_compat'}:
        return False
    for name, value in recorded.items():
        if value == current[name]:
            continue
        if value not in registry['legacy_sha256'].get(name, []):
            return False
    return True


def compatible_run_keys(run_key, settings):
    """Enumerate only approved old code identities; keep every runtime setting.

    Includes known LF, CRLF and the original mixed checkout's identities.
    Source/weight/config changes still change the run key. Old sidecars remain
    byte-identical, preserving patch manifests and checkpoint provenance.
    """
    keys = {run_key}
    registry = _approved_registry()
    code = settings.get('code_sha256', {})
    names = ['c_test_core.py', 'c_transfer.py', 'part_c.py', 'part_b.py']
    if (registry is None or set(code) != set(names) | {'b_refactor_compat'}
            or code['b_refactor_compat'] != compatibility_fingerprint()):
        return keys
    for hashes in itertools.product(*(registry['legacy_sha256'][name] for name in names)):
        legacy = dict(settings, code_sha256=dict(zip(names, hashes)))
        keys.add(_digest(json.dumps(legacy, sort_keys=True).encode()))
    return keys


def compatible_plan(recorded, current):
    """Resume an unchanged extraction plan without editing its provenance."""
    if recorded == current:
        return True
    if set(recorded) != set(current) or 'code_sha256' not in recorded:
        return False
    return (all(recorded[k] == current[k] for k in recorded if k != 'code_sha256')
            and compatible_code_hashes(recorded['code_sha256'], current['code_sha256']))
