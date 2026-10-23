"""Bind resume and case reuse to every executed implementation and source input."""
from pathlib import Path
import json

from foundation import sha256


def canonical_record(value):
    return json.loads(json.dumps(value, allow_nan=False))


def implementation_hashes(new, reused):
    return {f'{label}/{p.name}': sha256(p)
            for label, directory in [('feedback', Path(new)), ('reused', Path(reused))]
            for p in sorted(directory.glob('*.py'))}


def input_digest(paths):
    files = sorted({Path(p).resolve() for p in paths}, key=str)
    if not all(p.is_file() for p in files):
        raise ValueError('Required evaluation input is missing')
    return {str(p): sha256(p) for p in files}
