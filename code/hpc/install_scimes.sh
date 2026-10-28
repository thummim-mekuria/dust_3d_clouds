#!/bin/bash
# Install scimes and repair it for modern sklearn/python. Idempotent -- safe to
# re-run. Run on the login node inside the activated conda env, not in a job.
set -euo pipefail

# scimes declares its dependency as "sklearn", the dead shim package that has
# errored on install since Dec 2023, so a plain `pip install scimes` fails.
pip install --no-deps scimes
pip install scikit-learn networkx astrodendro h5py

python - <<'PY'
import os, scimes
d = os.path.dirname(scimes.__file__)

fixes = [
    # Python-2 style implicit relative import; never worked on python 3.
    (os.path.join(d, 'scimes.py'),
     'from old_spectral_embedding import spectral_embedding',
     'from .old_spectral_embedding import spectral_embedding'),
    # sklearn dropped its lobpcg re-export; it has always lived in scipy.
    (os.path.join(d, 'old_spectral_embedding.py'),
     'from sklearn.utils.fixes import lobpcg',
     'from scipy.sparse.linalg import lobpcg  # sklearn dropped its shim'),
]

for path, old, new in fixes:
    s = open(path).read()
    if new in s:
        print('already patched:', os.path.basename(path))
    elif old in s:
        if not os.path.exists(path + '.orig'):
            open(path + '.orig', 'w').write(s)
        open(path, 'w').write(s.replace(old, new))
        print('patched:', os.path.basename(path))
    else:
        raise SystemExit('NEITHER old nor new text found in %s -- scimes version '
                         'differs from the one this was written for' % path)
PY

python -c "from scimes import SpectralCloudstering; print('scimes imports OK')"
