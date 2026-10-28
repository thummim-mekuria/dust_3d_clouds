"""Load the native Edenhofer cube, convert to density, wrap-pad in longitude.

Replaces the pre-baked edenhofer_lbd_wrap.fits, which carried only MEAN. The
native file has both MEAN and STD. as separate ImageHDUs, so padding here keeps
the uncertainty available for signal-to-noise cuts.

The l axis spans exactly 360.000 deg in 1572 px, so the padding is a true
circular wrap: np.pad(mode='wrap') and CRPIX1 += npad, nothing else to correct.
30 deg is exactly 131 px, which reproduces the npad=131 of the old file.
"""
import os

import numpy as np
from astropy.io import fits

# Anchored to this file, not the cwd: the sbatch runs from slurm/ but the
# notebooks run from code_and_data/, and a relative default breaks one of them.
CUBE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../../../data/maps/edenhofer_lbd.fits'))

# dust_3d_functions.A_to_n = lambda A: ((A*u.mag/u.pc)*(2.2e21*u.cm**-2/u.mag)).cgs
# applied to (edenhofer * 2.8). It is linear in A, so the identical scalar
# converts the STD. extension into a 1-sigma density uncertainty.
PC_CM = 3.0856775814913673e18
A_TO_N = 2.8 * 2.2e21 / PC_CM          # ~1996 cm-3 per (mag/pc)


def npad_for(header, pad_deg):
    """Padding in pixels, checked against an exactly-360-deg longitude axis."""
    span = header['NAXIS1'] * abs(header['CDELT1'])
    if not np.isclose(span, 360.0, atol=1e-6):
        raise ValueError('l axis spans %.6f deg, not 360; wrap is not circular' % span)
    npad = pad_deg / abs(header['CDELT1'])
    if not np.isclose(npad, round(npad), atol=1e-6):
        raise ValueError('%g deg is %.4f px, not a whole number' % (pad_deg, npad))
    return int(round(npad))


def padded_header(header, npad):
    h = header.copy()
    h['NAXIS1'] = header['NAXIS1'] + 2 * npad
    h['CRPIX1'] = header['CRPIX1'] + npad
    h['BUNIT'] = 'cm-3'
    h.add_history('GLON wrapped +-%.4g deg past +-180 seam (npad=%d)'
                  % (npad * abs(header['CDELT1']), npad))
    h.add_history('values n[cm-3] = A_to_n(ednhfr*2.8); density, not extinction')
    return h


def header_only(path=CUBE, pad_deg=30.0, ext='MEAN'):
    """The padded WCS header, without touching the ~6 GB of pixel data."""
    with fits.open(path, memmap=True) as h:
        hdr = h[ext].header
        return padded_header(hdr, npad_for(hdr, pad_deg))


def _load_one(hdu, npad, verbose):
    # Native byte order matters: the file is big-endian ('>f4') and every
    # downstream numpy op on it runs through a slow path.
    arr = np.asarray(hdu.data, dtype=np.float32)
    arr *= A_TO_N
    arr = np.pad(arr, ((0, 0), (0, 0), (npad, npad)), mode='wrap')
    if verbose:
        print('    -> %s %s %.2f GiB' % (arr.shape, arr.dtype, arr.nbytes / 2**30),
              flush=True)
    return arr


def load_padded(path=CUBE, pad_deg=30.0, with_std=True, verbose=True):
    """Return (mean, std, header) in cm-3, wrap-padded. std is None if not wanted.

    Peak memory is about 20 GiB with std, 13 GiB without: np.pad allocates the
    padded output while the unpadded input is still alive.
    """
    with fits.open(path, memmap=True) as h:
        npad = npad_for(h['MEAN'].header, pad_deg)
        if verbose:
            print('padding %d px (%.4g deg) each side of the l seam'
                  % (npad, npad * abs(h['MEAN'].header['CDELT1'])), flush=True)
            print('  MEAN', flush=True)
        mean = _load_one(h['MEAN'], npad, verbose)

        std = None
        if with_std:
            if verbose:
                print('  STD.', flush=True)
            std = _load_one(h['STD.'], npad, verbose)

        hdr = padded_header(h['MEAN'].header, npad)

    return mean, std, hdr
