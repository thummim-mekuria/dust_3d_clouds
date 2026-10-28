"""Stage 1: compute an astrodendro dendrogram on the full Edenhofer l/b/d cube.

Reads the native two-extension cube and does the +-30 deg wrap padding here, so
MEAN and STD. stay together and the threshold can be signal-to-noise rather than
a flat density floor.

This is the expensive, single-threaded, memory-hungry step -- it is separated
from SCIMES so the dendrogram is computed once and clustered many times.

Padded shape is (1182, 786, 1834) float32 = 6.8 GiB per extension. At
min_value = 2.9 cm-3 about 1.7e7 of 1.7e9 voxels are significant.
"""
import argparse
import os
import time

import numpy as np
from astrodendro import Dendrogram

import cube_io
import ppp_catalog


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cube', default=cube_io.CUBE)
    p.add_argument('--out', required=True, help='output .hdf5 for the dendrogram')
    p.add_argument('--pad-deg', type=float, default=30.0,
                   help='longitude wrap padding each side of the +-180 seam')
    # p99 = 2.86, p99.5 = 4.54, p99.9 = 10.79 cm-3
    p.add_argument('--min-value', type=float, default=2.9,
                   help='density floor in cm-3; voxels below are ignored')
    p.add_argument('--min-delta', type=float, default=1.0,
                   help='contrast in cm-3 a peak needs to survive as its own leaf')
    p.add_argument('--min-npix', type=int, default=100,
                   help='minimum voxels per structure (a voxel is ~1.2x1.2x1 pc at 300 pc)')
    # Median per-voxel S/N above 2.9 cm-3 is only ~2.2, so a cut at 3 discards
    # ~91% of the signal and one at 2 discards ~33%. Off by default; the STD is
    # still read so the noise level lands in the sidecar header either way.
    p.add_argument('--min-snr', type=float, default=None,
                   help='additionally require MEAN/STD above this (default: no cut)')
    p.add_argument('--no-std', action='store_true',
                   help='skip the STD. extension entirely, saving ~7 GiB')
    p.add_argument('--keep-seam-duplicates', action='store_true',
                   help='leave wrap duplicates in the dendrogram (default: mask '
                        'and recompute, which costs a second pass)')
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    t0 = time.time()
    data, std, header = cube_io.load_padded(
        args.cube, pad_deg=args.pad_deg, with_std=not args.no_std)
    print('loaded and padded in %.0fs' % (time.time() - t0), flush=True)

    sig = data > args.min_value
    n_sig = int(sig.sum())
    print('%d voxels above %.3g cm-3 (%.3f%% of %d)'
          % (n_sig, args.min_value, 100 * sig.mean(), data.size), flush=True)

    rms_med = np.nan
    if std is not None:
        rms_med = float(np.median(std[sig]))
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = data[sig] / std[sig]
        print('median STD over those voxels: %.4g cm-3 (median S/N %.2f)'
              % (rms_med, np.nanmedian(snr)), flush=True)
        del snr

        if args.min_snr is not None:
            # astrodendro ignores NaN, which is how the cut is applied. Scaling
            # std in place avoids a third full-size (6.8 GiB) temporary.
            std *= args.min_snr
            lowsnr = ~(data > std)
            n_cut = int((sig & lowsnr).sum())
            data[lowsnr] = np.nan
            print('S/N < %g removed %d of the %d significant voxels (%.1f%%)'
                  % (args.min_snr, n_cut, n_sig, 100 * n_cut / max(n_sig, 1)),
                  flush=True)
            del lowsnr
        del std
    del sig

    def compute(arr):
        t = time.time()
        dd_ = Dendrogram.compute(arr, min_value=args.min_value,
                                 min_delta=args.min_delta,
                                 min_npix=args.min_npix, verbose=True)
        print('dendrogram: %d structures (%d leaves) in %.0fs'
              % (len(dd_), len(dd_.leaves), time.time() - t), flush=True)
        return dd_

    d = compute(data)

    # SCIMES indexes the catalog positionally against the dendrogram, so wrap
    # duplicates cannot simply be dropped from the catalog, and prune() only
    # merges leaves upward -- it will not delete a whole trunk. Masking the
    # duplicate trunks and recomputing is the way to keep them out of the
    # affinity matrix entirely.
    if args.keep_seam_duplicates or args.pad_deg == 0:
        print('keeping wrap duplicates', flush=True)
    else:
        dupes = ppp_catalog.duplicate_trunks(d, header)
        n_vox = 0
        for s in dupes:
            idx = s.indices(subtree=True)
            n_vox += len(idx[0])
            data[idx] = np.nan
        print('masking %d duplicate trunks (%d voxels) and recomputing'
              % (len(dupes), n_vox), flush=True)
        if dupes:
            d = compute(data)

    d.save_to(args.out)

    # Sidecar header so stage 2 gets the padded WCS without re-reading 12 GB.
    header['DENDMINV'] = (args.min_value, 'astrodendro min_value [cm-3]')
    header['DENDMIND'] = (args.min_delta, 'astrodendro min_delta [cm-3]')
    header['DENDMINP'] = (args.min_npix, 'astrodendro min_npix')
    if args.min_snr is not None:
        header['DENDSNR'] = (args.min_snr, 'minimum MEAN/STD required')
    header['DENDRMS'] = (rms_med, 'median STD over significant voxels [cm-3]')
    header.tofile(os.path.splitext(args.out)[0] + '.hdr', overwrite=True)
    print('wrote %s' % args.out, flush=True)


if __name__ == '__main__':
    main()
