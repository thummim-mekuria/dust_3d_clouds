"""Stage 2: cluster the dendrogram into clouds with SCIMES, write an assignment cube.

Cheap compared to stage 1 -- rerun this with different --user-k or --criteria
without recomputing the dendrogram.
"""
import argparse
import os
import time

import numpy as np
from astropy.io import fits
from astrodendro import Dendrogram

import cube_io
from ppp_catalog import ppp_catalog, seam_duplicates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dendro', required=True, help='.hdf5 from dendro_full.py')
    p.add_argument('--cube', default=None,
                   help='only needed if the .hdr sidecar is missing')
    p.add_argument('--pad-deg', type=float, default=30.0)
    p.add_argument('--out', required=True, help='output prefix')
    p.add_argument('--criteria', default='volume,luminosity')
    p.add_argument('--user-k', type=int, default=None,
                   help='number of clusters; SCIMES estimates it if unset')
    p.add_argument('--rms', type=float, default=None,
                   help='noise in cm-3 for the S/N cut; defaults to DENDRMS '
                        '(median STD over significant voxels) from stage 1')
    p.add_argument('--s2nlim', type=float, default=3.0)
    p.add_argument('--catalog-only', action='store_true',
                   help='write the PPP catalog and stop, without running SCIMES')
    args = p.parse_args()

    t0 = time.time()
    d = Dendrogram.load_from(args.dendro)

    hdr_path = os.path.splitext(args.dendro)[0] + '.hdr'
    if os.path.exists(hdr_path):
        header = fits.Header.fromfile(hdr_path)
    else:
        # Falls back to rebuilding the padded WCS; reads no pixel data.
        header = cube_io.header_only(args.cube or cube_io.CUBE, pad_deg=args.pad_deg)
    print('loaded %d structures (%d leaves) in %.0fs'
          % (len(d), len(d.leaves), time.time() - t0), flush=True)

    rms = args.rms
    if rms is None and np.isfinite(header.get('DENDRMS', np.nan)):
        rms = float(header['DENDRMS'])
        print('using DENDRMS = %.4g cm-3 from stage 1' % rms, flush=True)

    t0 = time.time()
    cat = ppp_catalog(d, header)
    # After stage 1's trunk-level dedup these are no longer duplicates: they are
    # sub-structures of clouds that genuinely straddle the seam, which is the
    # reason the padding exists. Only meaningful as duplicates if stage 1 ran
    # with --keep-seam-duplicates.
    cat['in_padding'] = seam_duplicates(cat)
    print('catalog: %d structures in %.0fs (%d centred in the padding)'
          % (len(cat), time.time() - t0, cat['in_padding'].sum()), flush=True)
    cat.write(args.out + '_dendrocat.fits', overwrite=True)

    if args.catalog_only:
        return

    # Imported late: scimes pulls in sklearn, and a broken install should not
    # cost you the catalog above.
    from scimes import SpectralCloudstering

    kwargs = dict(
        dendrogram=d,
        catalog=cat,
        header=header,
        criteria=args.criteria.split(','),
        blind=True,
    )
    if args.user_k is not None:
        kwargs['user_k'] = args.user_k
    if rms is not None:
        kwargs['rms'] = rms
        kwargs['s2nlim'] = args.s2nlim

    t0 = time.time()
    res = SpectralCloudstering(**kwargs)
    print('SCIMES: %d clusters in %.0fs' % (len(res.clusters), time.time() - t0), flush=True)

    np.savetxt(args.out + '_cluster_idx.txt', np.asarray(res.clusters), fmt='%d')

    asgn = getattr(res, 'clusters_asgn', None)
    if asgn is not None:
        data = asgn.data if hasattr(asgn, 'data') else asgn
        fits.PrimaryHDU(np.asarray(data, dtype='i4'), header).writeto(
            args.out + '_asgn.fits', overwrite=True)
        print('wrote %s_asgn.fits' % args.out, flush=True)
    else:
        print('no clusters_asgn on the result; cluster indices written instead',
              flush=True)

    print('wrote %s_dendrocat.fits, %s_cluster_idx.txt'
          % (args.out, args.out), flush=True)


if __name__ == '__main__':
    main()
