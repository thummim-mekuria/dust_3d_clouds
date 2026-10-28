"""A PPP replacement for astrodendro's ppv_catalog.

astrodendro.ppv_catalog assumes the third axis is velocity and wants a spectral
WCS; this cube's third axis is CTYPE3='distance' in pc. Because that axis is a
real spatial coordinate, we can do better than the PPV version: convert every
voxel to Cartesian (x, y, z) in pc, take the density-weighted covariance, and
diagonalize it. The three principal sigmas give a genuine ellipsoid.

SCIMES computes volume as pi*radius**2*v_rms when the column is missing. We
supply 'volume' and 'luminosity' directly so it never falls back to that.
"""
import numpy as np
from astropy.table import Table

# n[cm-3] -> mass assumes the density is of particles of mean molecular weight
# MU (the cube's HISTORY records a factor 2.8 applied to the Edenhofer values).
# Only affects the mass column and the luminosity scale, not the clustering
# geometry -- check it against how you built the cube before quoting masses.
MU = 2.8
M_H_G = 1.6735575e-24
PC_CM = 3.0856775814913673e18
MSUN_G = 1.98892e33


def voxel_centres(header):
    """Return 1-D arrays of (d_pc, b_deg, l_deg) for each pixel along each axis."""
    def axis(n, crval, crpix, cdelt):
        return crval + (np.arange(n) + 1 - crpix) * cdelt

    l = axis(header['NAXIS1'], header['CRVAL1'], header['CRPIX1'], header['CDELT1'])
    b = axis(header['NAXIS2'], header['CRVAL2'], header['CRPIX2'], header['CDELT2'])
    d = axis(header['NAXIS3'], header['CRVAL3'], header['CRPIX3'], header['CDELT3'])
    return d, b, l


def ppp_catalog(dendro, header, verbose=True):
    """Build a SCIMES-ready catalog of PPP structure properties."""
    d_ax, b_ax, l_ax = voxel_centres(header)
    dl = np.abs(np.radians(header['CDELT1']))
    db = np.abs(np.radians(header['CDELT2']))
    dd = np.abs(header['CDELT3'])

    rows = []
    for s in dendro.all_structures:
        k, j, i = s.indices(subtree=True)          # (distance, b, l)
        n = np.asarray(s.values(subtree=True), dtype='f8')

        dpc = d_ax[k]
        brad = np.radians(b_ax[j])
        lrad = np.radians(l_ax[i])

        # Voxels are frustums, not cubes: their volume grows as d**2, and the
        # longitude edge is foreshortened by cos(b). Ignoring this biases every
        # integrated quantity toward the near side of the map.
        dV = (dpc * dl * np.cos(brad)) * (dpc * db) * dd     # pc**3

        cosb = np.cos(brad)
        x = dpc * cosb * np.cos(lrad)
        y = dpc * cosb * np.sin(lrad)
        z = dpc * np.sin(brad)

        w = n * dV                                            # mass-like weight
        wsum = w.sum()
        xyz = np.vstack([x, y, z])
        mean = (xyz * w).sum(axis=1) / wsum
        dxyz = xyz - mean[:, None]
        cov = (dxyz * w) @ dxyz.T / wsum
        # Symmetric by construction; eigh returns ascending eigenvalues.
        evals = np.linalg.eigvalsh(cov)
        evals = np.clip(evals, 0.0, None)
        sig3, sig2, sig1 = np.sqrt(evals)                     # pc, sig1 largest

        # SCIMES' two default criteria, in physical units.
        radius = np.sqrt(sig1 * sig2)                         # pc
        volume = np.pi * radius**2 * sig3                     # pc**3
        luminosity = wsum                                     # cm-3 pc**3

        # Kept in the cube's own wrapped frame (+210..-210), not recovered from
        # x/y -- arctan2 folds back to +-180 and would erase the padding.
        l_mean = (l_ax[i] * w).sum() / wsum
        b_mean = (b_ax[j] * w).sum() / wsum

        rows.append((
            s.idx, s.is_leaf, len(n),
            mean[0], mean[1], mean[2],
            np.hypot(np.hypot(*mean[:2]), mean[2]),
            l_mean, b_mean,
            sig1, sig2, sig3, radius, volume, luminosity,
            luminosity * MU * M_H_G * PC_CM**3 / MSUN_G,
            n.max(), dV.sum(),
        ))
        if verbose and len(rows) % 500 == 0:
            print('  catalogued %d structures' % len(rows), flush=True)

    return Table(
        rows=rows,
        names=('_idx', 'is_leaf', 'npix',
               'x_pc', 'y_pc', 'z_pc', 'dist_pc', 'l_deg', 'b_deg',
               'sig1_pc', 'sig2_pc', 'sig3_pc', 'radius', 'volume', 'luminosity',
               'mass_msun', 'n_peak', 'vol_voxels_pc3'),
    )


def _in_primary_window(l_deg):
    """True for the one copy of each structure that we keep.

    Copies created by the wrap are separated by exactly 360 deg, so a half-open
    360-deg window selects exactly one of every pair. A closed +-180 test does
    not: a cloud centred on the seam has copies at +180.0 and -180.0 and both
    would survive.
    """
    return (l_deg >= -180.0) & (l_deg < 180.0)


def seam_duplicates(catalog):
    """Mask of catalog rows that are wrap duplicates of another row."""
    return ~_in_primary_window(np.asarray(catalog['l_deg'], dtype=float))


def duplicate_trunks(dendro, header):
    """Trunks that are wrap duplicates, to be masked out before recomputing.

    Works at the trunk level because a duplicate is spatially disconnected from
    its original (1572 px apart), so the whole tree is either kept or dropped.
    """
    _, b_ax, l_ax = voxel_centres(header)
    dl = np.abs(np.radians(header['CDELT1']))
    db = np.abs(np.radians(header['CDELT2']))
    dd = np.abs(header['CDELT3'])
    d_ax = voxel_centres(header)[0]

    dupes = []
    for s in dendro.trunk:
        k, j, i = s.indices(subtree=True)
        n = np.asarray(s.values(subtree=True), dtype='f8')
        dpc = d_ax[k]
        brad = np.radians(b_ax[j])
        w = n * (dpc * dl * np.cos(brad)) * (dpc * db) * dd
        l_mean = (l_ax[i] * w).sum() / w.sum()
        if not _in_primary_window(l_mean):
            dupes.append(s)
    return dupes
