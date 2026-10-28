"""
dust3d_lib -- every calculation and plotting function of the dust_3d cloud
pipeline in one module. dust3d_pipeline.py is the driver that calls them.

Consolidated on 2026-10-21 from the scripts in code_and_data/ (now code/). Function bodies
are copied verbatim from those scripts; the only edits are
  * module prefixes removed (D. = dust_3d_functions, M. =
    choosing_distances_from_healpix, SKY. / XYZ. = the projection scripts);
  * names that clashed between scripts made unique:
        plot_sky_projections.build_maps / prepare / out_of_plane
            -> sky_build_maps / sky_prepare / sky_out_of_plane
        plot_xyz_projections.build_maps / prepare / out_of_plane
            -> xyz_build_maps / xyz_prepare / xyz_out_of_plane
        plot_xyz_projections.XYZ / K / L / PAD
            -> XYZ_CUBE / XYZ_K / XYZ_SPINE_L / XYZ_PAD
        plot_av1_aperture.plot_cloud / LEGEND -> aperture_plot_cloud / APERTURE_LEGEND
        plot_av1_contours.plot_cloud / legend -> contours_plot_cloud / contours_legend
        plot_saddle_windows.plot_one / LEGEND -> plot_saddle_one / SADDLE_LEGEND
        plot_leak_niso.COLORS -> LEAK_COLORS
        plot_sample_spine_windows.plot_windows -> plot_sample_spine_windows
        plot_inclination_panels.angles -> inclination_angles
  * every path made absolute (they were relative to code_and_data/), so the
    module works from any directory;
  * each script's main() rewritten as a function taking its command-line
    options as arguments (last section of this file).

The old scripts were retired on 2026-10-28. Their functions that the
pipeline never calls (the skin-mesh code, the pre-HEALPix leaf selection,
spine_angles, cloud_masses, the iNICEST fetcher, ...) are kept in the LEGACY
section at the end of this file.

Module-level state a caller may set:
    FIG_LOC    where sample_summary saves its projmaps figures
"""

import os
import time
import pickle
import functools
import warnings

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import astropy.units as u
import astropy.constants as c

from scipy.optimize import minimize, curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import map_coordinates, binary_erosion
from scipy import ndimage as ndi

from joblib import Parallel, delayed
from astropy_healpix import HEALPix, bilinear_interpolation_weights
from astrodendro import Dendrogram

plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size']   = 15
plt.rcParams['mathtext.fontset'] = 'stix'


# plot_rho0_correlations.py and plot_spine_inclinations.py never imported
# dust_3d_functions, so they drew with matplotlib's default fonts rather than
# the STIX settings above; this keeps their figures as they were.
_DEFAULT_FONTS = {k: matplotlib.rcParamsDefault[k]
                  for k in ('font.family', 'font.size', 'mathtext.fontset')}


def _default_fonts(fn):
    @functools.wraps(fn)
    def wrapped(*a, **kw):
        with plt.rc_context(_DEFAULT_FONTS):
            return fn(*a, **kw)
    return wrapped


# ============================================================================
# Paths (absolute, so the module works from any directory)
# ============================================================================
HERE     = os.path.dirname(os.path.abspath(__file__))           # .../dust_3d/github_repo/code
PROJECT  = os.path.dirname(HERE)                                # .../dust_3d/github_repo
RESEARCH = os.path.dirname(os.path.dirname(PROJECT))            # .../Documents/research

INPUTS   = os.path.join(PROJECT, 'data') + '/'                  # small input tables (in git)
LOCAL    = os.path.join(os.path.dirname(PROJECT), 'local_only') + '/'   # .../dust_3d/local_only, outside the repo
RUNS     = LOCAL + 'runs/'                                      # pipeline output, one dir per run

DATA       = os.path.join(RESEARCH, 'data', 'maps') + '/'
CAT        = os.path.join(RESEARCH, 'data', 'catalogs', 'mi_tab1.csv')
YSO_CAT    = os.path.join(RESEARCH, 'data', 'catalogs', 'kyso_rgeo_le500pc.fits')
YSO_ALL    = os.path.join(RESEARCH, 'data', 'maps', 'marton_ysos.fits')
FIGURES    = os.path.join(PROJECT, 'figures') + '/'           # the paper's figures only
FIG_LOC    = RUNS
NICEST_DIR = LOCAL + 'nicest_maps/zucker2020_fwhm0.229deg/nicest'   # fetch with inicest_main()
WRAP       = DATA + 'edenhofer_lbd_wrap.fits'
XYZ_CUBE   = DATA + 'edenhofer_xyz.fits'
IFB_DEFAULT = INPUTS + 'imara_forbes_2023_fig4fig5_digitized.csv'
IFB_FIG7    = INPUTS + 'imara_forbes_2023_fig7_digitized.csv'

# caches and the stored per-cloud sample cubes (env vars as before)
CACHE    = os.environ.get('HPX_CACHE', LOCAL + 'cache/healpix_dustcubes.npz')
SCACHE   = os.environ.get('HPX_SCACHE', LOCAL + 'cache/healpix_sample_profiles.npz')
# Per-cloud padded cubes for every posterior sample, written by
# extract_padded_sample_cubes(). Everything below (mean cubes, padded boxes,
# sample cubes, sample profiles) is derived from these files.
CUBE_DIR = os.environ.get('HPX_CUBES', LOCAL + 'cache/sample_cubes_pad8')
PAD_STORED = 8.0     # deg of padding the stored cubes carry; no box can grow past it

# columns of spine_angles_and_fits_samples_n<N>.csv
FIT_COLS = ['cloud', 'win_lo', 'win_hi', 'M', 'l', 'l_sd', 'b', 'b_sd', 'd', 'd_sd',
            'pa_lb', 'pa_lb_sd', 'inc_ld', 'inc_ld_sd', 'inc_bd', 'inc_bd_sd',
            'rho0', 'rho0_sd', 'r0', 'r0_sd', 'alpha', 'alpha_sd', 'n_fit']


# ============================================================================
# Geometry, spine fitting, profiles and per-cloud figure primitives
#   (from dust_3d_functions.py)
# ============================================================================

pts2line   = lambda p1, p2: ((p1+p2)/2, p2-p1)


line2pts   = lambda pt, vec, t=1: (pt + t*vec, pt - t*vec)


coord_idx  = lambda val, dx, x0: ((np.array(val) - x0)/dx).astype('int')


def plummer(r, rho0, r0, alpha):
    num = rho0
    den = 1 + (r/r0)**2

    return num/(den**alpha)


def get_subcube(bigcube, coords, xyz0s=[211, 90.6, 68], dxyzs=[-0.23, 0.23, 1],
                dmask=None):
    """Slice `bigcube` to the l/b/d box in `coords`.

    `dmask` optionally selects individual distance slices instead of taking the
    whole [do, di] run -- pass a boolean array to keep only the slices where it
    is True. Written for dendrogram leaf masks: leaf.get_mask() is a boolean of
    length source_array.shape[0], so it can be handed straight in.

    The mask may be given at either length and is interpreted accordingly:
      len == bigcube.shape[0]  -> indexes the PARENT distance axis, and is cut
                                  down to [zo:zi] here
      len == zi - zo           -> already indexes this subcube's distance axis
    (When the box is the full distance extent the two coincide, as they must.)

    A mask with interior gaps is allowed, but then the returned stack's distance
    axis is no longer evenly spaced -- only the selected slices are present, in
    order. Anything that assumes a 1 pc grid downstream must account for that.
    """
    dx, dy, dz = dxyzs
    x0, y0, z0 = xyz0s

    lo, li, bo, bi, do, di = coords

    xo, xi = coord_idx([lo, li], dx, x0)
    yo, yi = coord_idx([bo, bi], dy, y0)
    zo, zi = coord_idx([do, di], dz, z0)

    cube = bigcube[zo:zi, yo:yi, xo:xi]

    if dmask is None:
        return cube

    m = np.asarray(dmask)
    if m.dtype != bool:
        raise TypeError(f"dmask must be boolean, got {m.dtype}")
    if m.ndim != 1:
        raise ValueError(f"dmask must be 1-D along the distance axis, got {m.ndim}-D")

    if m.size == bigcube.shape[0]:
        m = m[zo:zi]
    elif m.size != cube.shape[0]:
        raise ValueError(
            f"dmask has length {m.size}; expected {bigcube.shape[0]} (parent "
            f"distance axis) or {cube.shape[0]} (this subcube's)")
    if not m.any():
        raise ValueError("dmask selects no distance slices")

    return cube[m]


def dmask_is_contiguous(dmask):
    """True if the mask's True entries form one unbroken run.

    Matters because anything that turns a cube index back into a distance
    (get_99th_percs, and so the whole spine chain) assumes the distance axis is
    contiguous and unit spaced. Dendrogram leaves always satisfy this; an
    arbitrary hand-built mask need not.
    """
    sel = np.flatnonzero(np.asarray(dmask))
    return sel.size > 0 and sel[-1] - sel[0] + 1 == sel.size


def find_dip(r, n, persist=3, n_start=5):
    """Radius where a radial profile stops falling.

    Starts at the highest of the first `n_start` bins and walks outward while n
    falls. The first bin where it stops falling is a candidate; it is accepted
    only if none of the next `persist` bins drops below it (otherwise it was a
    shoulder and the walk carries on). Returns (index, flag), flag being 'dip',
    or 'edge' if the profile falls all the way to its last bin.
    """
    r, n = np.asarray(r, float), np.asarray(n, float)
    i = int(np.nanargmax(n[:n_start]))
    while True:
        while i + 1 < len(n) and n[i + 1] < n[i]:
            i += 1
        if i + 1 >= len(n):
            return i, 'edge'
        ahead = n[i + 1:i + 1 + persist]
        if np.all(ahead >= n[i]):
            return i, 'dip'
        i += 1 + int(np.argmin(ahead))          # resume from the lower point ahead


def bin_by_r(ds, As, bns=2):
    if type(bns) == int:
        bnsz = bns
        d_bins = np.arange(0, np.nanmax(ds)+5, bnsz)
    else:
        d_bins = bns

    df = pd.DataFrame(np.array([np.ravel(ds),
                                 np.ravel(As),
                                 ]).transpose(),
                       columns=['d', 'rho'])

    df_binned = df.groupby(pd.cut(df['d'], bins=d_bins)).median().dropna()

    return df_binned


def dperp_pt_line(p0s, p1, p2):
    num = np.linalg.norm(np.cross(p0s-p1[np.newaxis, :], p0s-p2[np.newaxis, :], axis=1), axis=1)
    den = np.linalg.norm(p2-p1)
    return num/den


def dparl_pt_line(p0s, p1, p2):
    # Signed scalar projection onto the spine: negative behind p1.
    num = np.nansum((p1[np.newaxis, :]-p0s)*(p2-p1)[np.newaxis,:],axis=1)
    den = np.linalg.norm(p2-p1)#**2
    # np.dot
    return -num/den


def coords_to_cube(coords, vals):
    zs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    xs = np.unique(coords[:, 2])

    x_idx = {v: i for i, v in enumerate(xs)}
    y_idx = {v: i for i, v in enumerate(ys)}
    z_idx = {v: i for i, v in enumerate(zs)}

    cube = np.full((len(zs), len(ys), len(xs)), np.nan)

    for (z, y, x), val in zip(coords, vals):
        cube[z_idx[z], y_idx[y], x_idx[x]] = val

    return cube


def sph_2_xyz(lbd):  # checked
    l, b, d = lbd
    z = d*np.sin(b*u.deg)
    y = d*np.cos(b*u.deg)*np.cos(l*u.deg)
    x = d*np.cos(b*u.deg)*np.sin(l*u.deg)

    return x, y, z


def xyz_2_sph(xyz):  # works for a point
    x, y, z = xyz
    d = np.linalg.norm([x, y, z])
    b = np.arcsin(z/d)*u.rad.to(u.deg)

    l = np.arctan(x/y)*u.rad.to(u.deg)

    if y < 0:
        if x > 0:
            l += 180
        if x < 0:
            l -= 180

    return l, b, d


def get_cloud_info(i, source_array, cloud_catalog, drange='by r', xyz0s=[210, -90.2, 68], dxyzs=[-0.229, 0.229, 1],
                   dmask=None, pad=0.0):
    # `cloud_catalog` is the cloud table (a DataFrame), passed in like `source_array`.
    lb = cloud_catalog.iloc[i][['l_max(deg)', 'l_min(deg)', 'b_min(deg)', 'b_max(deg)']].to_numpy().astype('float')

    d  = cloud_catalog.iloc[i]['d_mean(pc)']#.to_numpy().astype('float')
    # r = np.nanmax((np.nanstd(lb[:2]),np.nanstd(lb[2:])),axis=0) * u.deg.to(u.rad) * d

    # d,r  = cloud_catalog.iloc[i][['d_mean(pc)','d_std(pc)']].to_numpy().astype('float')

    # `pad` grows the catalog box by that many degrees on all four sides
    # (l_max+pad, l_min-pad, b_min-pad, b_max+pad). 0 keeps the box as catalogued.
    pad = np.array([1,-1,-1,1])*pad

    if drange == 'by r':
        coords = np.concatenate((lb + pad, np.array([d-r,d+r])))

    elif drange == 'full':
        # Full extent of the distance axis, read off the array itself rather
        # than hardcoded: index 0..nz maps back to d = z0 .. z0 + nz*dz.
        nz = source_array.shape[0]
        coords = np.concatenate((lb + pad, xyz0s[2] + np.array([0, nz])*dxyzs[2]))
    else:
        coords = np.concatenate((lb + pad, np.array([d-drange,d+drange])))

    # `dmask` is an alternative to `drange`, not a filter applied on top of it:
    # the mask itself defines which distances are kept, so the box is opened to
    # the full axis first and the mask does the selecting. Any drange passed
    # alongside is ignored (the l/b box still comes from the catalog as usual).
    if dmask is not None:
        nz = source_array.shape[0]
        coords = np.concatenate((lb + pad, xyz0s[2] + np.array([0, nz])*dxyzs[2]))

    coords[4] = max(coords[4], xyz0s[2])

    cube = get_subcube(source_array, coords, xyz0s=xyz0s, dxyzs=dxyzs, dmask=dmask)

    # Report the distance range actually returned rather than the box asked for.
    # For a gappy mask this is the outer envelope, not a promise of contiguity.
    if dmask is not None:
        m = np.asarray(dmask)
        if m.size == source_array.shape[0]:
            sel = np.flatnonzero(m)
            coords[4] = xyz0s[2] + sel[0]*dxyzs[2]
            coords[5] = xyz0s[2] + (sel[-1] + 1)*dxyzs[2]

    # print(coords,cube.shape)

    return cube, coords


def get_99th_percs(i, source_array, cloud_catalog, drange='by r', which='99', db=0.23, dl=-0.23,
                   dmask=None, pad=0.0):
    # A gappy dmask is rejected here rather than downstream: the index-to-distance
    # line below assumes the distance axis is contiguous and unit spaced, so a
    # mask with holes would silently mislabel every d it returns.
    if dmask is not None and not dmask_is_contiguous(dmask):
        raise ValueError("get_99th_percs needs a contiguous dmask: the spine chain "
                         "maps cube index -> distance assuming no gaps")
    cube, coords = get_cloud_info(i, source_array, cloud_catalog, drange, dmask=dmask, pad=pad)

    l0, b0, d0 = coords[[0, 2, 4]]
    ds_, bs, ls = np.indices(cube.shape).reshape(3, -1) * np.array([1, db, dl])[:, None] + np.array([d0, b0, l0])[:, None]
    xs, ys, zs = sph_2_xyz([ls, bs, ds_])

    df = pd.DataFrame(np.column_stack([xs, ys, zs, ls, bs, ds_, np.ravel(cube)]), columns=['x', 'y', 'z', 'l', 'b', 'd', 'n'])
    df2 = df.dropna().query('n > 0.5')  # choose pixels w data above this thresh

    per = np.nanpercentile(df2['n'], 99)  # 99th percentile

    df99 = df2.query('n > {}'.format(per))
    # df99_1 = df.query('n > {}'.format(per))

    if which == '99':
        return df99
    elif which == 'both':
        return df99, df
    else:
        return df.to_numpy()


def pick_p1p2(coords99, rng=np.random):
    # rng defaults to the global numpy RNG (unchanged behavior). Pass a
    # np.random.RandomState so parallel workers draw independent, reproducible
    # point pairs instead of sharing/duplicating global RNG state.
    xyz = coords99[:, 0:3]

    N = len(xyz)
    i1, i2 = rng.randint(N, size=2)
    while i1 == i2:  # picking the same point
        i2 = rng.randint(N)

    p1 = xyz[i1]
    p2 = xyz[i2]

    return p1, p2


def L1ds_minimizable(params, pixels):
    p1 = params[0:3]
    p2 = params[3:6]

    Avs = pixels[:, 6]  # filter out for nans first
    xyz = pixels[:, 0:3]

    dperps = dperp_pt_line(xyz, p1, p2)

    L1_d = np.nansum(dperps * Avs) / np.nansum(Avs)

    return L1_d


def L1ds(p1, p2, pixels):
    Avs = pixels[:, 6]
    xyz = pixels[:, 0:3]

    dperps = dperp_pt_line(xyz, p1, p2)
    dparls = dparl_pt_line(xyz, p1, p2)

    L1_d = np.nansum(dperps*Avs)/np.nansum(Avs)

    return L1_d, np.concatenate((pixels, dperps[:, np.newaxis], dparls[:, np.newaxis]), axis=1)


def orient_spine_endpoints(p1, p2):
    # The fit constrains the LINE, not a direction along it: pick_p1p2 draws its
    # seed pair in a random order and the L1 minimizer is free to hand the moved
    # endpoints back either way round. That randomness leaks downstream through
    # dparl_pt_line, whose d_parl is a SIGNED projection onto (p2 - p1), so a
    # subset of per-sample d_parl profiles come out mirrored against the rest.
    # Pin it here: the endpoint nearer the l = 0 meridian is p1, so (p2 - p1)
    # always points away from the Galactic-centre direction and d_parl is zeroed
    # at the same physical end on every fit.
    # abs(l) is degenerate for a spine straddling l = +/-180 (anti-centre); no
    # catalogued cloud has an endpoint that near the wrap, and the <= still
    # breaks such a tie consistently if one ever does.
    l1, l2 = xyz_2_sph(p1)[0], xyz_2_sph(p2)[0]
    return (p1, p2) if abs(l1) <= abs(l2) else (p2, p1)


def _fit_one_line(dats_arr, seed):
    # One Monte-Carlo spine fit: random point pair -> L1-minimized line.
    # Pure function of (dats_arr, seed) so it can run in a parallel worker.
    rng = np.random.RandomState(seed)
    p1, p2 = pick_p1p2(dats_arr, rng)

    ps_min = minimize(L1ds_minimizable, np.concatenate([p1, p2]), args=dats_arr)

    # canonical endpoint order (see orient_spine_endpoints): every consumer of
    # this row -- pick_best_line, spine_candidates -- inherits a d_parl that runs
    # the same way, so posterior-sample profiles stop mirroring each other.
    p1_, p2_ = orient_spine_endpoints(ps_min.x[0:3], ps_min.x[3:6])

    L1d_min, _ = L1ds(p1_, p2_, dats_arr)

    return [L1d_min, p1_[0], p1_[1], p1_[2], p2_[0], p2_[1], p2_[2]]


def pick_best_line(i, source_array, cloud_catalog, drange, n_iter, n_jobs=1, dmask=None, pad=0.0):
    # Each iteration is an independent random-line fit. n_jobs=1 keeps the original
    # serial behavior; n_jobs=-1 uses all cores, or pass a positive int. Seeds are
    # 0..n_iter-1 so results are reproducible regardless of n_jobs.
    dats = get_99th_percs(i, source_array, cloud_catalog, drange, which='both', dmask=dmask, pad=pad)
    dats_arr = dats[0].to_numpy()

    if n_jobs == 1:
        rows = [_fit_one_line(dats_arr, seed) for seed in range(n_iter)]
    else:
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one_line)(dats_arr, seed) for seed in range(n_iter))

    df = pd.DataFrame(rows, columns=['d', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2'])

    df = df.sort_values('d')
    dmin = df.iloc[0]['d']
    p1 = df.iloc[0][['x1', 'y1', 'z1']].to_numpy()
    p2 = df.iloc[0][['x2', 'y2', 'z2']].to_numpy()

    return p1,p2,dmin


def get_spine(i, source_array, cloud_catalog, drange='by r', n_iter=5, n_jobs=1, dmask=None, pad=0.0):
    # Source notebook returned `np.around(pt_lbd,2),n` -- `n` is undefined here,
    # likely a leftover from pick_best_line's loop variable.
    dats = get_99th_percs(i, source_array, cloud_catalog, drange, which='both', dmask=dmask, pad=pad)

    p1_xyz, p2_xyz, dmin = pick_best_line(i, source_array, cloud_catalog, drange, n_iter,
                                          n_jobs=n_jobs, dmask=dmask, pad=pad)

    # pt_, vec_ = pts2line(p1_xyz,p2_xyz) #increase the distance between the two points 
    # p1_xyz, p2_xyz = line2pts(pt_,vec_,10)

    L1d, all_dat = L1ds(p1_xyz, p2_xyz, dats[1].to_numpy())

    all_df = pd.DataFrame(all_dat, columns=['x', 'y', 'z', 'd', 'b', 'l', 'n', 'dperp', 'dparl'])
    cube_perp  = coords_to_cube(all_df[['d', 'b', 'l']].to_numpy(), all_df['dperp'])
    cube_perp  = np.flip(cube_perp.transpose(), axis=2)  # reorients to look like cube

    cube_parl = coords_to_cube(all_df[['d', 'b', 'l']].to_numpy(), all_df['dparl'])
    cube_parl = np.flip(cube_parl.transpose(), axis=2)

    p1_lbd = np.array(xyz_2_sph(p1_xyz))
    p2_lbd = np.array(xyz_2_sph(p2_xyz))

    pt_lbd, vec_lbd = pts2line(p1_lbd, p2_lbd)

    return np.around(pt_lbd, 2), np.around(vec_lbd, 2), cube_perp, cube_parl, L1d


def radii_on_box(i, source_array, cloud_catalog, drange, spine_xyz, dmask=None, pad=0.0):
    """(cube, r_perp, r_parl, coords) on a box grown by `pad`, for a FIXED spine.

    No fitting happens here. `spine_xyz` is the (p1, p2) pair an earlier get_spine
    produced -- pass ha()'s 'spine_xyz'. Every voxel of the padded box gets its
    perpendicular and parallel distance to that same line, so widening the box
    extends the profile outward instead of re-deciding where the cloud's axis is.

    The column relabelling below is get_spine's, kept verbatim: get_99th_percs
    builds its frame as (x, y, z, l, b, d, n) and get_spine reads columns 3-5
    back as ('d','b','l'), so coords_to_cube is fed (l, b, d) and the following
    transpose+flip is what puts the cube back into (d, b, l). The names are
    wrong but the array is right, and matching it exactly is what keeps this
    consistent with the unpadded path.
    """
    cube, coords = get_cloud_info(i, source_array, cloud_catalog, drange,
                                  dmask=dmask, pad=pad)
    pix = get_99th_percs(i, source_array, cloud_catalog, drange, which='else',
                         dmask=dmask, pad=pad)
    p1_xyz, p2_xyz = spine_xyz
    _, all_dat = L1ds(p1_xyz, p2_xyz, pix)
    all_df = pd.DataFrame(all_dat, columns=['x', 'y', 'z', 'd', 'b', 'l', 'n',
                                            'dperp', 'dparl'])
    lbd = all_df[['d', 'b', 'l']].to_numpy()
    r_perp = np.flip(coords_to_cube(lbd, all_df['dperp']).transpose(), axis=2)
    r_parl = np.flip(coords_to_cube(lbd, all_df['dparl']).transpose(), axis=2)
    return cube, r_perp, r_parl, coords


def ha(i, source_array, cloud_catalog, drange, n='none', n_jobs=1, dmask=None, pad=0.0):
    # `b_perp` and `dist_cube` are the same array; `dist_cube` is kept because
    # histogram, Sig_radial_profiles, make_Av_isosurface and
    # rho_radial_profiles still read it under that name.
    # n_jobs is forwarded to the Monte-Carlo spine fit (see pick_best_line).
    cube, coords = get_cloud_info(i, source_array, cloud_catalog, drange, dmask=dmask, pad=pad)
    if type(n) != str:
        cube = cube*(cube > n)

    pta, veca, rperp, rparl, L1da = get_spine(i, source_array, cloud_catalog, drange,
                                              n_jobs=n_jobs, dmask=dmask, pad=pad)

    # get_spine hands the spine back as a midpoint/difference pair in (l,b,d);
    # recover the two endpoints and carry them in xyz as well, so a position
    # along the spine can be put back on the sky. get_spine rounds to 2 dp,
    # ~0.01 deg, which is a few percent of a pixel -- fine for plotting.
    p2_lbd, p1_lbd = line2pts(pta, veca, 0.5)

    # L1da -- the Av-weighted mean |d_perp| the spine fit minimized -- is carried
    # out too: it is the fit residual, and so the number to compare when the same
    # cloud is fit over several distance ranges (see fit_spines_over_dranges).
    return {'coords': coords, 'dust_cube': cube,'r_perp': rperp, 'r_parl': rparl,
            'spine': [pta, veca], 'L1d': L1da, 'drange': drange,
            'spine_xyz': [np.array(sph_2_xyz(p1_lbd)), np.array(sph_2_xyz(p2_lbd))]}


def plot_spine(axs, spine, scolor='cyan'):
    # [print(ax) for ax in axs]
    # Reads the first three panels only, so it takes either plot_moment_maps'
    # four-tuple (the fourth is the spare panel) or a bare (l-b, l-d, b-d) row.
    lbax, ldax, bdax = axs[0], axs[1], axs[2]
    pt, vect = spine

    kws = {'color': scolor, 'pivot': 'mid', 'scale': 1,
           'angles': 'xy', 'width': 2e-2, 'alpha': 0.7, 'headwidth': 0.}

    p1,p2 = line2pts(pt,vect)

    lbax.quiver(pt[0], pt[1], vect[0], vect[1], **kws)
    lbax.scatter(pt[0], pt[1],marker='x',color='white')
    lbax.scatter(p1[0], p1[1],marker='x',color='black')
    lbax.scatter(p2[0], p2[1],marker='x',color='black')

    ldax.quiver(pt[0], pt[2], vect[0], vect[2], **kws)
    ldax.scatter(pt[0], pt[2],marker='x',color='white')
    ldax.scatter(p1[0], p1[2],marker='x',color='black')
    ldax.scatter(p2[0], p2[2],marker='x',color='black')

    bdax.quiver(pt[2], pt[1], vect[2], vect[1], **kws)
    bdax.scatter(pt[2], pt[1],marker='x',color='white')
    bdax.scatter(p1[2], p1[1],marker='x',color='black')
    bdax.scatter(p2[2], p2[1],marker='x',color='black')


def plot_moment_maps(cube, coords, which, cmp='magma',unit='Av', r_perp=None, r_parl=None, axs=None,
                     iso=None, iso_color='cyan', iso_lw=1.0, d_margin=0.0):
    """...`iso` draws the projected 3-D isosurface at that VOLUME DENSITY.

    The projection of a 3-D isosurface onto a plane is the silhouette the
    enclosed region casts: `(cube > iso).any(axis=k)`, outlined at 0.5. That is
    NOT the same curve as contouring the zeroth-moment map, which also depends
    on how much path length each sightline has -- a long thin filament seen
    end-on makes a bright column without ever exceeding `iso` anywhere.
    Same construction as the `outline` option in plot_skin_panels.
    """
    # `axs` lets a caller draw these panels into a grid it already owns (see
    # plot_cloud_summary): pass three axes, read as (l-b, l-d, b-d), or four to
    # keep the spare panel. Left at None the function owns its own 2x2 figure,
    # exactly as before, and takes care of the suptitle and tight_layout.
    if unit == 'Av':
        im0 = ((np.nansum(cube*u.cm**-3, axis=0)*u.pc).cgs/(2.2e21*u.cm**-2)).value
        im1 = ((np.nansum(cube*u.cm**-3, axis=1)*u.pc).cgs/(2.2e21*u.cm**-2)).value
        im2 = ((np.nansum(cube*u.cm**-3, axis=2)*u.pc).cgs/(2.2e21*u.cm**-2)).value

        lvls = np.arange(0.5,3.1,0.5)  #[0.5, 1]
        cmin = 0.25
        cmax = 2.1
    else:
        im0 = ((np.nansum(cube*u.cm**-3, axis=0)*u.pc)*c.m_p).to(u.Msun/u.pc**2).value
        im1 = ((np.nansum(cube*u.cm**-3, axis=1)*u.pc)*c.m_p).to(u.Msun/u.pc**2).value
        im2 = ((np.nansum(cube*u.cm**-3, axis=2)*u.pc)*c.m_p).to(u.Msun/u.pc**2).value

        lvls = [10, 20, 50, 70]
        cmin = 1
        cmax = 20


    # cmp = 'binary'#'magma'
    kws = {'levels': 40, 'cmap': cmp, 'extend': 'max', 'norm': 'log'}
    kws2 = {'levels': lvls, 'colors': 'black', 'linewidths': [0.75,1.5]}

    if axs is None:
        # fig, axs = plt.subplots(1, 3, dpi=200, figsize=(15, 5))#,sharey='row',sharex='col')
        fig, axs = plt.subplots(2, 2, dpi=200, figsize=(10, 10))#,sharey='row',sharex='col')
        fig.suptitle(which)

        # lbax, ldax, bdax = axs.flat
        lbax, bdax, ldax, spax = axs.flat
    else:
        fig = None
        axs = np.ravel(np.asarray(axs, dtype=object))
        lbax, ldax, bdax = axs[:3]
        spax = axs[3] if len(axs) > 3 else None

    lbax.contourf(im0, extent=coords[:4], **kws)
    # Black contours of the moment map itself, matching the l-d and d-b panels
    # below -- these had been commented out, leaving the l-b panel the only one
    # without them. (The line that replaced them, contour(cube*(cube>5), level=1),
    # passed the 3-D cube to contour and misspelled `levels`; the cyan `iso`
    # silhouette further down is what it was reaching for.)
    lbax.contour(im0, extent=coords[:4], **kws2)
    # lbax.clabel(cs, levels=[0.5,1,2,3],rightside_up=True,fontsize='x-small')

    ldax.contourf(im1, extent=coords[[0, 1, 4, 5]], **kws)
    ldax.contour(im1, extent=coords[[0, 1, 4, 5]], **kws2)

    bdax.contourf(im2.transpose(), extent=coords[[4,5,2,3]], **kws)
    bdax.contour(im2.transpose(), extent=coords[[4,5,2,3]], **kws2)

    # projected 3-D isosurface: one silhouette per projection axis, matching each
    # panel's axis order and transpose exactly as the contourf calls above do.
    if iso is not None and np.isfinite(iso):
        dense = np.nan_to_num(cube, nan=0.0) > iso
        for ax_, axis_, extent_, tr_ in [(lbax, 0, coords[:4], False),
                                         (ldax, 1, coords[[0, 1, 4, 5]], False),
                                         (bdax, 2, coords[[4, 5, 2, 3]], True)]:
            sil = dense.any(axis=axis_)
            if sil.any() and not sil.all():
                ax_.contour(sil.transpose() if tr_ else sil, extent=extent_,
                            levels=[0.5], colors=iso_color, linewidths=iso_lw)

    # sp = np.nansum(cube,axis=(1,2))* (0.125*u.deg)**2 * np.arange(coords[4],coords[5]) 
    # spax.plot(np.arange(coords[4],coords[5]),sp,color='black',linewidth=1.5,)

    # vals,bins = np.histogram(np.ravel(cube),bins=10**np.arange(-3,3.1,0.1))
    # spax.plot(bins[:-1],vals,color='black')
    # spax.vlines(np.nanpercentile(cube[cube>1],99),0,1.4*np.nanmax(vals),color='red',linestyle='--')
    # spax.loglog()


    for ax in [lbax,ldax]:
        ax.set_xlim(coords[:2])

    for ax in [lbax,bdax]:
        ax.set_ylim(coords[2:4])

    # `d_margin` pc of blank either side of the distance window, so the near and
    # far edges of the cut (the dmask edges, when one is used -- get_cloud_info
    # rewrites coords[4:6] to the mask's span) sit inside the panel with some
    # context rather than flush against the frame. 0 restores the tight fit.
    d_lo, d_hi = coords[4] - d_margin, coords[5] + d_margin
    ldax.set_ylim(d_lo, d_hi)
    bdax.set_xlim(d_lo, d_hi)

    # spax.set_ylim(1e1,1.2*np.nanmax(vals))

    lbax.set_ylabel('Galactic Latitude [deg]')
    lbax.set_xlabel('Galactic Longitude [deg]')

    ldax.set_ylabel('Heliocentric Distance [pc]')
    ldax.set_xlabel('Galactic Longitude [deg]')

    bdax.set_xlabel('Heliocentric Distance [pc]')
    bdax.set_ylabel('Galactic Latitude [deg]')

    # spax.set_xlabel('$n\ [\mathrm{cm^{-3}}$]')
    # spax.set_ylabel('Number of pixels')

    # spax.set_axis('off')

    for arr, ls in [(r_perp, 'solid'), (r_parl, '--')]:
            if arr is None:
                continue
    
            kws3 = {'levels': 6, 'colors': 'black', 'linewidths': 1.2,
                    'linestyles': ls}
    
            css = [lbax.contour(np.nanmin(arr, axis=0),
                                extent=coords[:4], **kws3),
                   ldax.contour(np.nanmin(arr, axis=1),
                                extent=coords[[0, 1, 4, 5]], **kws3),
                   bdax.contour(np.nanmin(arr, axis=2).transpose(),
                                extent=coords[[4, 5, 2, 3]], **kws3)]
    
            for cs in css:
                for lbl in plt.clabel(cs, inline=True, fontsize=10, fmt='%.0f'):
                    lbl.set_rotation(0)

    if fig is not None:
        fig.tight_layout()

    return (lbax, ldax, bdax, spax) #axs


def rho_radial_profiles_compute(i, source_array, cloud_catalog, drange='by r', tube_r='px', n_jobs=1,
                                parl_origin='edge', cloud_dict=None, perp_slab='L', dmask=None,
                                perp_pad=0.5, perp_nmin=None, perp_fit='log'):
    # Pure compute half of rho_radial_profiles: no matplotlib, returns a picklable
    # dict, so it can run inside a Parallel(...) over clouds and be plotted later.
    # n_jobs forwards to the Monte-Carlo spine fit inside ha (see pick_best_line):
    # pass n_jobs=-1 to fit one cloud across all cores. Leave at 1 when this call
    # is itself already inside an outer Parallel over clouds, to avoid nesting.
    # Depends on module-level globals ds, drs.
    # d_perp is fit with the plummer factor of the cylinder model, d_parl with
    # the superGaussian one. d_parl is run end to end as a single profile: one
    # curve from the near edge of the cloud to the far edge, not two branches
    # folded onto |d_parl| about p1.
    # `tube_r` is the radius in pc of the cylinder the d_parl profile averages
    # within -- 'px' for the pixel scale, or np.inf for no cut.
    # `parl_origin` sets where d_parl is zeroed: 'edge' puts 0 at the near edge
    # so the profile runs 0 -> cloud length, 'p1' keeps the raw signed spine
    # coordinate. Either way `parl_p1` is where p1 sits on the plotted axis.
    # Pass `cloud_dict` to reuse an ha() call the caller has already paid for --
    # the spine fit inside it is the expensive part. It must have been built with
    # the same i/drange, which is not checked.
    if cloud_dict is None:
        cloud_dict = ha(i, source_array, cloud_catalog, drange, n_jobs=n_jobs, dmask=dmask)

    log_px_scl = np.log10(0.125*u.deg.to(u.rad)*cloud_catalog['d_mean(pc)'].iloc[i])

    # d_parl is fit FIRST: with perp_slab on, its z0 and L set the slab the
    # d_perp profile is then averaged within, so the order cannot be reversed.

    # d_parl spans both sides of p1, so bin it once across its whole range --
    # near edge to far edge -- on linear bins, which carry through zero where
    # the old log bins could not. Only pixels inside a tube of radius tube_r
    # about the spine are averaged in; d_perp is nan off the cloud, and those
    # compare False, so they drop out here too.
    if tube_r == 'px':
        tube_r = 3*10**log_px_scl

    tube = np.ravel(cloud_dict['r_perp']) <= tube_r
    parl = np.ravel(cloud_dict['r_parl'])[tube]
    tube_n = np.ravel(cloud_dict['dust_cube'])[tube]

    df_parl, mins_parl, fit_parl, parl_p1 = None, None, None, np.nan

    near, far = np.nanmin(parl), np.nanmax(parl)
    if np.isfinite(near) and np.isfinite(far):
        px_scl = 10**log_px_scl  # one bin per pixel along the spine
        parl_bins = np.arange(near, far+px_scl,3*px_scl)

        if len(parl_bins) > 2:
            df_parl = bin_by_r(parl, tube_n, parl_bins)
            # bin_by_r bins on the raw coordinate, so re-zero after the fact
            parl_p1 = -near if parl_origin == 'edge' else 0.0
            df_parl['d'] = df_parl['d'] + parl_p1
            mins_parl = find_peaks(-df_parl['rho'])

            # SUPERGAUSSIAN FIT DISABLED. fit_parl stays None (set above), which
            # both consumers already tolerate: the perp_slab block below falls
            # back to the crest of df_parl for z0, and the plot draws a flat line
            # at the profile's median instead of a fitted curve. Uncomment to
            # restore.
            #
            # # d_parl is the model's z axis, so fit it with the superGaussian
            # # factor the way d_perp is fit with the Plummer one. eta is the boxy
            # # exponent: 2 is Gaussian, larger is flat-topped with steep walls.
            # # Seed L with the observed extent and z0 with the crest.
            # zs, ns = df_parl['d'].to_numpy(), df_parl['rho'].to_numpy()
            # p0 = [np.nanmax(ns), zs[np.nanargmax(ns)], zs[-1]-zs[0], 2.]
            #
            # try:
            #     # bounds keep L and eta positive, so pass max_nfev (curve_fit
            #     # switches from lm to trf once bounds are given and maxfev is
            #     # an lm-only name).
            #     fit_parl = curve_fit(superGaussian_scaled, zs, ns, p0=p0,
            #                          bounds=([0, -np.inf, 1e-3, 0.1],
            #                                  [np.inf, np.inf, np.inf, 50.]),
            #                          max_nfev=5000)[0]
            #     fit_parl = list(np.around(fit_parl, 2))
            # except (RuntimeError, ValueError):
            #     # a cloud whose axial profile the superGaussian cannot reach
            #     # shouldn't take down a whole Parallel batch -- plot skips it
            #     fit_parl = None
            pass

    # n_med, the median of the binned d_parl profile, in cm^-3. Computed here --
    # ahead of the d_perp branch, which thresholds on it, and of the mom0 map --
    # so every use of it comes from one place.
    med_parl = np.nanmedian(df_parl['rho'].to_numpy()) if df_parl is not None else np.nan

    # --- d_perp, averaged within a slab along the spine ---------------------
    # Without a slab the radial bins run the full length of the subcube, so every
    # bin mixes the cloud with however much empty volume the distance cut left on
    # either side of it, and the median collapses (Corona Australis at +/-100 pc:
    # a d_perp profile topping out at 0.6 cm^-3 and a Plummer fit with alpha < 0,
    # i.e. rising outward). Cutting to the cloud's own axial extent is the d_perp
    # counterpart of the tube_r cut d_parl already gets.
    #
    # `perp_slab`: 'L' takes the slab from the fit above, z0 +/- L/2; a number is
    # a fixed full thickness in pc about z0 (or about the profile's crest if the
    # superGaussian fit failed); a (lo, hi) pair is an explicit interval in raw
    # r_parl; None keeps the old full-range behavior.
    # The d_perp profile is measured on its own cube: the catalog box grown by
    # `perp_pad` degrees on every side, so the profile runs out past the
    # catalogued edge instead of stopping at it. The SPINE IS NOT REFITTED --
    # radii_on_box reuses the one already fitted above, so the extra sky only
    # adds voxels at larger d_perp rather than re-deciding where the axis is.
    # (Refitting on the padded box moves it a long way -- Ophiuchus' spine
    # direction swung 18 deg -- because the padding brings in real dense gas the
    # catalog box was clipping.)
    perp_cube_raw, r_perp, r_parl, perp_coords = (
        cloud_dict['dust_cube'], cloud_dict['r_perp'], cloud_dict['r_parl'],
        cloud_dict['coords'])
    if perp_pad:
        perp_cube_raw, r_perp, r_parl, perp_coords = radii_on_box(
            i, source_array, cloud_catalog, drange, cloud_dict['spine_xyz'],
            dmask=dmask, pad=perp_pad)

    slab, slab_lo, slab_hi = None, -np.inf, np.inf

    if isinstance(perp_slab, (tuple, list)):
        # an explicit (lo, hi) interval in the RAW r_parl coordinate, (x - p1).u
        # -- e.g. the two ends of the walked spine (plot_cylinder_slice)
        slab_lo, slab_hi = float(perp_slab[0]), float(perp_slab[1])
        slab = (slab_lo, slab_hi)
    elif perp_slab is not None and df_parl is not None:
        if fit_parl is not None:
            z0, L = fit_parl[1], fit_parl[2]
        else:
            # no fit to take z0 from, so centre on the observed crest
            z0, L = df_parl['d'].iloc[np.nanargmax(df_parl['rho'].to_numpy())], np.nan

        thick = L if perp_slab == 'L' else float(perp_slab)

        if np.isfinite(z0) and np.isfinite(thick) and thick > 0:
            # z0 is in the re-zeroed coordinate df_parl['d'] uses; r_parl is raw,
            # so undo the shift rather than shifting the whole cube
            slab_lo, slab_hi = z0 - thick/2 - parl_p1, z0 + thick/2 - parl_p1
            slab = (slab_lo, slab_hi)

    # nan'ing the radius drops those pixels: bin_by_r cuts on it, and pd.cut
    # sends nan to no bin at all
    in_slab = (r_parl >= slab_lo) & (r_parl <= slab_hi)
    r_perp_cut = np.where(in_slab, r_perp, np.nan)

    # a degenerate axial fit can hand back a slab a couple of pc thick, which
    # would leave nothing to fit a profile to -- fall back to the full range and
    # say so in the result rather than return a fit made of two bins
    slab_thin = np.count_nonzero(np.isfinite(r_perp_cut)) < 0.02*np.count_nonzero(np.isfinite(r_perp))

    if slab_thin:
        r_perp_cut, slab = r_perp, None

    r_perp_bins = 10**np.arange(log_px_scl, np.log10(np.nanmax(r_perp_cut))+0.2, 0.1)

    # Keep only gas above a volume-density floor. NaN rather than 0 so bin_by_r's
    # per-bin median SKIPS these voxels entirely -- the profile is then the median
    # of the dense gas at each radius, not the median over all volume at that
    # radius.
    # `perp_nmin` sets that floor in cm^-3. Default (None) keeps the historical
    # n_med/10, which is per-cloud and so not comparable between clouds; pass an
    # absolute number for a common cut. NOTE this enters the Plummer fit, so a
    # higher floor truncates the profile's outskirts and moves rho0/r0/alpha.
    perp_cube = perp_cube_raw
    n_cut = perp_nmin if perp_nmin is not None else (
        med_parl/10 if np.isfinite(med_parl) else np.nan)
    if np.isfinite(n_cut):
        perp_cube = np.where(perp_cube > n_cut, perp_cube, np.nan)

    df = bin_by_r(r_perp_cut, perp_cube, r_perp_bins)
    mins = find_peaks(-df['rho'])

    # The same profile on LINEAR bins from 0, two 0.125 deg pixel scales wide
    # (0.25 deg, ~1.1 native Edenhofer pixels; the d_parl bins are 3), for display
    # and the dip search, and for the Plummer fit with perp_fit='lin_dip'.
    # df_lin is the n >= perp_nmin masked profile (as fitted); df_lin_all takes the
    # median over ALL voxels in each shell, so it is not floored by the threshold.
    w_lin = 2*10**log_px_scl
    lin_bins = np.arange(0, np.nanmax(r_perp_cut) + w_lin, w_lin)
    df_lin = bin_by_r(r_perp_cut, perp_cube, lin_bins)
    df_lin_all = bin_by_r(r_perp_cut, perp_cube_raw, lin_bins)

    # `perp_fit`: 'log' (default, historical) fits the Plummer law to the
    # log-binned profile out to its last bin; 'lin_dip' fits the LINEAR bins
    # (df_lin) only out to the first persistent dip (find_dip), bounded so r0 and
    # alpha stay positive.
    r_dip, dip_flag = np.nan, None
    if perp_fit == 'lin_dip':
        k_dip, dip_flag = find_dip(df_lin['d'], df_lin['rho'])
        r_dip = float(df_lin['d'].iloc[k_dip])
        sel = (df_lin['d'] <= r_dip).to_numpy()
        x_, y_ = df_lin['d'].to_numpy()[sel], df_lin['rho'].to_numpy()[sel]
        p_, _ = curve_fit(plummer, x_, y_, p0=[np.nanmax(y_), 1.0, 1.0],
                          bounds=([0, 1e-3, 1e-3], [np.inf, 1e3, 50]), max_nfev=20000)
        rho0, r0, alpha = np.around(p_, 2)
    else:
        rho0, r0, alpha = np.around((curve_fit(plummer, df['d'], df['rho'], p0=[10, 10, 3], maxfev=5000))[0], 2)

    # Mass of the cylinder: every voxel between the spine caps (the slab, i.e.
    # wherever r_perp_cut is finite) and within r_dip of the spine, over the
    # distance window, with no density cut. Each pixel covers (CD * d)^2 of its
    # 1 pc slice, as in mass_profile.
    M_cyl, n_cyl = np.nan, 0
    if np.isfinite(r_dip) and dmask is not None:
        d_sl = ds[np.asarray(dmask, bool)]
        if len(d_sl) == perp_cube_raw.shape[0]:
            cyl = np.isfinite(r_perp_cut) & (r_perp_cut <= r_dip)
            per_slice = np.nansum(np.where(cyl, perp_cube_raw, 0.0), axis=(1, 2))
            area = (CD * u.deg.to(u.rad) * d_sl)**2
            M_cyl = float((np.sum(per_slice * area) * 1.36 * c.m_p * u.cm**-3
                           * u.pc**3).to(u.Msun).value)
            n_cyl = int(cyl.sum())

    # zeroth moment map for the sky panel: collapse the distance axis (the cube
    # is [d, b, l]) to Av, matching plot_moment_maps' im0. Carried here rather
    # than in _plot so the plotting half stays free of the cube.
    #
    # KEEP ONLY GAS DENSER THAN n_med/10. The cut is applied to the VOXELS before
    # the sum, not to the collapsed map: n_med is a volume density in cm^-3 while
    # mom0 is an A_V column in mag, so thresholding the map by it directly would
    # be comparing different quantities. Masked voxels go to 0, so mom0 becomes
    # the A_V contributed by dense gas alone; sightlines with nothing above the
    # cut come out at 0 rather than NaN, which the log contourf then drops.
    dense = cloud_dict['dust_cube']
    if np.isfinite(med_parl):
        dense = np.where(dense > med_parl/10, dense, 0.0)
    mom0 = ((np.nansum(dense*u.cm**-3, axis=0)*u.pc).cgs/(2.2e21*u.cm**-2)).value

    # 'slab' is the d_parl interval the d_perp profile was averaged within, in the
    # RAW r_parl coordinate, or None if the whole range was used (perp_slab off,
    # no axial fit to take it from, or the slab came out too thin to fit in).
    return {'i': i, 'df': df, 'df_lin': df_lin, 'df_lin_all': df_lin_all, 'lin_bins': lin_bins,
            'perp_fit': perp_fit, 'r_dip': r_dip, 'dip_flag': dip_flag,
            'M_cyl': M_cyl, 'n_cyl': n_cyl,
            'mins': mins, 'fit': [rho0, r0, alpha],
            'df_parl': df_parl, 'mins_parl': mins_parl, 'parl_p1': parl_p1,
            'fit_parl': fit_parl, 'mom0': mom0, 'med_parl': med_parl,
            'perp_pad': perp_pad, 'perp_coords': perp_coords, 'perp_nmin': n_cut,
            'coords': cloud_dict['coords'],
            'slab': slab, 'slab_thin': slab_thin, 'spine_xyz': cloud_dict['spine_xyz']}


def rho_radial_profiles_plot(result, plt_='new', color='black', lbl='by r', mark_scales=True):
    # Drawing half of rho_radial_profiles -- consumes a dict from _compute. Kept
    # serial (matplotlib is not process-safe). `plt_` is 'new' or an axes array
    # unpacking into (map_ax, parl_ax, perp_ax); the returned array feeds back in
    # to overplot. Two axes are read as (parl_ax, perp_ax) and the sky panel is
    # dropped, for callers that want the profiles alone (see plot_cloud_summary).
    i = result['i']
    df = result['df']
    mins = result['mins']
    rho0, r0, alpha = result['fit']
    df_parl = result['df_parl']
    mins_parl = result['mins_parl']
    fit_parl = result['fit_parl']

    if lbl == 'by r':
        lbl = f'r={round(drs[i], 2)}'

    # Built by hand rather than plt.subplots(1, 3, sharey=True): the sky panel's
    # y axis is galactic latitude, so only the two profile panels may share one.
    # sharex is gone too -- d_perp stays log, d_parl is linear and runs edge to edge.
    if type(plt_) == str:
        fig = plt.figure(figsize=(18, 5), constrained_layout=True)
        gs = fig.add_gridspec(1, 3)

        map_ax = fig.add_subplot(gs[0])
        parl_ax = fig.add_subplot(gs[1])
        perp_ax = fig.add_subplot(gs[2], sharey=parl_ax)

        axs = np.array([map_ax, parl_ax, perp_ax])
        # fig.suptitle(names[i])
    else:
        axs = np.ravel(np.asarray(plt_, dtype=object))

    if len(axs) == 2:
        map_ax = None
        parl_ax, perp_ax = axs
    else:
        map_ax, parl_ax, perp_ax = axs[:3]

    s = 100

    # Bottom of the density axis, in cm^-3. Both panels are pinned to it (they
    # share a y axis when this function builds them), and the fitted curves are
    # trimmed at it. Set just under the n_med/10 threshold everything else is cut
    # at, so the lowest drawn point is the threshold itself rather than a decade
    # of empty axis below it. Falls back to the old fixed 1e-1 when there is no
    # d_parl profile to take a median from.
    med_floor = result.get('med_parl', np.nan)
    n_floor = 0.6*med_floor/10 if np.isfinite(med_floor) else 1e-1

    # no label: the fitted parameters are in the text box below, and a legend
    # entry here would repeat them (plot_cloud_summary relabels these handles)
    perp_ax.scatter(df['d'], df['rho'], color=color, marker='.', s=s)
    perp_ax.scatter(df['d'].iloc[mins[0]], df['rho'].iloc[mins[0]], color=color, marker='+', s=s)
    perp_ax.plot(df['d'], plummer(df['d'], rho0, r0, alpha), color=color)

    err_sum = np.nansum(np.abs(plummer(df['d'], rho0, r0, alpha)-df['rho']))/np.nansum(df['rho'])

    # the fitted scale radius. r0 enters plummer only as (r/r0)^2, so a fit that
    # lands on -r0 is the same model -- mark |r0| rather than drop the line.
    # DISABLED: the |r0| rule was cluttering the d_perp panel. Uncomment to restore.
    # if mark_scales:
    #     perp_ax.axvline(abs(r0), color=color, linewidth=0.8, alpha=0.7)

    # The fitted Plummer parameters, on the panel they were fitted to. A text box
    # rather than a legend entry: plot_cloud_summary relabels this panel's legend
    # handles by their drange key, and renaming the scatter would break that.
    perp_ax.text(0.97, 0.95,
                 '\n'.join([rf'$\rho_0 = {rho0:g}$ cm$^{{-3}}$',
                            rf'$r_0 = {r0:g}$ pc',
                            rf'$\alpha = {alpha:g}$']),
                 transform=perp_ax.transAxes, ha='right', va='top',
                 fontsize=16, color=color, linespacing=1.4)

    perp_ax.loglog()
    perp_ax.set_xlabel(r'$d_\perp\ [\mathrm{pc}]$')
    perp_ax.set_ylabel(r'$n\ (cm^{-3})$')
    # perp_ax.set_xlim(1, np.nanmax(cloud_dict['b_perp'])+5)


    # one curve end to end, with p1 marked where it falls on the axis
    if df_parl is not None:
        # no label: the d_parl legend is for the reference lines' values only
        parl_ax.scatter(df_parl['d'], df_parl['rho'], color=color, marker='.', s=s)
        parl_ax.scatter(df_parl['d'].iloc[mins_parl[0]], df_parl['rho'].iloc[mins_parl[0]],
                        color=color, marker='+', s=s)
        # parl_ax.axvline(result['parl_p1'], color=color, linestyle=':', linewidth=0.75)

        # With the superGaussian fit disabled in _compute, the references drawn
        # here are flat lines at the MEDIAN of the binned profile (df_parl['rho'],
        # the same values plotted as points above) and one decade below it,
        # rather than a fitted curve. The values go in the labels so they can be
        # read straight off the panel instead of measured against the log axis.
        # n_med/10 is solid to separate it from the dashed median.
        med_parl = result.get('med_parl', np.nanmedian(df_parl['rho'].to_numpy()))
        for fac, tag, ls, al in [(1, r'$n_\mathrm{med}$', '--', 0.9),
                                 (0.1, r'$n_\mathrm{med}/10$', '-', 0.6)]:
            parl_ax.axhline(med_parl*fac, color=color, linestyle=ls, linewidth=1,
                            alpha=al,
                            label=r'%s = %.3g cm$^{-3}$' % (tag, med_parl*fac))
        # legend here rather than leaving it to the caller, so a standalone call
        # shows the values too. plot_cloud_summary re-legends this axis later and
        # rebuilds from the same labelled artists, so the entries survive that.
        parl_ax.legend(fontsize=16, frameon=False)

        # # the line is the fitted superGaussian, as the perp line is the fitted
        # # Plummer. Drawn on a dense grid rather than the bin centers -- a boxy
        # # eta puts steep walls on this curve that the bins would render as kinks.
        # if fit_parl is not None:
        #     zs = np.linspace(df_parl['d'].min(), df_parl['d'].max(), 300)
        #     ns = superGaussian_scaled(zs, *fit_parl)
        #
        #     # Cut the curve off at the bottom of the axis instead of drawing it
        #     # down to ~1e-17 in the wings. set_ylim moves the VIEW limit but not
        #     # the data limit, so those invisible points stay in the autoscale and
        #     # stretch the log-space data range over ~19 decades -- and the 5%
        #     # margin matplotlib then adds to the top is 5% of that, which pushes
        #     # the upper limit a decade past the data. Steep walls (large eta)
        #     # made it worst.
        #     parl_ax.plot(zs, np.where(ns > n_floor, ns, np.nan), color=color)

    parl_ax.set_yscale('log')
    parl_ax.set_xlabel(r'$d_\parallel\ [\mathrm{pc}]$')
    parl_ax.set_ylabel(r'$n\ (cm^{-3})$')
    # parl_ax.set_xlim(1, np.nanmax(cloud_dict['b_parl'])+5)

    # parl_ax.set_xlim(1e-1,)
    # both explicitly: perp_ax only shares y when this function built the axes,
    # not when plot_cloud_summary passes its own in
    parl_ax.set_ylim(n_floor,)
    perp_ax.set_ylim(n_floor,)

    # the taper's edges: z0 +/- L/2, where the superGaussian is down to 1/e of
    # its crest whatever eta does. z0 and the d_parl axis are both in the
    # coordinate parl_origin re-zeroed, so these plot as they are. nan when the
    # superGaussian fit failed, which draws nothing.
    z0, L = (fit_parl or [np.nan]*4)[1:3]

    if mark_scales:
        [parl_ax.axvline(z, color=color, linewidth=0.8, alpha=0.7)
         for z in z0 + np.array([-1, 1])*L/2]


    # Sky panel: the zeroth moment map, with each d_parl bin put back where it
    # sits on the spine. Contour style follows plot_moment_maps' l-b panel.
    if map_ax is not None:
        map_ax.contourf(result['mom0'], extent=result['coords'][:4],
                        levels=40, cmap='magma', extend='max', norm='log')
        map_ax.contour(result['mom0'], extent=result['coords'][:4],
                       levels=np.arange(0.5, 3.1, 0.5), colors='black',
                       linewidths=[0.75, 1.5])

        if df_parl is not None:
            # invert the re-zeroing to get back the signed spine coordinate, step
            # that far along the spine from p1, and convert the result to (l, b).
            p1, p2 = result['spine_xyz']
            u_hat = (p2-p1)/np.linalg.norm(p2-p1)

            steps = df_parl['d'].to_numpy() - result['parl_p1']
            pts = p1[np.newaxis, :] + steps[:, np.newaxis]*u_hat[np.newaxis, :]

            # xyz_2_sph branches on the sign of y, so it takes one point at a time
            lbd = np.array([xyz_2_sph(p) for p in pts])

            map_ax.plot(lbd[:, 0], lbd[:, 1], color=color, linewidth=0.75, alpha=0.6)
            map_ax.scatter(lbd[:, 0], lbd[:, 1], color=color, marker='.', s=s)
            map_ax.scatter(lbd[mins_parl[0], 0], lbd[mins_parl[0], 1],
                           color=color, marker='+', s=s)

            # p1 itself, the dotted vline's counterpart on the sky
            # p1_lb = xyz_2_sph(p1)
            # map_ax.scatter(p1_lb[0], p1_lb[1], facecolors='none', edgecolors=color, s=s)

        map_ax.set_xlabel(r'$l\ [\mathrm{deg}]$')
        map_ax.set_ylabel(r'$b\ [\mathrm{deg}]$')
        map_ax.set_xlim(result['coords'][:2])
        map_ax.set_ylim(result['coords'][2:4])

    # print('rho0', rho0, 'r0', r0, 'alpha', alpha)

    return axs, [rho0, r0, alpha]


def as_list(x):
    # A single drange (a number, or 'by r'/'full') vs a list of them.
    return list(x) if isinstance(x, (list, tuple, np.ndarray)) else [x]


def drange_lbl(drange):
    return drange if type(drange) == str else f'$\\pm${round(float(drange), 1)} pc'


def mass_per_slice(cube, d0, px_deg=0.229, mu=1.36):
    """
    Mass in each distance slice of a cube, in Msun/pc.

    The cube's pixels are angular, so the physical area a pixel subtends grows
    as d^2 -- hence the (px_deg * d)^2 factor rather than a constant. `d0` is
    the heliocentric distance of the cube's first slice, and slices are 1 pc
    apart, so the returned masses are already per unit length.

    Returns (distances [pc], mass per slice [Msun/pc]).
    From 23_incorporating_L_into_the_model.ipynb cell 10.
    """
    ds = (d0 + np.arange(cube.shape[0]))*u.pc

    mps = (np.nansum(cube, axis=(1, 2))*mu*c.m_p*u.cm**-3 *
           (px_deg*u.deg.to(u.rad)*ds)**2).to(u.Msun/u.pc)

    return ds.value, mps


def plot_cloud_summary(i, source_array, cloud_catalog, drange, mps=None, mps_d=None,
                       cloud_dict=None, result=None, color='black', lbl=None,
                       unit='Av', tube_r='px', n_jobs=1, parl_origin='edge',
                       show_spine=True, show_r_contours=False, cmap='Paired',
                       perp_slab='L', dmask=None, perp_pad=0.5,
                       iso='n_med/10', iso_color='cyan',
                       save=False, savedir=RUNS, figsize=(19, 10), dpi=150):
    """
    One cloud on a 2x3 grid: the three moment maps over the three 1-D views.

    Top row    l-b, l-d, b-d moment maps of the subcube `drange` cuts out,
               with the fitted spine drawn on each.
    Bottom row mass per slice along the whole line of sight (the `drange`
               window marked on it), then the d_parl and d_perp density
               profiles with their superGaussian and Plummer fits.

    The panels are the ones plot_moment_maps and rho_radial_profiles_plot draw,
    borrowed into one figure rather than reimplemented, so the styling stays in
    one place.

    Parameters
    ----------
    i, source_array, cloud_catalog : as everywhere else -- the cloud's row, the
        big dust cube, the table.
    drange : number, 'by r'/'full', or a LIST of them.
        Given a list, the spine and both profiles are fit once per range and
        overplotted in one colour per range, in the style of
        22_finalizing_for_paper cell 28 -- e.g. r*np.array([0.5, 1, 2, 3, 5]).
        Every range gets its own spine (the fit sees a different set of bright
        pixels), so the top row is drawn from the WIDEST range, for context,
        with the narrower spines laid over it and each range's distance window
        marked on the mass-per-slice panel in its colour. Pass n_jobs=-1: this
        is one full spine fit per range.
    mps, mps_d : optional
        A precomputed mass-per-slice curve and its distance axis, e.g. a member
        of the `masses_per_slice` list built in the notebook. Left at None the
        curve is computed here from the full-distance-range subcube, so the
        panel always shows the cloud in line-of-sight context, not just inside
        the `drange` window.
    cloud_dict, result : optional
        Precomputed ha() and rho_radial_profiles_compute() output -- one, or a
        list matching `drange` (fit_spines_over_dranges returns exactly this).
        Pass them to replot without refitting: the Monte-Carlo spine fit
        dominates the cost of this figure. Reused as given, not checked against
        i/drange.
    color, cmap : the curve colour for a single range; the colormap the per-range
        colours are drawn from for several (cell 26's plt.cm.Paired).
    show_r_contours : bool
        Overlay the d_perp / d_parl contours on the moment maps.

    Returns (fig, axs, results, fit). `results` is always a LIST, one entry per
    range, each carrying its fits under 'fit' (Plummer: rho0, r0, alpha) and
    'fit_parl' (superGaussian: rho0, z0, L, eta).

    `fit` pulls the Plummer parameters out of `results` for convenience, in the
    same shape rho_radial_profiles_plot returns them: [rho0, r0, alpha] for a
    single range, or a list of those, one per range, when several were given.

        fig, axs, results, (rho0, r0, alpha) = plot_cloud_summary(i, ...)
    """
    name = cloud_catalog.iloc[i]['Name']

    dranges = as_list(drange)

    # `dmask` replaces the distance cut rather than adding to it (get_cloud_info
    # opens the box to the full axis and lets the mask select), so a list of
    # dranges collapses to one window when a mask is given -- every entry would
    # draw the same thing. Take the first and say so.
    if dmask is not None and len(dranges) > 1:
        warnings.warn(f"dmask overrides drange, so the {len(dranges)} ranges given "
                      f"would all be identical -- using only the first ({dranges[0]!r})")
        dranges = dranges[:1]

    cloud_dicts = as_list(cloud_dict) if cloud_dict is not None else \
        [ha(i, source_array, cloud_catalog, d, n_jobs=n_jobs, dmask=dmask) for d in dranges]

    results = as_list(result) if result is not None else \
        [rho_radial_profiles_compute(i, source_array, cloud_catalog, d, tube_r,
                                     n_jobs, parl_origin, cloud_dict=cd, perp_slab=perp_slab,
                                     dmask=dmask, perp_pad=perp_pad)
         for d, cd in zip(dranges, cloud_dicts)]

    lbls = as_list(lbl) if lbl is not None else [drange_lbl(d) for d in dranges]

    # one colour per range, as cell 28 does; a lone range keeps `color` so the
    # single-range figure looks the way it did before.
    clrs = [color] if len(dranges) == 1 else \
        list(plt.get_cmap(cmap).resampled(len(dranges))(range(len(dranges))))

    fig, axs = plt.subplots(2, 3, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.suptitle(name)

    # Both profile panels plot the same quantity, n in cm^-3, so they have to be
    # read against one common axis -- the standalone figure in
    # rho_radial_profiles_plot shares it, and a plain subplots() grid does not.
    # Linked before anything is drawn so the two autoscale together, and so the
    # set_ylim(1e-1,) floor that function applies to the d_parl panel reaches
    # the d_perp one as well.
    axs[1, 2].sharey(axs[1, 1])

    # --- top row: the moment maps -----------------------------------------
    # the widest cut, so every narrower spine falls inside the panels
    widest = int(np.argmax([cd['coords'][5]-cd['coords'][4] for cd in cloud_dicts]))
    base = cloud_dicts[widest]

    # `iso='n_med/10'` takes the threshold from the profile fitted above, so the
    # outline on the maps is the same surface the d_perp cut and the mom0 mask use.
    iso_val = iso
    if isinstance(iso, str) and iso == 'n_med/10':
        iso_val = results[widest].get('med_parl', np.nan)/10

    map_axs = plot_moment_maps(base['dust_cube'], base['coords'], name, unit=unit,
                               r_perp=base['r_perp'] if show_r_contours else None,
                               r_parl=base['r_parl'] if show_r_contours else None,
                               axs=axs[0], iso=iso_val, iso_color=iso_color)

    if show_spine:
        for cd, clr in zip(cloud_dicts, clrs):
            plot_spine(map_axs, cd['spine'],
                       scolor='deepskyblue' if len(dranges) == 1 else clr)

    # --- bottom left: mass per slice --------------------------------------
    mpsax = axs[1, 0]

    if mps is None:
        # deliberately NOT masked: this is the whole-sightline context panel the
        # dmask'd window is drawn against, so it must show everything.
        longcube, longcoords = get_cloud_info(i, source_array, cloud_catalog, 'full')
        mps_d, mps = mass_per_slice(longcube, longcoords[4])

    mps = u.Quantity(mps)
    if mps_d is None:
        # fall back to the notebook's axis for a full-range curve
        mps_d = np.arange(len(mps)) + 68

    mpsax.plot(mps_d, mps.value, color='black')

    # the distance window each fit was cut from, so the maps and the profiles
    # can be read against the whole line of sight
    for cd, clr in zip(cloud_dicts, clrs):
        mpsax.vlines(cd['coords'][4:6], 0, np.nanmax(mps.value),
                     color='deepskyblue' if len(dranges) == 1 else clr)

    mpsax.set_xlabel('Heliocentric distance [pc]')
    mpsax.set_ylabel(r'Mass per slice [$M_\odot\ \mathrm{pc^{-1}}$]')
    mpsax.set_ylim(0,)

    # --- bottom middle/right: the two profiles ----------------------------
    for res, clr, lb in zip(results, clrs, lbls):
        rho_radial_profiles_plot(res, plt_=axs[1, 1:], color=clr, lbl=lb)

    # The fitted parameters are annotated by rho_radial_profiles_plot itself now
    # -- rho0/r0/alpha in a text box on the d_perp panel, n_med and n_med/10 in
    # the d_parl legend -- so there is nothing to relabel here. Repeating them
    # would put two copies of the same numbers on one panel.
    #
    # The mass-per-slice colour key used to harvest handles from the d_perp
    # scatter; that scatter is unlabelled now, so build proxies instead. Same
    # swatch colours, and it no longer depends on another panel's labelling.
    if len(dranges) > 1:
        mpsax.legend([plt.Line2D([], [], color=clr, marker='.', linestyle='none')
                      for clr in clrs], list(lbls),
                     fontsize=11, frameon=False)

    # `save` writes {savedir}/{cloud}_projmaps_profiles -- no extension, so
    # matplotlib's default (png) applies, matching the savefig convention in
    # get_best_leaf. `name` is the catalog string verbatim, so clouds with a
    # space in the name ('Orion B') produce a filename with a space in it.
    if save:
        os.makedirs(savedir, exist_ok=True)
        fig.savefig(os.path.join(savedir, f'{name}_projmaps_profiles'),
                    bbox_inches='tight', dpi=dpi)

    # Plummer parameters lifted out of `results`. Bare triple for a lone range,
    # matching how `color`/`clrs` and `lbl`/`lbls` behave above; a list of triples
    # when several ranges were asked for.
    fit = [list(res['fit']) for res in results]
    if len(dranges) == 1:
        fit = fit[0]

    # return fig, axs, results, fit
    return fit


# ============================================================================
# Edenhofer HEALPix cubes, catalog, distance windows, sample_summary
#   (from choosing_distances_from_healpix.py)
# ============================================================================
# Rebuild the `<Name>_choosing_distances` figures straight from the native
# Edenhofer+2023 HEALPix release instead of the pre-reprojected
# `edenhofer_lbd_wrap.fits` cube.
#
# Why bother: edenhofer_lbd_wrap.fits is a derived product -- someone already
# made choices about angular interpolation, the radial resampling onto a 1 pc
# grid, and the E -> n_H conversion, and those choices are only recoverable from
# a HISTORY card. Going back to edenhofer_healpix.fits makes all three explicit
# and drops one resampling step from the chain.
#
# The sampling recipe reproduces edenhofer_lbd_wrap.fits to ~0.2% (verified
# against it directly, see `validate()`):
#
#     n_H[cm^-3](l, b, d) = K * interp_r( bilinear_healpix(E, l, b) )
#     K = 2.8 * 2.2e21 / (1 pc in cm)          # E -> A_V -> n_H, as A_to_n(E*2.8)
#
# Run through dust3d_pipeline.py --steps fits.

# --- the l/b/d grid ---------------------------------------------------------
# Taken from the WCS of edenhofer_lbd_wrap.fits rather than the [210, -90.2, 68]
# defaults hardcoded in get_cloud_info(), which are rounded: the true grid is
# CDELT 0.22900763 deg with b starting at -90.0 (not -90.2) and d at 69 pc (not
# 68). Sampling HEALPix directly means we can just use the exact values.
NSIDE = 256


CD    = 0.22900763358778625          # deg per pixel, both axes


L0    =  210.0006                    # l of pixel x = 0, decreasing


B0    =  -90.0                       # b of pixel y = 0, increasing


D0    =  69.0                        # pc of slice z = 0


ND    =  1182                        # slices, 1 pc apart


ds = D0 + np.arange(ND)              # heliocentric distance of each slice, pc


PC_CM    = u.pc.to(u.cm)                    # 3.0857e18, the 1 pc slice depth


K_E_TO_N = 2.8 * 2.2e21 / PC_CM             # 1996.32 cm^-3 per (E / pc)


ALIASES = {'Aquila South': 'Aquila_South', 'Corona Australis': 'Corona_Australis',
           'Monoceros OB1': 'Monoceros_OB1', 'Monoceros R2': 'Monoceros_R2',
           'Orion A': 'Orion_A', 'Orion B': 'Orion_B',
           'Camelopardalis': 'Cam', 'λ-Orionis': 'lambda_Orionis'}


D_MAX = 500.0        # pc; the catalog cut. Cell 22's note: "restrict to clouds


EXCLUDE = {'λ-Orionis'}   # 2026-10-14: a ring/shell around the lambda Ori H II


# Adopted cloud-defining volume-density threshold (cm^-3); see
# n10_threshold_rationale.md. Was 7 through 2026-09-30.
N_ISO = 10.0


def load_catalog(d_max=D_MAX, exclude=EXCLUDE):
    table_mi = pd.read_csv(CAT)
    table_mi = table_mi[~table_mi['Name'].isin(exclude)]
    return table_mi[table_mi['d_mean(pc)'] < d_max]


def box_grid(row, pad=0.0):
    """The l/b pixel centres of one catalog box, on the Edenhofer grid.

    `pad` grows the box by that many degrees a side, matching get_cloud_info's
    own pad argument. rho_radial_profiles_compute re-slices with perp_pad=0.5,
    so a proxy standing in for the full-sky cube has to carry that margin.

    Index ranges come from coord_idx with the same rounded origins the existing
    pipeline uses, so the boxes line up pixel-for-pixel with the old cubes; the
    centres themselves are then evaluated on the exact grid.
    """
    lmax, lmin, bmin, bmax = row[['l_max(deg)', 'l_min(deg)',
                                  'b_min(deg)', 'b_max(deg)']].to_numpy().astype(float)
    xo, xi = coord_idx([lmax + pad, lmin - pad], -0.229, 210)
    yo, yi = coord_idx([bmin - pad, bmax + pad],  0.229, -90.2)
    lc = L0 - CD * np.arange(xo, xi)
    bc = B0 + CD * np.arange(yo, yi)
    return lc, bc, np.array([lmax, lmin, bmin, bmax]), (xo, yo)


_BOX_COLS = ['l_max(deg)', 'l_min(deg)', 'b_min(deg)', 'b_max(deg)']


@functools.lru_cache(maxsize=2)
def _stored(cloud):
    path = os.path.join(CUBE_DIR, f'{cloud}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} missing -- run extract_padded_sample_cubes.py')
    z = np.load(path)
    return {'cube': z['cube'], 'indep': [int(k) for k in z['indep']],
            'xo': int(z['xo']), 'yo': int(z['yo']), 'pad': float(z['pad']),
            'row': z['row']}


def load_stored(table1, cloud):
    """Stored (nsamp, ND, nb, nl) cube + bookkeeping for one cloud, checked
    against the catalog row it is being asked for: a box edited in the CSV after
    the cubes were extracted must not be served silently."""
    row = table1.iloc[list(table1['Name']).index(cloud)]
    st = _stored(cloud)
    _, _, _, (xo, yo) = box_grid(row, pad=st['pad'])
    if (xo, yo) != (st['xo'], st['yo']) or not np.allclose(
            row[_BOX_COLS].to_numpy().astype(float), st['row'][:4]):
        raise ValueError(f'{cloud}: stored cube was extracted for a different box '
                         f'than the catalog now gives; rerun extract_padded_sample_cubes.py')
    return st


def _unpadded(arr, row, st):
    """Crop the trailing (b, l) axes of a stored-box array to the catalog box."""
    lc, bc, _, (xo, yo) = box_grid(row)
    y0, x0 = yo - st['yo'], xo - st['xo']
    return arr[..., y0:y0 + len(bc), x0:x0 + len(lc)]


def _box_key(table1):
    return np.array([[*r[_BOX_COLS].astype(float)] for _, r in table1.iterrows()])


def build_cubes(table1, cache=CACHE):
    """The MEAN (nd, nb, nl) cube per cloud on the unpadded catalog box.

    The mean of the stored posterior samples (it reproduces the released mean map
    to float32 rounding, ~1e-7 relative), cropped out of the padded cube. Cached
    in `cache`, keyed on each cloud's box so an edited catalog forces a rebuild.
    """
    names = list(table1['Name'])
    key = _box_key(table1)
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        cached = [str(n) for n in z['names']]
        if (names == cached and 'boxes' in z.files and z['boxes'].shape == key.shape
                and np.allclose(z['boxes'], key)):
            print(f'[cache] {cache}')
            return [z[f'c{i}'] for i in range(len(names))]
        print(f'[cache] {cache} does not match this catalog; rebuilding')

    cubes = []
    for i, nm in enumerate(names):
        st = load_stored(table1, nm)
        cubes.append(np.ascontiguousarray(
            _unpadded(st['cube'].mean(axis=0), table1.iloc[i], st), dtype=np.float32))
        print(f'  [{i + 1}/{len(names)}] {nm}', flush=True)
    if cache:
        np.savez_compressed(cache, names=np.array(names, dtype=object), boxes=key,
                            **{f'c{i}': cb for i, cb in enumerate(cubes)})
        print(f'[cache] wrote {cache}')
    return cubes


SAMPLES   = 'edenhofer_full_samples.fits'


def antithetic_pairs(samples):
    """(nsamp, npix) log-normal samples -> [(i, j), ...] partner pairs.

    The Edenhofer samples are drawn antithetically: every draw is shipped
    alongside its reflection through the mean, so log s_2i = mu + d_i and
    log s_2i+1 = mu - d_i. The two members of a pair correlate at exactly -1,
    which is how they are found here rather than by assuming the file order.
    """
    r = np.log(np.asarray(samples, dtype=float))
    r = r - r.mean(0)
    C = np.corrcoef(r)
    np.fill_diagonal(C, 0.0)
    pairs, seen = [], set()
    for i in range(len(C)):
        if i in seen:
            continue
        j = int(np.argmin(C[i]))
        if C[i, j] > -0.9:
            raise ValueError(f'sample {i} has no antithetic partner '
                             f'(best corr {C[i, j]:+.3f}); the file layout has changed')
        pairs.append((i, j))
        seen |= {i, j}
    return pairs


def build_sample_profiles(table1, cache=SCACHE):
    """Mass-per-slice profile for every posterior sample, per cloud.

    Summed over each cloud's unpadded catalog box, from the stored sample cubes
    (same unit chain as masses_per_slice). Cached, keyed on the boxes.

    Returns (profiles, indep) where profiles is a list of (nsamp, ND) arrays in
    Msun/pc and indep is the list of mutually independent sample indices (one
    from each antithetic pair).
    """
    names = list(table1['Name'])
    key = _box_key(table1)
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        cached = [str(n) for n in z['names']]
        if (names == cached and 'boxes' in z.files and z['boxes'].shape == key.shape
                and np.allclose(z['boxes'], key)):
            print(f'[cache] {cache}')
            return [z[f'p{i}'] for i in range(len(names))], list(z['indep'])
        print(f'[cache] {cache} does not match this catalog; rebuilding')

    geom = (1.36 * c.m_p * u.cm**-3 * (CD * u.deg.to(u.rad) * ds * u.pc)**2).to(u.Msun / u.pc).value
    profiles, indep = [], None
    for i, nm in enumerate(names):
        st = load_stored(table1, nm)
        box = _unpadded(st['cube'], table1.iloc[i], st)        # (nsamp, ND, nb, nl)
        profiles.append(np.nansum(box, axis=(2, 3), dtype=np.float64) * geom)
        if indep is None:
            indep = st['indep']
        elif indep != st['indep']:
            raise ValueError(f'{nm}: independent-sample set differs between clouds')
        print(f'  [{i + 1}/{len(names)}] {nm}', flush=True)

    if cache:
        np.savez_compressed(cache, names=np.array(names, dtype=object), boxes=key,
                            indep=np.array(indep),
                            **{f'p{i}': pr for i, pr in enumerate(profiles)})
        print(f'[cache] wrote {cache}')
    return profiles, indep


PAD_DEG = 0.75      # margin on each cloud box; rho_radial_profiles_compute


class CloudBoxProxy:
    """Presents one cloud's padded box as if it were the whole-sky l/b/d cube.

    get_cloud_info/get_subcube only ever reach source_array through
    `bigcube[zo:zi, yo:yi, xo:xi]` and `.shape[0]`, so translating those global
    indices into a stored box lets every existing routine -- ha(),
    rho_radial_profiles_compute(), plot_cloud_summary() -- run on a single
    posterior sample without materialising a 6.8 GB cube per draw.
    """

    def __init__(self, box, xo, yo, full_shape=(ND, 786, 1834)):
        self.box, self.xo, self.yo = box, xo, yo
        self.shape = full_shape

    def __getitem__(self, key):
        zs, ys, xs = key
        y0, y1 = ys.start - self.yo, ys.stop - self.yo
        x0, x1 = xs.start - self.xo, xs.stop - self.xo
        if y0 < 0 or x0 < 0 or y1 > self.box.shape[1] or x1 > self.box.shape[2]:
            raise IndexError(f'request [{ys}, {xs}] falls outside the stored box; '
                             f'rebuild with a larger PAD_DEG')
        return self.box[zs, y0:y1, x0:x1]


def build_sample_cubes(table1, cloud, pad=PAD_DEG, samples=None):
    """Padded l/b/d cubes for one cloud, one per posterior sample.

    Read from the stored cubes (which carry PAD_STORED degrees), so `pad` is only
    checked, not used to cut: CloudBoxProxy translates global pixel indices into
    whatever box it is handed, and a request inside the stored box just works.

    Returns (cubes, indep, (xo, yo)) with cubes of shape (nsamp, ND, nb, nl)
    in cm^-3, matching what get_cloud_info would slice out of the full cube.
    """
    if pad > PAD_STORED + 1e-9:
        raise ValueError(f'pad {pad:g} deg exceeds the {PAD_STORED:g} deg the stored '
                         f'cubes carry; re-extract with a larger --pad')
    st = load_stored(table1, cloud)
    indep = st['indep'] if samples is None else list(samples)
    return st['cube'][indep], indep, (st['xo'], st['yo'])


DLNR = 5.6217e-03   # log-spacing of the Edenhofer radial grid, read off its


def npix_for_distance(d, n_res=3.0):
    """min_npix for a cloud at distance d, in 1 pc slices.

    The dendrogram's width cut should be a resolution requirement, not a fixed
    number of parsecs: the radial element grows from 0.56 pc at 100 pc to
    2.8 pc at 500 pc, so a flat cut is far too strict far away. n_res is how
    many resolution elements wide a structure must be -- 2 is Nyquist, 3 is
    safe. The max(1.0, ...) is because the cube is resampled onto a 1 pc grid,
    so inside ~178 pc the grid rather than the native bin is the limit.

    The flat min_npix=30 this replaces was inert on 13 of 14 clouds and
    catastrophic on the 14th: Camelopardalis' 213 pc peak is 28 pc wide above
    the threshold, so a 30 pc cut deleted it, criterion 1 could not fire, and
    the selection fell through to a leaf 220 pc away.
    """
    return int(np.ceil(n_res * max(1.0, DLNR * d)))


def robust_min_value(prof, nsigma=3.0, q=25.0):
    """A per-sightline dendrogram floor: background level + nsigma * scatter.

    Both estimates are taken from the lowest quartile rather than from the whole
    profile. The obvious choice, median + 3*MAD, silently assumes the cloud
    occupies a small fraction of the distance axis -- and for the clouds that
    fill a lot of it (California, Aquila South) the median sits ON the cloud, so
    the threshold lands at ~70% of the peak and cuts into the structure instead
    of its skirt. Estimating from the quiet quartile keeps a bright, wide cloud
    from dragging its own floor up.

    The 1.4826 rescales the MAD so it equals the standard deviation for Gaussian
    noise, which is what makes "3 sigma" mean the usual thing.

    Note this trims leaf skirts; it never splits a peak. min_value only prunes
    below the threshold, so it cannot create a local maximum that was not
    already in the profile.
    """
    prof = np.asarray(prof, dtype=float)
    floor = np.nanpercentile(prof, q)
    lo = prof[prof <= floor]
    sigma = 1.4826 * np.nanmedian(np.abs(lo - floor))
    return floor + nsigma * sigma


def select_leaf(prof, d_mean, min_value=None, min_npix=None, min_delta=20):
    """Dendrogram the mass profile and apply the two selection criteria.

    Returns (dend, j, fracs, spans, holds_dmean, min_value_used); j indexes
    dend.leaves. min_value=None means use robust_min_value(prof);
    min_npix=None means use npix_for_distance(d_mean).
    """
    if min_value is None:
        min_value = robust_min_value(prof)
    if min_npix is None:
        min_npix = npix_for_distance(d_mean)
    dend = Dendrogram.compute(prof, min_value=min_value,
                              min_npix=min_npix, min_delta=min_delta)
    if not dend.leaves:
        return dend, None, None, None, None, min_value

    total = np.nansum(prof)
    fracs = np.array([np.nansum(prof * l.get_mask()) / total for l in dend.leaves])
    spans = [(ds[l.get_mask()].min(), ds[l.get_mask()].max()) for l in dend.leaves]
    holds = np.array([lo <= d_mean <= hi for lo, hi in spans])

    if holds.any():
        j = int(np.flatnonzero(holds)[fracs[holds].argmax()])
    else:
        j = int(fracs.argmax())
    return dend, j, fracs, spans, holds, min_value


def validate(table1, cubes, n=4):
    """Compare the HEALPix-derived cubes against edenhofer_lbd_wrap.fits."""
    f = fits.open(DATA + 'edenhofer_lbd_wrap.fits', memmap=True)
    print(f'{"cloud":18s} {"median ratio":>13s} {"frac scatter":>13s}')
    for i in range(0, len(table1), max(1, len(table1) // n)):
        lmax, lmin, bmin, bmax = table1.iloc[i][['l_max(deg)', 'l_min(deg)',
                                                 'b_min(deg)', 'b_max(deg)']].to_numpy().astype(float)
        xo, xi = coord_idx([lmax, lmin], -0.229, 210)
        yo, yi = coord_idx([bmin, bmax],  0.229, -90.2)
        ref = np.asarray(f[0].data[:, yo:yi, xo:xi])
        r = ref / cubes[i]
        m = np.nanmedian(r)
        print(f'{table1["Name"].iloc[i]:18s} {m:13.4f} {np.nanstd(r) / m:13.4f}')


def mass_profile(cube):
    return (np.nansum(cube, axis=(1, 2)) * 1.36 * c.m_p * u.cm**-3
            * (CD * u.deg.to(u.rad) * ds * u.pc)**2).to(u.Msun / u.pc).value


def mean_padded_box(table1, cloud, pad=PAD_DEG):
    """The MEAN map (average of the stored samples) on the stored padded grid.

    Returns the whole stored box, PAD_STORED degrees, whatever `pad` is asked for:
    callers index it through CloudBoxProxy, which works in global pixel offsets.
    """
    if pad > PAD_STORED + 1e-9:
        raise ValueError(f'pad {pad:g} deg exceeds the {PAD_STORED:g} deg the stored '
                         f'cubes carry; re-extract with a larger --pad')
    st = load_stored(table1, cloud)
    return st['cube'].mean(axis=0), (st['xo'], st['yo'])


def spine_curve(p1, p2, anchor, R, n=80):
    """Points along a spine, in (l, b, d), for drawing on the projection maps.

    Sampled in xyz then converted, because the line is straight there and curved
    in (l, b, d) -- drawing a straight segment between two lbd endpoints would
    be visibly wrong over a long spine.

    R may be a scalar half-length or a (t_neg, t_pos) pair, which is what
    spine_extent returns: the two directions generally run out to different
    distances.
    """
    u = (p2 - p1) / np.linalg.norm(p2 - p1)
    tn, tp = (R, R) if np.isscalar(R) else R
    pts = [xyz_2_sph(anchor + t * u) for t in np.linspace(-tn, tp, n)]
    return np.array(pts)          # columns l, b, d


def spine_extent(p1, p2, anchor, box, xo, yo, iso_val, dlim,
                 step=0.25, t_max=400.0):
    """How far the spine runs before it leaves the n > iso_val region.

    Walks outward from the anchor in each direction and stops at the first point
    where the interpolated volume density drops below iso_val -- the same
    n_med/10 threshold whose projected silhouette is the limegreen contour on
    the maps -- or where the ray leaves the box or the distance window. The two
    directions are returned separately since they rarely run out equally.

    Stopping at the FIRST crossing is deliberate: a spine that re-enters denser
    gas further out has left the cloud in between, and bridging that gap would
    draw a line through material the cloud does not own.
    """
    u = (p2 - p1) / np.linalg.norm(p2 - p1)
    nz, ny, nx = box.shape
    out = []
    for sgn in (-1.0, 1.0):
        t = 0.0
        while t < t_max:
            t_try = t + step
            l, b, d = xyz_2_sph(anchor + sgn * t_try * u)
            xi = (L0 - l) / CD - xo
            yi = (b - B0) / CD - yo
            zi = d - D0
            if not (0 <= xi <= nx - 1 and 0 <= yi <= ny - 1 and 0 <= zi <= nz - 1):
                break
            if not (dlim[0] <= d <= dlim[1]):
                break
            if map_coordinates(box, [[zi], [yi], [xi]], order=1)[0] < iso_val:
                break
            t = t_try
        out.append(t)
    return out[0], out[1]


def spine_slab(i, source, table1, dmask, spine_xyz, box, xo, yo, iso_val, dlim):
    """The walked extent of a spine as a (lo, hi) interval in the raw d_parl
    coordinate, (x - p1).u, for rho_radial_profiles_compute(perp_slab=...).

    Anchor = centroid of the top-1% pixels projected onto the line (as for the
    40-seed spines); if the density there is below iso_val, restart from the
    densest point on the line inside the distance window (as walk_spine does in
    plot_cylinder_slice). Then spine_extent walks out to iso_val in `box`.
    Returns None if no point on the line reaches iso_val.
    """
    p1, p2 = (np.asarray(q, float) for q in spine_xyz)
    u = (p2 - p1) / np.linalg.norm(p2 - p1)
    dats = get_99th_percs(i, source, table1, 'full', which='99', dmask=dmask).to_numpy()
    anchor = anchor_point(p1, p2, dats)

    nz, ny, nx = box.shape
    def n_at(pts):
        lbd = np.array([xyz_2_sph(q) for q in pts])
        xi, yi, zi = (L0 - lbd[:, 0]) / CD - xo, (lbd[:, 1] - B0) / CD - yo, lbd[:, 2] - D0
        ok = ((xi >= 0) & (xi <= nx - 1) & (yi >= 0) & (yi <= ny - 1)
              & (zi >= 0) & (zi <= nz - 1) & (lbd[:, 2] >= dlim[0]) & (lbd[:, 2] <= dlim[1]))
        n = np.full(len(pts), np.nan)
        n[ok] = map_coordinates(box, [zi[ok], yi[ok], xi[ok]], order=1)
        return n

    if not n_at([anchor])[0] >= iso_val:
        ts = np.arange(-400.0, 400.25, 0.25)
        nline = n_at([anchor + t * u for t in ts])
        if np.all(np.isnan(nline)) or np.nanmax(nline) < iso_val:
            return None
        anchor = anchor + ts[np.nanargmax(nline)] * u
    t_neg, t_pos = spine_extent(p1, p2, anchor, box, xo, yo, iso_val, dlim)
    s0 = float(np.dot(anchor - p1, u))
    return (s0 - t_neg, s0 + t_pos)


def draw_spine_curve(map_axs, lims, crv, **kw):
    """Draw one spine on the three projection panels, clipped to each.

    Clipping is necessary rather than cosmetic: xyz_2_sph flips branch at
    l = +-180, so a spine extended far enough past the box reappears on the
    other side and paints a spurious line across the figure. Several of these
    clouds sit near the anti-centre.
    """
    for ax, (cx, cy), (xl, yl) in zip(map_axs, [(0, 1), (0, 2), (2, 1)], lims):
        x, y = crv[:, cx].copy(), crv[:, cy].copy()
        inside = ((x >= min(xl)) & (x <= max(xl)) &
                  (y >= min(yl)) & (y <= max(yl)))
        x[~inside] = np.nan; y[~inside] = np.nan
        ax.plot(x, y, **kw)


def sample_summary(table1, cloud, cubes, sprofs, n_jobs=1, save=True,
                   suffix='_projmaps_profiles_samples', min_value=None,
                   n_seeds=40, min_npix=None, n_iso=None,
                   lb_pad=1.0, d_edge_pad=10.0, d_search=400.0, lb_search_pad=PAD_STORED,
                   spine_slab_cut=True):
    """projmaps_profiles for one cloud, with the posterior samples overlaid.

    The spine is REFIT on every sample, so the spread carries both the density
    uncertainty and the geometric uncertainty -- each draw gets its own spine,
    its own d_parl / d_perp profiles and its own Plummer fit. The mean map's
    spine is still drawn underneath for reference.

    Returns (fig, F, stats) where F is the per-sample Plummer parameters and
    stats the pooled spine descriptors.
    """
    i = list(table1['Name']).index(cloud)
    prof = mass_profile(cubes[i])
    d_mean = float(table1.iloc[i]['d_mean(pc)'])
    dend, j, fr, spn, holds, mv = select_leaf(prof, d_mean, min_value, min_npix)
    leaf_mask = dend.leaves[j].get_mask()

    # the stored box must cover the widest thing asked of the proxy: the maps
    # are drawn at lb_pad, which can exceed the PAD_DEG the box defaults to
    mbox, (xo, yo) = mean_padded_box(table1, cloud, pad=max(PAD_DEG, lb_pad))
    mproxy = CloudBoxProxy(mbox, xo, yo)
    scubes, indep, (sx, sy) = build_sample_cubes(table1, cloud)

    # ---- THE CLOUD'S DISTANCE WINDOW ------------------------------------
    # Defined by the gas rather than by the dendrogram leaf: the contiguous run
    # of distance slices whose sky box reaches `n_iso` anywhere, extended by
    # `d_edge_pad` pc each side. Free to come out NARROWER than the leaf as well
    # as wider -- the leaf is a structure in mass per slice and can span slices
    # holding no dense gas at all (Perseus loses 96 pc; Orion B gains 227).
    # Grown outward from the densest slice INSIDE the leaf, so an unrelated
    # clump further along the sightline cannot capture it; search capped at
    # `d_search` pc a side.
    # This mask is what everything downstream uses -- mass, the d_parl/d_perp
    # profiles, the Plummer fits, the spine refit and the maps. `leaf_mask` is
    # kept only to draw the leaf's own edges for reference.
    idx = np.flatnonzero(leaf_mask)
    lo_i, hi_i = int(idx[0]), int(idx[-1])
    cap = int(round(d_search))
    lo_s, hi_s = max(0, lo_i - cap), min(len(leaf_mask) - 1, hi_i + cap)
    m_s = np.zeros_like(leaf_mask); m_s[lo_s:hi_s + 1] = True
    big, _ = get_cloud_info(i, mproxy, table1, 'full', dmask=m_s, pad=lb_pad)
    nmax = np.nanmax(big, axis=(1, 2))
    thr = float(n_iso) if n_iso is not None else np.nan
    lo_c, hi_c = lo_i, hi_i
    if np.isfinite(thr):
        inside = nmax[lo_i - lo_s:hi_i - lo_s + 1]
        if np.nanmax(inside) >= thr:
            k = int(np.nanargmax(inside)) + (lo_i - lo_s)
            a_, b_ = k, k
            while a_ - 1 >= 0 and nmax[a_ - 1] >= thr:
                a_ -= 1
            while b_ + 1 < len(nmax) and nmax[b_ + 1] >= thr:
                b_ += 1
            lo_c, hi_c = a_ + lo_s, b_ + lo_s
        else:
            print(f'  !! no slice in the leaf reaches n = {thr:g} cm^-3; '
                  f'falling back to the leaf window')
    pad_i = int(round(d_edge_pad))
    lo_d = max(0, lo_c - pad_i)
    hi_d = min(len(leaf_mask) - 1, hi_c + pad_i)
    dmask = np.zeros_like(leaf_mask); dmask[lo_d:hi_d + 1] = True
    print(f'  window: leaf {ds[lo_i]:.0f}-{ds[hi_i]:.0f} pc | '
          f'n>={thr:g} run {ds[lo_c]:.0f}-{ds[hi_c]:.0f} | '
          f'ADOPTED {ds[lo_d]:.0f}-{ds[hi_d]:.0f} pc '
          f'({(hi_d-lo_d)-(hi_i-lo_i):+d} pc vs leaf)')

    cd_mean = ha(i, mproxy, table1, 'full', dmask=dmask, n_jobs=n_jobs)
    # the d_perp profile (and so the Plummer fit) only from voxels between the
    # two walked ends of the spine it is measured around (2026-10-14); each
    # sample below gets its own spine's ends, walked on its own cube
    dlim_s = (float(ds[lo_d]), float(ds[hi_d]))
    slab_of = lambda cd_, box_, bx_, by_, src_: (
        spine_slab(i, src_, table1, dmask, cd_['spine_xyz'], box_, bx_, by_,
                   float(n_iso), dlim_s)
        if (spine_slab_cut and n_iso is not None) else None)
    res_mean = rho_radial_profiles_compute(i, mproxy, table1, 'full',
                                             cloud_dict=cd_mean, dmask=dmask,
                                             perp_nmin=n_iso,
                                             perp_slab=slab_of(cd_mean, mbox, xo, yo, mproxy))

    # ---- MEAN SPINE DIRECTION: fit on the small (lb_pad) box, UNCHANGED ----
    # This is what determines pa_lb/inc_ld/inc_bd -- nothing about the drawing
    # or box size below feeds back into it.
    rows0, dats0 = spine_candidates(i, mproxy, table1, dmask, n_seeds, n_jobs=-1)
    r00 = rows0[rows0[:, 0].argmin()]
    p1m, p2m = r00[1:4], r00[4:7]
    a00 = anchor_point(p1m, p2m, dats0)

    # ---- WALK TO n_iso FIRST; THAT DETERMINES THE DRAWN LONGITUDE RANGE ----
    # Previously the walk ran inside the same lb_pad-sized box used for display,
    # so a cloud whose straight-line spine reached further than lb_pad got
    # silently cut off -- e.g. Chamaeleon's spine stopped at n = 17 and n = 10
    # cm^-3, comfortably above the n_iso = 5 threshold, simply because it hit
    # the box wall in longitude first. The green n_iso silhouette kept going
    # past that point because it is a max over ALL longitude in the box, while
    # the spine is one straight line -- so the two are not the same test.
    # Fix: walk in a generously wide box FIRST (lb_search_pad, capped), then
    # size the DRAWN box from where the walk actually stopped, growing the
    # display beyond the default lb_pad only for the clouds that need it.
    iso_val = (float(n_iso) if n_iso is not None
               else float(res_mean.get('med_parl', np.nan)) / 10.0)
    dlim = (float(ds[lo_d]), float(ds[hi_d]))
    dl_deg = abs(table1.iloc[i]['l_max(deg)'] - table1.iloc[i]['l_min(deg)'])
    R = max(ds[hi_d] - ds[lo_d], np.deg2rad(dl_deg) * d_mean)   # fallback only

    if np.isfinite(iso_val):
        wbox, (xo_w, yo_w) = mean_padded_box(table1, cloud, pad=lb_search_pad)
        t_neg, t_pos = spine_extent(p1m, p2m, a00, wbox, xo_w, yo_w, iso_val, dlim)
    else:
        wbox, xo_w, yo_w = None, 0, 0
        t_neg, t_pos = R, R

    u_m = (p2m - p1m) / np.linalg.norm(p2m - p1m)
    lmax_c = float(table1.iloc[i]['l_max(deg)']); lmin_c = float(table1.iloc[i]['l_min(deg)'])
    bmin_c = float(table1.iloc[i]['b_min(deg)']); bmax_c = float(table1.iloc[i]['b_max(deg)'])
    need = 0.0
    for sgn, tt in ((-1.0, t_neg), (1.0, t_pos)):
        l_, b_, d_ = xyz_2_sph(a00 + sgn * tt * u_m)
        need = max(need, lmin_c - l_, l_ - lmax_c, bmin_c - b_, b_ - bmax_c)
    lb_pad_draw = float(np.clip(need + 0.15, lb_pad, lb_search_pad))
    capped = (need + 0.15) > lb_search_pad
    print(f'  spine reach: {lb_pad_draw:.2f} deg beyond the catalog box'
          + (f'  (CAPPED at {lb_search_pad:g} -- did not fully close, treat as'
             ' a lower bound)' if capped else ''))

    # the wide proxy (built above for the walk) already has enough margin to
    # serve lb_pad_draw as well, since lb_pad_draw <= lb_search_pad by
    # construction -- reuse it rather than building a third box
    if wbox is None:
        wbox, (xo_w, yo_w) = mean_padded_box(table1, cloud, pad=lb_search_pad)
    wproxy = CloudBoxProxy(wbox, xo_w, yo_w)

    # Draw the moment maps over this box: it already reaches half a degree
    # beyond the catalog box from rho_radial_profiles_compute's own perp_pad,
    # and now also reaches out to wherever the spine itself required. Only
    # 'dust_cube'/'coords' are swapped: plot_cloud_summary reads the spine
    # from a separate key, so the fit itself still comes from the unpadded
    # box and no fitted number changes.
    pad_cube, pad_coords = get_cloud_info(i, wproxy, table1, 'full',
                                            dmask=dmask, pad=lb_pad_draw)
    cd_draw = dict(cd_mean)
    cd_draw['dust_cube'], cd_draw['coords'] = pad_cube, pad_coords

    plt.close('all')
    fit = plot_cloud_summary(
        i, mproxy, table1, 'full', dmask=dmask, cloud_dict=cd_draw,
        result=res_mean, mps=prof, mps_d=ds, save=False, n_jobs=n_jobs,
        iso=(n_iso if n_iso is not None else 'n_med/10'),
        iso_color='limegreen',   # default is cyan, same colour as the spine
        show_spine=False)        # drawn below instead, see the note there
    fig = plt.gcf()
    axs = np.array(fig.axes[:6]).reshape(2, 3)
    fig.suptitle(f'{cloud}  --  posterior samples (spine refit per sample)')

    for pr in sprofs[i]:
        axs[1, 0].plot(ds, pr, color='tab:orange', lw=0.6, alpha=0.7, zorder=0)

    lims = [(a.get_xlim(), a.get_ylim()) for a in axs[0]]

    # The mean spine is drawn here rather than by plot_spine, so that it and the
    # samples are drawn under identical rules. plot_spine puts down a quiver of
    # length |vec| -- the separation of whichever two endpoints the optimiser
    # left behind, 1.4 deg on Camelopardalis against the ~25 deg the sample
    # curves span -- and draws it straight, while the true projection of a line
    # that is straight in xyz curves by ~1 deg over that span. Both differences
    # made the fan look like disagreement when most of it was just two different
    # drawing conventions.
    # ext() now walks in the SAME wide box the mean spine's own extent came
    # from, so a sample whose fitted line points further than lb_pad_draw is
    # not cut off either -- only the mean spine sizes the box, but every
    # curve's own true n_iso crossing is what gets drawn.
    # Each sample's line is walked through THAT sample's own cube (2026-10-14;
    # previously every sample read the mean map wbox). build_sample_cubes returns
    # the full PAD_STORED box, so the walk has the same room as the mean's.
    ext = lambda q1, q2, an, box=wbox, bx=xo_w, by=yo_w: (
        spine_extent(q1, q2, an, box, bx, by, iso_val, dlim)
        if np.isfinite(iso_val) else (R, R))

    mean_curve = spine_curve(p1m, p2m, a00, (t_neg, t_pos))

    fits_s, descs = [], []
    # the two profile panels share a y axis and autoscale on the mean points
    # alone, so sample curves that run below the lowest mean bin get clipped.
    # Collect everything actually drawn and set the limits from that.
    yvals = [res_mean['df_parl'].iloc[:, 1].to_numpy(),
             res_mean['df'].iloc[:, 1].to_numpy()]
    for k in range(len(indep)):
        pxy = CloudBoxProxy(scubes[k], sx, sy)
        cd = ha(i, pxy, table1, 'full', dmask=dmask, n_jobs=n_jobs)
        r = rho_radial_profiles_compute(i, pxy, table1, 'full',
                                          cloud_dict=cd, dmask=dmask,
                                          perp_nmin=n_iso,
                                          perp_slab=slab_of(cd, scubes[k], sx, sy, pxy))
        if r['slab'] is None or r['slab_thin']:
            print(f'  !! sample {indep[k]}: no spine slab (fit on the full box)')
        axs[1, 1].plot(r['df_parl'].iloc[:, 0], r['df_parl'].iloc[:, 1],
                       color='tab:orange', lw=0.7, alpha=0.7, zorder=0)
        axs[1, 2].plot(r['df'].iloc[:, 0], r['df'].iloc[:, 1],
                       color='tab:orange', lw=0.7, alpha=0.7, zorder=0)
        yvals += [r['df_parl'].iloc[:, 1].to_numpy(), r['df'].iloc[:, 1].to_numpy()]
        fits_s.append(r['fit'])

        rows, dats_arr = spine_candidates(i, pxy, table1, dmask, n_seeds, n_jobs=-1)
        r0 = rows[rows[:, 0].argmin()]
        p1, p2 = r0[1:4], r0[4:7]
        a0 = anchor_point(p1, p2, dats_arr)
        descs.append(spine_descriptors(np.array(xyz_2_sph(a0)),
                                       lbd_tangent(p1, p2, a0)))
        draw_spine_curve(axs[0], lims,
                         spine_curve(p1, p2, a0, ext(p1, p2, a0, scubes[k], sx, sy)),
                         color='cyan', lw=1.0, alpha=0.9, zorder=6)

    draw_spine_curve(axs[0], lims, mean_curve, color='deepskyblue', lw=3.0,
                     alpha=0.85, zorder=7)
    for ax, (xl, yl) in zip(axs[0], lims):
        ax.set_xlim(xl); ax.set_ylim(yl)

    # faint outline of the catalog box, so the padding is visible as padding
    lmax, lmin, bmin, bmax = cd_mean['coords'][:4]
    kw = dict(color='white', lw=0.8, ls=':', alpha=0.9, zorder=5)
    for x in (lmax, lmin):                 # l-b and l-d share the l axis
        axs[0, 0].axvline(x, **kw); axs[0, 1].axvline(x, **kw)
    for y in (bmin, bmax):                 # l-b and d-b share the b axis
        axs[0, 0].axhline(y, **kw); axs[0, 2].axhline(y, **kw)
    # and the leaf's own distance edges, so the opened-out range reads as
    # padding the same way the l/b padding does
    for dd in (ds[lo_i], ds[hi_i]):        # l-d has d on y, b-d has d on x
        axs[0, 1].axhline(dd, **kw); axs[0, 2].axvline(dd, **kw)

    # The distance-window markers are a vlines LineCollection on the mps panel,
    # drawn from the cube that was PLOTTED -- which is now the opened-out range,
    # not the leaf. Drop them and mark the leaf itself, so this panel agrees with
    # the white dotted leaf edges on the maps.
    for coll in list(axs[1, 0].collections):
        coll.remove()
    for dd in (ds[lo_d], ds[hi_d]):        # adopted window
        axs[1, 0].axvline(dd, color='red', ls='--', lw=1.2, zorder=2)
    for dd in (ds[lo_i], ds[hi_i]):        # the dendrogram leaf, for reference
        axs[1, 0].axvline(dd, color='0.35', ls=':', lw=1.0, zorder=2)
    axs[1, 0].axvline(d_mean, color='grey', ls='--', lw=1.2, zorder=1)

    yy = np.concatenate(yvals)
    yy = yy[np.isfinite(yy) & (yy > 0)]          # log axis
    if yy.size:
        axs[1, 1].set_ylim(yy.min() / 1.6, yy.max() * 1.6)   # sharey carries it

    F = np.array(fits_s)
    F[:, 1] = np.abs(F[:, 1])          # plummer uses (r/r0)**2: sign degenerate

    # Reject collapsed fits. rho0 and r0 are strongly anti-correlated in the
    # Plummer form, so a fit that lets r0 run to zero compensates with a runaway
    # central density -- on Hercules one sample returned r0 = 0.00 pc and
    # rho0 = 1224 cm^-3 against ~30 for the other five, which alone drove the
    # quoted spread to +-487. The floor is a fraction of the transverse voxel
    # size at this distance: no core radius below that is resolved anyway.
    L_min = 4.0e-3 * d_mean
    good = F[:, 1] > 0.2 * L_min
    n_bad = int((~good).sum())
    if n_bad:
        print(f'  !! dropped {n_bad} collapsed Plummer fit(s) '
              f'(r0 < {0.2 * L_min:.2f} pc); {int(good.sum())} left')
    Fg = F[good] if good.sum() > 1 else F
    # d_parl panel: rho_radial_profiles_plot labels BOTH reference lines, which
    # makes a two-entry legend where only the lower one is the cut that matters.
    # Keep both lines drawn, re-legend on the n_med/10 artist alone.
    # d_parl panel: drop the n_med/10 rule itself and state the number instead.
    # The dashed n_med line stays -- it sits among the data and is worth seeing;
    # n_med/10 is a threshold far below the points, so the value is the useful
    # part, not the rule.
    n10 = (float(n_iso) if n_iso is not None
           else float(res_mean.get('med_parl', np.nan)) / 10.0)
    for ln in list(axs[1, 1].lines):
        if '/10' in (ln.get_label() or ''):
            ln.remove()
    if axs[1, 1].get_legend() is not None:
        axs[1, 1].get_legend().remove()
    lbl = (rf'$n = {n10:.3g}$ cm$^{{-3}}$ cut' if n_iso is not None
           else rf'$n_\mathrm{{med}}/10 = {n10:.3g}$ cm$^{{-3}}$')
    axs[1, 1].text(0.02, 0.97, lbl,
                   transform=axs[1, 1].transAxes, ha='left', va='top', fontsize=16)

    # d_perp panel: drop the black mean-map fit box so the sample values, which
    # carry the uncertainties, are the only numbers on the panel.
    for t in list(axs[1, 2].texts):
        t.remove()

    txt = '\n'.join([
        rf'$\rho_0 = {Fg[:, 0].mean():.1f} \pm {Fg[:, 0].std(ddof=1):.1f}$ cm$^{{-3}}$',
        rf'$r_0 = {Fg[:, 1].mean():.2f} \pm {Fg[:, 1].std(ddof=1):.2f}$ pc',
        rf'$\alpha = {Fg[:, 2].mean():.2f} \pm {Fg[:, 2].std(ddof=1):.2f}$'
        + ('' if not n_bad else f'   [{len(Fg)}/6 fits]')])
    axs[1, 2].text(0.97, 0.97, txt, transform=axs[1, 2].transAxes,
                   ha='right', va='top', fontsize=17, color='black')

    stats = {}
    for k in ('l', 'b', 'd', 'pa_lb', 'inc_ld', 'inc_bd'):
        v = np.array([x[k] for x in descs])
        stats[k], stats[k + '_sd'] = v.mean(), v.std(ddof=1)
    ang = (rf'PA$_{{lb}}$ = {stats["pa_lb"]:.1f}$\pm${stats["pa_lb_sd"]:.1f}$^\circ$   '
           rf'inc$_{{ld}}$ = {stats["inc_ld"]:.1f}$\pm${stats["inc_ld_sd"]:.1f}$^\circ$   '
           rf'inc$_{{bd}}$ = {stats["inc_bd"]:.1f}$\pm${stats["inc_bd_sd"]:.1f}$^\circ$')
    axs[0, 1].set_title(ang, fontsize=10)

    print(f'  rho0={Fg[:, 0].mean():7.2f}+-{Fg[:, 0].std(ddof=1):5.2f}  '
          f'r0={Fg[:, 1].mean():5.2f}+-{Fg[:, 1].std(ddof=1):4.2f}  '
          f'alpha={Fg[:, 2].mean():5.3f}+-{Fg[:, 2].std(ddof=1):5.3f}')
    print(f'  PA_lb={stats["pa_lb"]:7.2f}+-{stats["pa_lb_sd"]:5.2f}  '
          f'inc_ld={stats["inc_ld"]:7.2f}+-{stats["inc_ld_sd"]:5.2f}  '
          f'inc_bd={stats["inc_bd"]:7.2f}+-{stats["inc_bd_sd"]:5.2f}')

    if save:
        fig.savefig(f'{FIG_LOC}{cloud}{suffix}.png', bbox_inches='tight', dpi=150)
        print(f'  saved {FIG_LOC}{cloud}{suffix}.png')
    plt.close(fig)

    stats['cloud'] = cloud
    stats['M'] = np.nansum(prof * dmask)
    stats['win_lo'], stats['win_hi'] = dlim   # (ds[lo_d], ds[hi_d]), the adopted window
    for n, key in enumerate(('rho0', 'r0', 'alpha')):
        stats[key], stats[key + '_sd'] = Fg[:, n].mean(), Fg[:, n].std(ddof=1)
    stats['n_fit'] = len(Fg)
    return fig, F, stats


def axis_angle(dy, dx):
    """Undirected line angle in degrees, wrapped to [-90, 90).

    A spine has no preferred direction -- vec and -vec are the same line -- so
    angles must be reduced mod 180 before any mean or std, exactly as |r0| had
    to be taken before quoting a spread on the Plummer core radius.
    """
    a = np.degrees(np.arctan2(dy, dx))
    return (a + 90.0) % 180.0 - 90.0


def spine_descriptors(pt_lbd, vec_lbd):
    """A spine as a point plus the three projection angles.

    pa_lb  position angle in the l-b plane; both axes are degrees, so this is a
           plain position angle.
    inc_ld angle out of the plane of the sky in the l-d plane. The raw slope
           dd/dl is pc per degree, whose "angle" would depend on the plot
           aspect, so the angular offset is converted to transverse pc at the
           cloud's distance (with cos b) before taking the arctan. That makes it
           a physical inclination: 0 = lying in the plane of the sky, +-90 =
           pointing straight down the line of sight.
    inc_bd the same in the b-d plane.
    """
    l0, b0, d0 = pt_lbd
    dl, db, dd = vec_lbd
    kl = np.deg2rad(1.0) * d0 * np.cos(np.deg2rad(b0))   # pc per deg of l
    kb = np.deg2rad(1.0) * d0                            # pc per deg of b
    return dict(l=l0, b=b0, d=d0,
                pa_lb=axis_angle(db, dl),
                inc_ld=axis_angle(dd, dl * kl),
                inc_bd=axis_angle(dd, db * kb))


def spine_candidates(i, source, table1, dmask, n_seeds=40, n_jobs=-1):
    """Every Monte-Carlo spine fit for one cube, not just the best one.

    pick_best_line keeps only the lowest-residual line out of n_iter random
    starts and throws the rest away. On these clouds the L1 objective is flat --
    L1d agrees to four decimals across fits whose orientations differ by ~5 deg
    -- so the discarded solutions are not failures, they are the width of the
    fit. Keeping them is what makes an honest error bar possible.

    Returns (rows, dats_arr): rows of [L1d, x1,y1,z1, x2,y2,z2], and the 99th
    percentile pixels the fit saw.
    """
    dats = get_99th_percs(i, source, table1, 'full', which='both', dmask=dmask)
    dats_arr = dats[0].to_numpy()
    if n_jobs == 1:
        rows = [_fit_one_line(dats_arr, sd) for sd in range(n_seeds)]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one_line)(dats_arr, sd) for sd in range(n_seeds))
    return np.array(rows), dats_arr


def lbd_tangent(p1, p2, anchor, eps=0.01):
    """Local (dl, db, dd) of the line at `anchor`, all in xyz-consistent units.

    A line that is straight in xyz is CURVED in (l, b, d), so the chord
    p2_lbd - p1_lbd depends on where the two endpoints happen to sit along it --
    on Taurus that alone moved inc_ld by ~4 deg between fits that had bit-
    identical residuals. Differencing across a short step centred on the anchor
    measures the tangent instead, which is a property of the line and the anchor
    only.
    """
    u = (p2 - p1) / np.linalg.norm(p2 - p1)
    a = np.array(xyz_2_sph(anchor - eps * u))
    b = np.array(xyz_2_sph(anchor + eps * u))
    v = b - a
    v[0] = (v[0] + 180.0) % 360.0 - 180.0        # keep dl on the short way round
    return v


def anchor_point(p1, p2, dats_arr):
    """A stable point on the line: the data centroid projected onto it.

    get_spine hands back the midpoint of its two fitted endpoints, but the fit
    constrains the LINE and not where the endpoints sit on it, so that midpoint
    slides -- on Taurus it moved 4 deg in l between two runs on identical data.
    Projecting the centroid of the fitted pixels onto the line pins it down.
    """
    u = p2 - p1
    c = dats_arr[:, 0:3].mean(axis=0)
    t = np.dot(c - p1, u) / np.dot(u, u)
    return p1 + t * u


# ============================================================================
# Density-masked mass per posterior sample (M_5 / M_10 columns)
#   (from add_masked_mass.py)
# ============================================================================
# Append per-posterior-sample, density-masked mass (M_5/M_5_sd, M_10/M_10_sd)
# to the spine_angles_and_fits_samples_n5.csv / _n10.csv written by
# recompute_cloud_fits.py -- the columns plot_rho0_correlations.py's mass panel
# reads.
#
# For each cloud, masks each of the 6 independent posterior-sample cubes to
# n_H >= threshold over the cloud's own adopted distance window (win_lo/win_hi
# from the CSV), sums to a mass, then takes the mean/sd across the 6 samples.
# Reproduces the 2026-09-03 session's mass_samples.py (positional sample
# indexing into build_sample_cubes' own already-independent-only output).
#
# Run through dust3d_pipeline.py --steps fits.

def masked_mass_per_cloud(t1, cloud, win_lo, win_hi, n_thresh):
    scubes, indep, _ = build_sample_cubes(t1, cloud)
    # the stored cubes carry PAD_STORED deg of padding; the mean-map M column is the
    # catalog box, so crop to it
    st = load_stored(t1, cloud)
    scubes = _unpadded(scubes, t1.iloc[list(t1['Name']).index(cloud)], st)
    lo_idx = int(round(win_lo - D0)); hi_idx = int(round(win_hi - D0))
    dmask = np.zeros(ND, dtype=bool); dmask[lo_idx:hi_idx + 1] = True
    Ms = []
    for n in range(len(indep)):          # positional: scubes already holds
        cube_k = scubes[n]               # only the independent draws
        masked = np.where(cube_k >= n_thresh, cube_k, 0.0)
        prof_k = mass_profile(masked)
        Ms.append(np.nansum(prof_k * dmask))
    Ms = np.array(Ms)
    return Ms.mean(), Ms.std(ddof=1)


# ============================================================================
# 40-seed spines walked to n_iso, cylinder profiles, the cylinder_three_row figure
#   (from plot_cylinder_slice.py)
# ============================================================================
# Two-row view of one cloud's fitted cylinder.
#
# Top row: the three zeroth-moment (column density) maps -- l-b, l-d, b-d --
# the same panels plot_moment_maps/plot_cloud_summary draw for the
# projmaps_profiles figures, with the fitted spine overlaid.
#
# Bottom row: the three principal-axis profiles of the fitted cylinder --
# density vs d_perp (radial), density vs d_parallel (axial, same quantities as
# the projmaps_profiles figures), and density vs d_phi (azimuthal: voxels
# binned by their angle around the spine, averaging over both r and z).
#
# radii_on_box (dust_3d_functions.py) only carries a signed d_parallel and an
# UNSIGNED d_perp per voxel -- the cylinder model (mif_cylinder) is azimuthally
# symmetric, rho(r, z), so nothing downstream needs the angle. To get d_phi,
# radii_and_phi_on_box() below redoes that computation with the perpendicular
# vector kept instead of collapsed to its norm: reject each voxel's position
# onto the plane normal to the spine, then take its angle against two fixed
# orthonormal axes spanning that plane (arbitrary zero-point -- there is no
# preferred phi=0 direction for a cylinder).
#
# Reuses the adopted distance window (win_lo/win_hi) already recalculated in
# spine_angles_and_fits_samples.csv, so this does not redo the leaf/window
# selection -- only the profile geometry.
#
# Spine and box (changed from the first version of this figure):
#   * the spine is the best of `--n-seeds` (40) Monte-Carlo fits, same as
#     sample_summary's mean spine -- the first version used D.ha's 5;
#   * the fit itself runs on the catalog box, unpadded. Once it is fixed, the
#     spine is walked out from its anchor until the density along it drops below
#     `--n-iso` (N_ISO, 10 cm^-3; 7 through 2026-09-30), and the box used for the maps and all three profiles
#     is grown (up to --lb-search-pad, at most the PAD_STORED deg the stored cubes carry)
#     until it holds that whole stretch. The
#     profile voxels are therefore the same ones the maps show.
#
# Run through dust3d_pipeline.py --steps cylinder.

def radii_and_phi_on_box(i, source_array, cloud_catalog, drange, spine_xyz,
                         dmask=None, pad=0.0):
    """Like dust_3d_functions.radii_on_box, but also returns each voxel's
    azimuthal angle around the spine (degrees, arbitrary zero-point)."""
    cube, coords = get_cloud_info(i, source_array, cloud_catalog, drange,
                                    dmask=dmask, pad=pad)
    pix = get_99th_percs(i, source_array, cloud_catalog, drange, which='else',
                           dmask=dmask, pad=pad)
    p1_xyz, p2_xyz = spine_xyz
    _, all_dat = L1ds(p1_xyz, p2_xyz, pix)
    all_df = pd.DataFrame(all_dat, columns=['x', 'y', 'z', 'd', 'b', 'l', 'n',
                                            'dperp', 'dparl'])

    u = (p2_xyz - p1_xyz)
    u = u / np.linalg.norm(u)
    # any fixed direction not parallel to u; its component along u removed
    # and the result normalized gives one axis of the perpendicular plane
    ref = np.array([0., 0., 1.]) if abs(u[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = ref - np.dot(ref, u) * u
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(u, e1)

    xyz = all_df[['x', 'y', 'z']].to_numpy()
    v = xyz - p1_xyz[np.newaxis, :]
    v_along = np.outer(v @ u, u)
    v_perp = v - v_along   # magnitude of this is dperp, direction is new here
    all_df['phi'] = np.degrees(np.arctan2(v_perp @ e2, v_perp @ e1))

    lbd = all_df[['d', 'b', 'l']].to_numpy()
    r_perp = np.flip(coords_to_cube(lbd, all_df['dperp']).transpose(), axis=2)
    r_parl = np.flip(coords_to_cube(lbd, all_df['dparl']).transpose(), axis=2)
    r_phi = np.flip(coords_to_cube(lbd, all_df['phi']).transpose(), axis=2)
    return cube, r_perp, r_parl, r_phi, coords


def density_along_spine(p1, p2, anchor, box, xo, yo, t_max=400.0, step=0.25):
    """(t, n) of the interpolated density along the line through `anchor`,
    for t where the point lies inside the stored box (same index mapping as
    spine_extent)."""
    u = (p2 - p1) / np.linalg.norm(p2 - p1)
    nz, ny, nx = box.shape
    ts = np.arange(-t_max, t_max + step, step)
    lbd = np.array([xyz_2_sph(anchor + t * u) for t in ts])
    xi = (L0 - lbd[:, 0]) / CD - xo
    yi = (lbd[:, 1] - B0) / CD - yo
    zi = lbd[:, 2] - D0
    ok = (xi >= 0) & (xi <= nx - 1) & (yi >= 0) & (yi <= ny - 1) & (zi >= 0) & (zi <= nz - 1)
    n = np.full(ts.shape, np.nan)
    n[ok] = map_coordinates(box, [zi[ok], yi[ok], xi[ok]], order=1)
    return ts, n, u


def walk_spine(p1, p2, anchor, box, xo, yo, dlim, n_iso):
    """Walk a fixed line out to where the density along it drops below n_iso.

    Starts at `anchor`; if the anchor itself is below n_iso (it is a centroid of
    the top-1% pixels projected onto the line, so on an elongated or clumpy
    cloud it can sit in thin gas), restarts from the densest point of the line
    inside the distance window. Returns (start, t_neg, t_pos, note), note being
    None or a string saying what was unusual.
    """
    ts, nline, u = density_along_spine(p1, p2, anchor, box, xo, yo)
    n_anchor = nline[np.nanargmin(np.abs(ts))]
    note = None
    if not n_anchor >= n_iso:
        dline = np.array([xyz_2_sph(anchor + t * u)[2] for t in ts])
        nline_w = np.where((dline >= dlim[0]) & (dline <= dlim[1]), nline, np.nan)
        if np.all(np.isnan(nline_w)) or np.nanmax(nline_w) < n_iso:
            return anchor, 0.0, 0.0, f'no point on the spine reaches n = {n_iso:g}'
        anchor = anchor + ts[np.nanargmax(nline_w)] * u
        note = (f'anchor n = {n_anchor:.2f} < {n_iso:g}: restarted from densest '
                f'point (n = {np.nanmax(nline_w):.1f})')
    t_neg, t_pos = spine_extent(p1, p2, anchor, box, xo, yo, n_iso, dlim)
    return anchor, t_neg, t_pos, note


def fit_spine(i, proxy, table1, dmask, n_seeds, n_iso, dlim, box, xo, yo):
    """Best-of-n_seeds spine on the unpadded catalog box of `proxy`, walked to
    n_iso in `box`. Everything returned is plain numpy/float, so it pickles."""
    rows, dats_arr = spine_candidates(i, proxy, table1, dmask, n_seeds, n_jobs=-1)
    best = rows[rows[:, 0].argmin()]
    p1, p2 = best[1:4], best[4:7]
    anchor = anchor_point(p1, p2, dats_arr)
    start, t_neg, t_pos, note = walk_spine(p1, p2, anchor, box, xo, yo, dlim, n_iso)
    curve = spine_curve(p1, p2, start, (t_neg, t_pos))
    return {'p1': p1, 'p2': p2, 'anchor': anchor, 'start': start, 'L1d': float(best[0]),
            't_neg': float(t_neg), 't_pos': float(t_pos), 'curve': curve,
            'd_lo': float(np.nanmin(curve[:, 2])), 'd_hi': float(np.nanmax(curve[:, 2])),
            'note': note}


def get_spine_fits(cloud, table1, win_lo, win_hi, n_iso=N_ISO, n_seeds=40,
                   lb_search_pad=PAD_STORED, cache_dir=None, refit=False):
    """Mean-map spine plus one spine per independent posterior sample, each fit
    and walked on its OWN cube. Cached per cloud in `cache_dir` (the sample
    cubes cost ~40 s per cloud to build)."""
    path = None if cache_dir is None else os.path.join(
        cache_dir, f'{cloud}_spine_fits_n{n_iso:g}_s{n_seeds}.pkl')
    if path and os.path.exists(path) and not refit:
        with open(path, 'rb') as f:
            return pickle.load(f)

    i = list(table1['Name']).index(cloud)
    dmask = (ds >= win_lo) & (ds <= win_hi)
    dlim = (float(ds[dmask][0]), float(ds[dmask][-1]))

    wbox, (xo, yo) = mean_padded_box(table1, cloud, pad=lb_search_pad)
    mean = fit_spine(i, CloudBoxProxy(wbox, xo, yo), table1, dmask, n_seeds, n_iso,
                     dlim, wbox, xo, yo)

    scubes, indep, (sx, sy) = build_sample_cubes(table1, cloud, pad=lb_search_pad)
    samples = [fit_spine(i, CloudBoxProxy(scubes[k], sx, sy), table1, dmask, n_seeds,
                         n_iso, dlim, scubes[k], sx, sy) for k in range(len(indep))]
    out = {'cloud': cloud, 'n_iso': n_iso, 'n_seeds': n_seeds, 'window': (win_lo, win_hi),
           'mean': mean, 'samples': samples, 'indep': [int(k) for k in indep]}
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(out, f)
    return out


def cylinder_profile(i, proxy, table1, dmask, spine, lb_pad=1.0, lb_search_pad=PAD_STORED,
                     spine_slab=True, perp_nmin=0.0, perp_fit='lin_dip', quiet=False):
    """One spine's cylinder, on whichever cube `proxy` wraps (mean map or one
    posterior sample), in this order:
      1. the spine and its caps are fixed already (`spine`, a fit_spine record:
         walked to n_iso on that same cube);
      2. grow the box until it holds both caps (up to lb_search_pad);
      3. d_perp profile from the voxels between the caps;
      4. r_dip on that profile, Plummer fit out to r_dip (perp_fit='lin_dip');
      5. M_cyl, the mass between the caps and within r_dip.
    Returns (cd, res, cube, r_perp, r_parl, r_phi, pad)."""
    p1, p2 = spine['p1'], spine['p2']
    u = (p2 - p1) / np.linalg.norm(p2 - p1)
    row = table1.iloc[i]
    need = 0.0
    for sgn, tt in ((-1.0, spine['t_neg']), (1.0, spine['t_pos'])):
        l_, b_, _ = xyz_2_sph(spine['start'] + sgn * tt * u)
        need = max(need, row['l_min(deg)'] - l_, l_ - row['l_max(deg)'],
                   row['b_min(deg)'] - b_, b_ - row['b_max(deg)'])
    pad = float(np.clip(need + 0.15, lb_pad, lb_search_pad))
    if need + 0.15 > lb_search_pad and not quiet:
        print(f'  !! spine reach CAPPED at {lb_search_pad:g} deg -- did not close')

    # cube + radii on the grown box, for the FIXED spine
    cube, r_perp, r_parl, r_phi, coords = radii_and_phi_on_box(
        i, proxy, table1, 'full', (p1, p2), dmask=dmask, pad=pad)
    cd = {'dust_cube': cube, 'r_perp': r_perp, 'r_parl': r_parl,
          'coords': coords, 'spine_xyz': [p1, p2]}
    # the walked ends in the raw d_parl coordinate, (x - p1).u: the walk starts
    # at spine['start'] and runs t_neg back along u and t_pos forward
    s0 = float(np.dot(spine['start'] - p1, u))
    slab = (s0 - spine['t_neg'], s0 + spine['t_pos']) if spine_slab else None
    res = rho_radial_profiles_compute(i, proxy, table1, 'full', cloud_dict=cd,
                                        dmask=dmask, perp_pad=pad, perp_nmin=perp_nmin,
                                        perp_slab=slab, perp_fit=perp_fit)
    if spine_slab and res['slab_thin'] and not quiet:
        print('  !! spine slab held <2% of the voxels -- profile fell back to the full box')
    return cd, res, cube, r_perp, r_parl, r_phi, pad


def cylinder_samples(cloud, table1, win_lo, win_hi, fits, lb_pad=1.0,
                     lb_search_pad=PAD_STORED, spine_slab=True, perp_nmin=0.0,
                     perp_fit='lin_dip'):
    """cylinder_profile for each independent posterior sample, on that sample's
    own cube with its own spine and caps (fits['samples'], from get_spine_fits).
    A sample whose walk never reached n_iso (zero-length spine) is skipped.
    Returns one dict per sample: sample, rho0, r0, alpha, r_dip, dip_flag,
    M_cyl, pad, slab_thin."""
    i = list(table1['Name']).index(cloud)
    dmask = (ds >= win_lo) & (ds <= win_hi)
    scubes, indep, (sx, sy) = build_sample_cubes(table1, cloud, pad=lb_search_pad)
    if [int(k) for k in indep] != [int(k) for k in fits['indep']]:
        raise SystemExit(f'{cloud}: sample indices differ: cubes {list(indep)} '
                         f'vs spine fits {fits["indep"]}')
    out = []
    for k, sp in enumerate(fits['samples']):
        if sp['t_neg'] + sp['t_pos'] <= 0:
            print(f'  !! sample {fits["indep"][k]}: spine never reaches n_iso -- skipped')
            continue
        _, res, *_, pad = cylinder_profile(
            i, CloudBoxProxy(scubes[k], sx, sy), table1, dmask, sp, lb_pad,
            lb_search_pad, spine_slab, perp_nmin, perp_fit, quiet=True)
        rho0_, r0_, al_ = res['fit']
        out.append({'sample': int(fits['indep'][k]), 'rho0': rho0_, 'r0': abs(r0_),
                    'alpha': al_, 'r_dip': res['r_dip'], 'dip_flag': res['dip_flag'],
                    'M_cyl': res['M_cyl'], 'pad': pad, 'slab_thin': res['slab_thin']})
    return out


def build_cylinder_data(cloud, table1, win_lo, win_hi, n_iso=N_ISO, n_seeds=40,
                        lb_pad=1.0, lb_search_pad=PAD_STORED, fits=None, spine_slab=True,
                        perp_nmin=0.0, perp_fit='lin_dip'):
    """... spine_slab: keep only voxels whose position along the spine, d_parl,
    lies between the two ends of the walked spine when building the d_perp
    profile (and its fit). Off, every voxel in the box counts, however far
    beyond the spine's ends it sits along the axis.
    perp_nmin: density floor (cm^-3) for the d_perp profile. 0 (default since
    2026-10-14) keeps every voxel, so each bin is the median of ALL voxels in
    that shell inside the slab; n_iso gives the old n >= n_iso profile."""
    i = list(table1['Name']).index(cloud)
    dmask = (ds >= win_lo) & (ds <= win_hi)

    # one wide stored box serves the fit (pad 0), the walk and the drawn box
    wbox, (xo, yo) = mean_padded_box(table1, cloud, pad=lb_search_pad)
    wproxy = CloudBoxProxy(wbox, xo, yo)

    if fits is None:
        dlim = (float(ds[dmask][0]), float(ds[dmask][-1]))
        mean = fit_spine(i, wproxy, table1, dmask, n_seeds, n_iso, dlim, wbox, xo, yo)
    else:
        mean = fits['mean']
    if mean['note']:
        print('  ' + mean['note'])
    cd, res, cube, r_perp, r_parl, r_phi, pad = cylinder_profile(
        i, wproxy, table1, dmask, mean, lb_pad, lb_search_pad, spine_slab,
        perp_nmin, perp_fit)
    print(f'  spine walk: {mean["t_neg"]:.1f} / {mean["t_pos"]:.1f} pc each side, '
          f'box pad {pad:.2f} deg, L1d {mean["L1d"]:.4f}')
    sample_curves = [] if fits is None else [f['curve'] for f in fits['samples']]
    sample_ids = [] if fits is None else list(fits['indep'])
    return cd, res, cube, r_perp, r_parl, r_phi, mean['curve'], sample_curves, sample_ids


def sample_colors(n):
    """One colour per independent posterior sample, shared with
    plot_sample_spine_windows so a spine here has the colour of its spectrum
    and bar there (the mean is black in both)."""
    return plt.cm.tab10(np.arange(n))


def plot_mass_row(ax, mass, sample_ids, cols, zoom=True):
    """Mass per slice of the mean map and each independent sample over the
    unpadded catalog box, with the adopted window, the mean spine's distance
    extent and the catalog d_mean. zoom=False shows the whole 69-1250 pc axis."""
    prof, sprofs, (lo, hi), d_mean, (s_lo, s_hi) = mass
    ax.plot(ds, prof, color='k', lw=1.8, label='mean map', zorder=5)
    for k, (sp, c_) in enumerate(zip(sprofs, cols)):
        ax.plot(ds, sp, color=c_, lw=0.9, alpha=0.85, label=f'sample {sample_ids[k]}')
    ax.axvspan(lo, hi, color='tab:red', alpha=0.08, lw=0)
    for x in (lo, hi):
        ax.axvline(x, color='tab:red', ls='--', lw=1.2)
    ax.axvline(d_mean, color='0.5', ls=(0, (6, 3)), lw=1)
    ax.set_yscale('log')
    span = max(hi, s_hi) - min(lo, s_lo)
    if zoom:
        x0, x1 = max(ds[0], min(lo, s_lo) - 0.8 * span), min(ds[-1], max(hi, s_hi) + 0.8 * span)
    else:
        x0, x1 = ds[0], ds[-1]
    ax.set_xlim(x0, x1)
    sel = (ds >= x0) & (ds <= x1)
    top = np.nanmax(prof[sel])
    ax.set_ylim(top * 3 / 10**3, top * 3)      # a fixed 3 dex below the headroom
    # the mean spine's distance extent as a bar near the top
    ax.hlines(top * 1.8, s_lo, s_hi, color='k', lw=4)
    if zoom:
        ax.text(s_hi + 0.01 * (x1 - x0), top * 1.8, f'spine {s_lo:.0f}-{s_hi:.0f} pc',
                va='center', fontsize=9)
    ax.set_xlabel('heliocentric distance (pc)')
    ax.set_ylabel(r'mass per slice ($M_\odot\,\mathrm{pc}^{-1}$)')
    if not zoom:
        ax.set_title('mass per distance slice, catalog box: full range', fontsize=11)
        return
    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color='tab:red', ls='--', lw=1.2,
                       label=f'window {lo:.0f}-{hi:.0f} pc'),
                Line2D([], [], color='0.5', ls=(0, (6, 3)), lw=1,
                       label=rf'catalog $d_{{mean}}$ = {d_mean:.0f} pc')]
    ax.legend(handles=handles, fontsize=8, ncol=5, frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, -0.22))
    ax.set_title('zoom on the window', fontsize=11)


def save_axes_region(fig, axes, path, whole_rows=True, pad=0.1):
    """Save only the part of `fig` covered by `axes` (with their labels, titles
    and legends), e.g. one row or a few panels of a multi-panel figure, for the
    paper. whole_rows: also take in every other axes of the figure whose centre
    lies in the same vertical band (the colorbars of a row of maps)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    axes = list(axes)
    if whole_rows:
        y0 = min(a.get_position().y0 for a in axes)
        y1 = max(a.get_position().y1 for a in axes)
        axes += [a for a in fig.axes if a not in axes
                 and y0 <= 0.5 * (a.get_position().y0 + a.get_position().y1) <= y1]
    bb = Bbox.union([a.get_tightbbox(rend) for a in axes])
    fig.savefig(path, bbox_inches=bb.transformed(fig.dpi_scale_trans.inverted()).padded(pad))
    print('saved', path)


def plot_cylinder_slice(cloud, cd, res, cube, r_perp, r_parl, r_phi, spine, sample_spines,
                        sample_ids, out_png, n_iso=N_ISO, n_phi_bins=24, mass=None):
    """Two rows (maps / perp, parl, phi profiles), or with `mass` given three
    rows: mass per slice / maps / parl, perp, phi profiles. `mass` is
    (mean profile, sample profiles, (win_lo, win_hi), d_mean, (spine d_lo, d_hi))."""
    rho0, r0, alpha = res['fit']
    r0 = abs(r0)       # Plummer is even in r0; the fit can land on either sign
    dips = {}

    if mass is None:
        fig, axs = plt.subplots(2, 3, figsize=(17, 10.5), dpi=150,
                                constrained_layout=True)
        maps, (ax_perp, ax_parl, ax_phi) = axs[0], axs[1]
        tags = {'perp': '(a)', 'parl': '(b)', 'phi': '(c)'}
    else:
        fig = plt.figure(figsize=(17, 15), dpi=150, constrained_layout=True)
        gs = GridSpec(3, 3, figure=fig, height_ratios=[0.7, 1, 1])
        gtop = gs[0, :].subgridspec(1, 2, width_ratios=[1, 1])
        ax_full, ax_mass = fig.add_subplot(gtop[0]), fig.add_subplot(gtop[1])
        maps = np.array([fig.add_subplot(gs[1, k]) for k in range(3)])
        ax_parl, ax_perp, ax_phi = (fig.add_subplot(gs[2, k]) for k in range(3))
        tags = {'parl': '(a)', 'perp': '(b)', 'phi': '(c)'}
        plot_mass_row(ax_full, mass, sample_ids, sample_colors(len(sample_spines)), zoom=False)
        plot_mass_row(ax_mass, mass, sample_ids, sample_colors(len(sample_spines)))
    axs = np.array([maps, [ax_perp, ax_parl, ax_phi]])

    # --- top row: l-b / l-d / b-d moment maps over the grown box, with the
    # n_iso silhouette and the spine walked out to it ---------------------
    plot_moment_maps(cd['dust_cube'], cd['coords'], cloud, axs=axs[0],
                       iso=n_iso, iso_color='limegreen')
    lims = [(a.get_xlim(), a.get_ylim()) for a in axs[0]]
    cols = sample_colors(len(sample_spines))
    for crv, col in zip(sample_spines, cols):     # one refit per posterior sample
        draw_spine_curve(axs[0], lims, crv, color=col, lw=1.6, alpha=0.95, zorder=6)
    draw_spine_curve(axs[0], lims, spine, color='k', lw=3.0, alpha=0.9, zorder=5,
                     path_effects=[pe.withStroke(linewidth=4.5, foreground='white')])
    handles = [Line2D([], [], color='k', lw=3, label='mean')] + [
        Line2D([], [], color=c_, lw=1.6, label=f'sample {k}')
        for c_, k in zip(cols, sample_ids)]
    axs[0, 0].legend(handles=handles, fontsize=7, ncol=2, frameon=True, framealpha=0.8,
                     loc='upper right')

    # --- (a) d_perp profile, same as the projmaps panel ---------------------
    # linear bins from 0 (two 0.125 deg pixel scales wide); the fit was made on the
    # log bins of the same voxels. One profile: the median of every voxel in each
    # shell whose position along the axis lies between the spine's two ends (or
    # in the whole box with the slab off); res['perp_nmin'] is its density floor.
    df = res['df_lin']
    nfl = res.get('perp_nmin', np.nan)
    sel = ('between the spine ends' if res.get('slab') is not None else 'whole box') + (
        '' if not (np.isfinite(nfl) and nfl > 0) else rf', $n \geq$ {nfl:g} cm$^{{-3}}$')
    axs[1, 0].scatter(df['d'], df['rho'], color='black', marker='.', s=60,
                      label=f'median n, voxels {sel}')
    rr = np.geomspace(0.5 * df['d'].min(), df['d'].max(), 300)   # log x axis below
    fit_lab = ('Plummer fit (these bins, r $\\leq$ dip)' if res.get('perp_fit') == 'lin_dip'
               else 'Plummer fit (log-binned profile)')
    axs[1, 0].plot(rr, plummer(rr, rho0, r0, alpha), color='tab:red', label=fit_lab)
    k, flag = find_dip(df['d'], df['rho'])
    rd, nd = float(df['d'].iloc[k]), float(df['rho'].iloc[k])
    dips['profile'] = (rd, nd, flag)
    axs[1, 0].axvline(abs(r0), color='tab:red', ls=':', lw=1.4,
                      label=rf'$r_0$ = {abs(r0):.2g} pc')
    axs[1, 0].axvline(rd, color='black', ls='--', lw=1.2,
                      label=f'dip: {rd:.1f} pc' + ('' if flag == 'dip' else ' [no dip]'))
    axs[1, 0].legend(fontsize=8, frameon=False, loc='upper right',
                     bbox_to_anchor=(1.0, 0.82))

    # --- the d_perp bins on the maps. A bin is the cylindrical shell between two
    # edges; seen in projection, the shell r < R is the contour R of the minimum
    # d_perp along that projection axis (same construction as plot_moment_maps'
    # r_perp option). Every k-th edge is drawn so ~8 bands show; the masked dip
    # is drawn thicker. With the spine slab on, only voxels between the spine's
    # two ends enter, so each shell is capped there; off, it runs the full box.
    edges = np.asarray(res['lin_bins'])
    slab = res.get('slab')
    r_use = r_perp if slab is None else np.where(
        (r_parl >= slab[0]) & (r_parl <= slab[1]), r_perp, np.nan)
    k_every = max(1, int(np.ceil((len(edges) - 1) / 8)))
    show = edges[k_every::k_every]
    w_bin = edges[1] - edges[0]
    r_dip = dips['profile'][0]
    coords = cd['coords']
    for ax_, axis_, ext_, tr_ in [(axs[0, 0], 0, coords[:4], False),
                                  (axs[0, 1], 1, coords[[0, 1, 4, 5]], False),
                                  (axs[0, 2], 2, coords[[4, 5, 2, 3]], True)]:
        with np.errstate(all='ignore'):
            rmin = np.nanmin(r_use, axis=axis_)
        rmin = np.where(np.isfinite(rmin), rmin, np.inf)     # outside the slab
        rmin = rmin.T if tr_ else rmin
        lim = (ax_.get_xlim(), ax_.get_ylim())
        ax_.contour(rmin, extent=ext_, levels=show, colors='cyan', linewidths=0.7,
                    alpha=0.9, zorder=8)
        ax_.contour(rmin, extent=ext_, levels=[r_dip], colors='cyan', linewidths=2.2,
                    linestyles='--', zorder=9)
        ax_.set_xlim(lim[0]); ax_.set_ylim(lim[1])
    axs[0, 1].legend(handles=[
        Line2D([], [], color='cyan', lw=0.7,
               label=rf'$d_\perp$ bin edges, every {k_every} '
                     rf'({k_every * w_bin:.2g} pc; bins {w_bin:.2g} pc)'),
        Line2D([], [], color='cyan', lw=2.2, ls='--',
               label=f'dip, {r_dip:.1f} pc')],
        fontsize=7.5, loc='upper right', framealpha=0.8)
    axs[1, 0].text(0.97, 0.95,
               '\n'.join([rf'$\rho_0 = {rho0:.3g}$ cm$^{{-3}}$',
                          rf'$r_0 = {r0:.3g}$ pc', rf'$\alpha = {alpha:.3g}$']),
               transform=axs[1, 0].transAxes, ha='right', va='top', fontsize=11)
    # zoom on the cloud: 0 to the dip plus a margin (half the dip radius,
    # at least 3 pc), with y scaled to the points in view rather than to the fit,
    # which spikes at r = 0 when it collapses
    x_hi = rd + max(0.5 * rd, 3.0)
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_xlim(0.8 * df['d'].min(), x_hi)
    in_view = df['rho'][df['d'] <= x_hi].to_numpy()
    pos = in_view[in_view > 0]
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_ylim(0.7 * pos.min(), 1.5 * pos.max())
    axs[1, 0].set_xlabel(r'$d_\perp$ (pc)')
    axs[1, 0].set_ylabel(r'$n$ (cm$^{-3}$)')
    axs[1, 0].set_title(f"{tags['perp']} radial profile"
                        + ('' if res.get('slab') is None else ', between the spine ends'),
                        fontsize=11)

    # --- (b) d_parallel profile, same as the projmaps panel -----------------
    df_parl = res['df_parl']
    med_parl = res['med_parl']
    axs[1, 1].scatter(df_parl['d'], df_parl['rho'], color='black', marker='.', s=60)
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_xlabel(r'$d_\parallel$ (pc)')
    axs[1, 1].set_ylabel(r'$n$ (cm$^{-3}$)')
    axs[1, 1].set_title(f"{tags['parl']} axial profile", fontsize=11)

    # --- (c) d_phi profile: density vs azimuth around the spine, averaged
    # over both r and z (all voxels in the same box the other two profiles
    # use, regardless of their d_perp or d_parallel) -----------------------
    # capped like d_perp: only voxels between the spine's two ends along the axis
    phi_bins = np.linspace(-180, 180, n_phi_bins + 1)
    slab_ = res.get('slab')
    r_phi_use = r_phi if slab_ is None else np.where(
        (r_parl >= slab_[0]) & (r_parl <= slab_[1]), r_phi, np.nan)
    # phi = 0 is put along the direction in which the density falls off fastest
    # away from the spine: per 15 deg sector, ln(median n at r < r_half /
    # median n at r_half <= r < 3 r_half), smoothed over the sector and its two
    # neighbours; r_half is where the radial profile first drops to half its
    # central value. Same voxels as the radial profile (inside the slab).
    prof = res['df_lin']
    n_c = np.nanmax(prof['rho'].to_numpy()[:3])
    below = np.flatnonzero(prof['rho'].to_numpy() < 0.5 * n_c)
    r_half = float(prof['d'].iloc[below[0]]) if below.size else float(prof['d'].iloc[-1])
    ok = np.isfinite(r_phi_use) & np.isfinite(r_perp) & np.isfinite(cube)
    ph, rr_, nn = r_phi_use[ok], r_perp[ok], cube[ok]
    sec = np.clip(np.digitize(ph, phi_bins) - 1, 0, n_phi_bins - 1)
    steep = np.full(n_phi_bins, np.nan)
    for j in range(n_phi_bins):
        m = sec == j
        inn, out_ = nn[m & (rr_ < r_half)], nn[m & (rr_ >= r_half) & (rr_ < 3 * r_half)]
        if inn.size and out_.size and np.median(out_) > 0:
            steep[j] = np.log(np.median(inn) / np.median(out_))
    sm = np.array([np.nanmean(steep[[(j - 1) % n_phi_bins, j, (j + 1) % n_phi_bins]])
                   for j in range(n_phi_bins)])
    j0 = int(np.nanargmax(sm))
    phi0 = 0.5 * (phi_bins[j0] + phi_bins[j0 + 1])
    r_phi_use = (r_phi_use - phi0 + 180.0) % 360.0 - 180.0
    df_phi = bin_by_r(r_phi_use, cube, phi_bins)

    axs[1, 2].scatter(df_phi['d'], df_phi['rho'], color='black', marker='.', s=60)
    axs[1, 2].plot(df_phi['d'], df_phi['rho'], color='0.6', lw=0.8, zorder=0)
    axs[1, 2].set_yscale('log')
    axs[1, 2].set_xlim(-180, 180)
    axs[1, 2].set_xlabel(r'$\phi$ (deg; 0 = steepest radial fall-off)')
    axs[1, 2].text(0.02, 0.97, rf'$r_{{1/2}}$ = {r_half:.2g} pc; steepness ratio'
                   rf' {np.exp(np.nanmax(sm)):.2g}$\times$ (max) vs'
                   rf' {np.exp(np.nanmin(sm)):.2g}$\times$ (min)',
                   transform=axs[1, 2].transAxes, va='top', fontsize=8)
    axs[1, 2].set_ylabel(r'$n$ (cm$^{-3}$)')
    axs[1, 2].set_title(f"{tags['phi']} azimuthal profile"
                        + ('' if slab_ is None else ', between the spine ends'), fontsize=11)

    fig.suptitle(f'{cloud} -- fitted cylinder (40-seed spine, n_iso = {n_iso:g} cm$^{{-3}}$)',
                 y=1.02, fontsize=13)
    fig.savefig(out_png, bbox_inches='tight')
    print('saved', out_png)
    if mass is not None:
        # each row on its own too (<stem>_mass / _maps / _profiles), for the paper
        stem = out_png[:-4]
        for row, ax_row in (('mass', [ax_full, ax_mass]), ('maps', list(maps)),
                            ('profiles', [ax_parl, ax_perp, ax_phi])):
            save_axes_region(fig, ax_row, f'{stem}_{row}.png')
    return dips


# ============================================================================
# Mass spectra with each posterior sample spine's distance range
#   (from plot_sample_spine_windows.py)
# ============================================================================
# Mass-per-slice spectra with the distance range each posterior sample's own
# best-fit spine covers.
#
# For each cloud:
#   top    : mass per distance slice (Msun/pc) of the mean map (black) and of each
#            of the 6 independent posterior samples (colours), with the adopted
#            distance window the spines were fit inside (red dashed).
#   bottom : one horizontal bar per sample (+ the mean) from the nearest to the
#            farthest distance on that sample's spine, i.e. where its fitted line
#            runs before the density along it falls below n_iso (see
#            plot_cylinder_slice.walk_spine). Same colours as the spectra.
#
# The spines come from plot_cylinder_slice.get_spine_fits (40 seeds, each sample
# fit and walked on its own cube) and are cached in <outdir>/_spine_fits.
#
# Run through dust3d_pipeline.py --steps windows.

def plot_sample_spine_windows(cloud, prof_mean, sprofs, fits, out_png):
    n = len(fits['samples'])
    cols = sample_colors(n)
    lo, hi = fits['window']

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 6.5), dpi=150, sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1.6]},
                                 constrained_layout=True)
    ax.plot(ds, prof_mean, color='k', lw=1.6, label='mean map', zorder=5)
    for k in range(n):
        ax.plot(ds, sprofs[k], color=cols[k], lw=0.9, alpha=0.85,
                label=f'sample {fits["indep"][k]}')
    ax.set_ylabel(r'mass per slice ($M_\odot\,\mathrm{pc}^{-1}$)')
    ax.set_yscale('log')
    ax.legend(fontsize=8, ncol=4, frameon=False)
    ax.set_title(f'{cloud} -- mass spectra and the distance range of each sample\'s '
                 f'best-fit spine (n_iso = {fits["n_iso"]:g} cm$^{{-3}}$)', fontsize=11)

    rows = [('mean', fits['mean'], 'k')] + [(f'sample {fits["indep"][k]}', f, cols[k])
                                            for k, f in enumerate(fits['samples'])]
    for y, (lab, f, c) in enumerate(rows):
        bx.hlines(y, f['d_lo'], f['d_hi'], color=c, lw=6)
        bx.text(f['d_hi'] + 1, y, f'{f["d_lo"]:.0f}-{f["d_hi"]:.0f} pc', va='center',
                fontsize=7.5)
    bx.set_yticks(range(len(rows)))
    bx.set_yticklabels([r[0] for r in rows], fontsize=8)
    bx.invert_yaxis()
    for a in (ax, bx):
        a.axvspan(lo, hi, color='tab:red', alpha=0.08, zorder=0)
        a.axvline(lo, color='tab:red', ls='--', lw=1)
        a.axvline(hi, color='tab:red', ls='--', lw=1)
    bx.set_xlabel('heliocentric distance (pc)')
    # frame the plot on the region that matters rather than the whole 1182 pc axis
    dmin = min(lo, min(f['d_lo'] for _, f, _ in rows))
    dmax = max(hi, max(f['d_hi'] for _, f, _ in rows))
    span = dmax - dmin
    bx.set_xlim(dmin - 0.3 * span, dmax + 0.3 * span)
    fig.savefig(out_png, bbox_inches='tight')
    print('saved', out_png)


# ============================================================================
# Masses inside the A_V = 0.5 / 1 mag contour, threshold chosen by the leak test
#   (from compute_av1_masses.py)
# ============================================================================
# Cloud masses inside an A_V = 1 or 0.5 mag contour of the Edenhofer map -- the 3-D
# counterpart of Imara & Forbes (2023)'s A_V > 1 mag masses.
#
# Per cloud, on the unpadded catalog box:
#   A_V(l, b) = sum over the cloud's distance window (win_lo..win_hi) of n_H * 1 pc
#               / 2.2e21 cm^-2
#   threshold = 0.5 or 1 mag, from the leak test (--av-rule leak, the default): on
#               the same windowed A_V map over the 8 deg padded cube, label the
#               8-connected A_V >= 0.5 regions, take the one with the most in-box
#               A_V, and measure the share of its pixels outside the catalog box.
#               Leak < --leak-cut (0.3) -> 0.5 mag; otherwise 1 mag. --av-rule fixed
#               uses --av for every cloud.
#   aperture  = pixels of the catalog box with A_V >= threshold
#   M_av1     = mass of every voxel in the window along the aperture's sightlines
#               (1.36 m_p per H, (0.229 deg * d)^2 per pixel, 1 pc deep)
# Done on the mean map (M_av1) and on each independent posterior sample with its
# own A_V map and aperture (M_av1_sd = their standard deviation).
#
# Writes the columns into --out (adds or overwrites M_av1, M_av1_sd, av1_frac).
#
# Run through dust3d_pipeline.py --steps inputs.

NH_PER_AV = 2.2e21


def av1_mass(cube, win, av_cut):
    """(mass in Msun, aperture fraction) for one (nd, nb, nl) cube in cm^-3."""
    av = np.nansum(cube[win], axis=0) * PC_CM / NH_PER_AV
    ap = av >= av_cut
    prof = mass_profile(np.where(ap[None, :, :], cube, 0.0))   # Msun per 1 pc slice
    return float(prof[win].sum()), float(ap.mean())


def leak_05(table1, row, win):
    """Share of the cloud's A_V >= 0.5 mag region lying outside its catalog box,
    on the mean map over the PAD_STORED padded cube (see the module docstring)."""
    name = row['Name']
    mbox, (xo, yo) = mean_padded_box(table1, name, pad=PAD_STORED)
    av = np.nansum(mbox[win], axis=0) * PC_CM / NH_PER_AV
    lc, bc, _, (bxo, byo) = box_grid(row)
    cat = np.zeros(av.shape, bool)
    cat[byo - yo:byo - yo + len(bc), bxo - xo:bxo - xo + len(lc)] = True
    lab, k = ndi.label(av >= 0.5, np.ones((3, 3), bool))
    if k == 0:
        return np.nan
    insum = ndi.sum(np.where(cat, av, 0.0), lab, np.arange(1, k + 1))
    if insum.max() <= 0:
        return np.nan
    c = lab == int(np.argmax(insum)) + 1
    return float(1 - (c & cat).sum() / c.sum())


# ============================================================================
# rho0 vs r0 / alpha / M correlation figures, with the IF23 overlay
#   (from plot_rho0_correlations.py)
# ============================================================================
# rho0 vs r0 / alpha / M correlation figures, overlaid with the digitized
# Imara & Forbes (2023) points.
#
# Reads the rho0 inputs table (rho0, r0, alpha, cos_i, M_cyl and their sample
# spreads) and the digitized IF23 table; x = rho0 / cos i, IF23's scale.
# Run through dust3d_pipeline.py --steps rho0.

def rho_to_mass_density(rho0_cm3):
    """cm^-3 -> Msun/pc^3, as in dust_3d_functions.mass_profile (1.36*m_p)."""
    return (np.asarray(rho0_cm3) * u.cm**-3 * 1.36 * c.m_p).to(u.Msun / u.pc**3).value


def _style_log_axis(ax):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\rho_0\ (M_\odot\ pc^{-3})$')
    ax.grid(True, which='major', alpha=0.25)
    ax.grid(True, which='minor', alpha=0.1)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=8))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2, 5), numticks=8))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def x_rho(df):
    """rho0 in Msun/pc^3, divided by cos i when the CSV carries a cos_i column
    (i = spine inclination to the plane of the sky). IF23's x axis is
    rho0/cos i -- the amplitude a column-density fit to an inclined cylinder
    returns -- so this puts our 3-D rho0 on their scale. Returns (x, label)."""
    x = rho_to_mass_density(df['rho0'].to_numpy())
    if 'cos_i' in df.columns:
        return x / df['cos_i'].to_numpy(), r'$\rho_0 / \cos i\ (M_\odot\ pc^{-3})$'
    return x, r'$\rho_0\ (M_\odot\ pc^{-3})$'


# what each mass column measures, for legends and axis labels
MASS_LABELS = {'M_cyl': r'mass between the spine caps, out to $r_{\rm dip}$'}


def x_err(df, x):
    """1-sigma x error bars from rho0_sd (posterior-sample spread), on the same
    scale as x_rho's x (the cos i factor is a constant per cloud), or None."""
    if 'rho0_sd' not in df.columns:
        return None
    return x * df['rho0_sd'].to_numpy(float) / df['rho0'].to_numpy(float)


def _sample_errorbars(ax, x, df, col, xerr):
    """Posterior-sample error bars for one panel: y from <col>_sd, x from
    rho0_sd. Drawn under the markers; nothing if neither column exists."""
    yerr = df[col + '_sd'].to_numpy(float) if col + '_sd' in df.columns else None
    if yerr is None and xerr is None:
        return
    ax.errorbar(x, df[col].to_numpy(float), xerr=xerr, yerr=yerr, fmt='none',
                ecolor='tab:blue', elinewidth=1, capsize=2, alpha=0.7, zorder=2.5)


def loglog_fit(x, y):
    """Unweighted least-squares line in log-log space, log y = m log x + b.
    Returns (m, m_err, b, R^2), m_err from the fit covariance."""
    lx, ly = np.log10(np.asarray(x, float)), np.log10(np.asarray(y, float))
    (m, b), cov = np.polyfit(lx, ly, 1, cov=True)
    resid = ly - (m * lx + b)
    r2 = 1 - np.sum(resid**2) / np.sum((ly - ly.mean())**2)
    return m, np.sqrt(cov[0, 0]), b, r2


def _fit_line(ax, x, y, color, ls, who):
    """Draw the loglog_fit line over the x range of its own points; returns the fit."""
    m, me, b, r2 = loglog_fit(x, y)
    xf = np.geomspace(np.min(x), np.max(x), 100)
    ax.plot(xf, 10**(m * np.log10(xf) + b), color=color, ls=ls, lw=1.6, zorder=1.5,
            label=rf'{who} fit: slope {m:.2f} $\pm$ {me:.2f}, $R^2$ = {r2:.2f}')
    return m, me, b, r2


@_default_fonts
def plot_single(csv_n5, ifb_csv, out_png, mass_col='M_cyl', n_iso=5):
    """rho0 vs r0/alpha/M, one threshold, vs the IF23 points, with a log-log
    least-squares fit to each set drawn in each panel (IF23 publish no fits;
    the r0 panel also shows their stated r0 ~ 10 pc / (rho0/cos i), Sec. 5).
    Returns {panel: {'ours': fit, 'IF23': fit}} with fit = (m, m_err, b, R^2)."""
    df = pd.read_csv(csv_n5)
    ifb = pd.read_csv(ifb_csv)

    rho_mass, xlab = x_rho(df)
    ifb_rho = ifb['rho0_over_cosi'].to_numpy()
    ifb_xerr = (np.vstack([ifb_rho - ifb['rho0_lo'], ifb['rho0_hi'] - ifb_rho])
                if 'rho0_lo' in ifb.columns else None)
    ifb_cols = {'r0': ('r0_pc', 'r0_lo', 'r0_hi'), 'alpha': ('alpha', 'alpha_lo', 'alpha_hi'),
                mass_col: ('M_Av1_Msun', 'M_Av1_lo', 'M_Av1_hi')}

    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), dpi=150)

    xerr = x_err(df, rho_mass)

    fits = {}
    for ax, col in zip(axs, ('r0', 'alpha', mass_col)):
        y = df[col].to_numpy(float)
        _sample_errorbars(ax, rho_mass, df, col, xerr)
        ax.scatter(rho_mass, y, s=45, color='tab:blue', edgecolor='white',
                   linewidth=0.6, zorder=3)
        v, lo, hi = ifb_cols[col]
        iy = ifb[v].to_numpy(float)
        ax.errorbar(ifb_rho, iy, xerr=ifb_xerr, yerr=np.vstack([iy - ifb[lo], ifb[hi] - iy]),
                    fmt='D', ms=5.5, color='tab:red', ecolor='lightcoral', elinewidth=1,
                    capsize=2, markeredgecolor='white', markeredgewidth=0.5, zorder=2, alpha=0.85)
        fits[col] = {'ours': _fit_line(ax, rho_mass, y, 'tab:blue', '-', 'this work'),
                     'IF23': _fit_line(ax, ifb_rho, iy, 'tab:red', '--', 'IF23')}
        if col == 'r0':
            xr = np.geomspace(min(rho_mass.min(), ifb_rho.min()), max(rho_mass.max(), ifb_rho.max()), 50)
            ax.plot(xr, 10 / xr, color='0.35', ls=':', lw=1.4, zorder=1.4,
                    label=r'IF23 Sec. 5: $r_0 = 10\,{\rm pc}\,/\,(\rho_0/\cos i)$')

    axs[0].set_ylabel('$r_0$ (pc)')
    axs[1].set_ylabel(r'$\alpha$')
    axs[2].set_ylabel(rf'$M\ (M_\odot)$   [{MASS_LABELS.get(mass_col, mass_col)}]')

    for ax in axs:
        _style_log_axis(ax)
        ax.set_xlabel(xlab)
        ax.legend(loc='lower left', fontsize=7.5, framealpha=0.9)
    pts = [Line2D([], [], marker='o', ls='', ms=7, color='tab:blue', mec='white',
                  label=f'this work ({len(df)} clouds, n = {n_iso:g} cm$^{{-3}}$)'),
           Line2D([], [], marker='D', ls='', ms=6, color='tab:red', mec='white',
                  label=r'Imara & Forbes 2023 (8 clouds; Figs. 4, 5; $M$: $A_V > 1$ mag)')]
    fig.legend(handles=pts, loc='upper center', ncol=len(pts), fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(r'$\rho_0$ vs $r_0$ / $\alpha$ / $M$, each with a log-log least-squares fit',
                 y=1.06, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    print('saved', out_png)
    # the parameter panels and the mass panel on their own, for the paper
    save_axes_region(fig, axs[:2], out_png[:-4] + '_r0_alpha.png', whole_rows=False)
    save_axes_region(fig, axs[2:], out_png[:-4] + '_M.png', whole_rows=False)
    for col, f in fits.items():
        print(f'  {col:6s} slope  ours {f["ours"][0]:+.2f} +- {f["ours"][1]:.2f} (R2 {f["ours"][3]:.2f})'
              f'   IF23 {f["IF23"][0]:+.2f} +- {f["IF23"][1]:.2f} (R2 {f["IF23"][3]:.2f})')
    return fits


# ============================================================================
# Our version of Imara & Forbes (2023) Fig. 7
#   (from plot_if23_fig7.py)
# ============================================================================
# Our version of Imara & Forbes (2023) Figure 7: cloud mass, and the cloud's two
# sizes, against the Plummer scale radius r0.
#
#   left   M (inside the A_V = 0.5 or 1 mag contour, by the leak test; M_av1 from
#          compute_av1_masses.py) vs r0                       [IF23: M_1AV vs r0]
#   right  filled: projected spine length L cos i, L = the walked spine length
#          (t_neg + t_pos of the mean 40-seed spine), i = its inclination to the
#          plane of the sky                                    [IF23: b_par,max]
#          open:   r_dip, the radius where the radial profile stops falling
#                                                              [IF23: b_perp,max]
#
# r0, r_dip, M_av1, cos_i come from rho0_inputs_n10_linfit.csv; L and its spread
# over the posterior-sample spines from the _spine_fits pickles. Each panel also
# carries a log-log least-squares slope (IF23: both sizes scale with r0 with a
# log slope below 1).
#
# Run through dust3d_pipeline.py --steps if23.

MARKERS = ['o', 's', 'p', 'P', 'D', 'h', 'X', '^', 'v', '<', '>', '*', 'd']


def spine_sky(rec):
    """(3-D length, cos i) of one spine-fit record."""
    u = rec['p2'] - rec['p1']
    u = u / np.linalg.norm(u)
    los = np.asarray(rec['start'], float)
    los = los / np.linalg.norm(los)
    return rec['t_neg'] + rec['t_pos'], np.sqrt(1 - min(1.0, abs(u @ los)) ** 2)


def slope(x, y):
    m, b = np.polyfit(np.log10(x), np.log10(y), 1)
    return m


# ============================================================================
# Per-cloud sky-plane inclination and Galactic position angle (from the CSV angles)
#   (from plot_spine_inclinations.py)
# ============================================================================
# Two-panel summary of each cloud's spine orientation, built from the
# pa_lb/inc_ld/inc_bd columns `sample_summary` writes to
# spine_angles_and_fits_samples*.csv (see spine_descriptors() in
# choosing_distances_from_healpix.py for how those three angles are defined).
#
# Panel (a) -- inclination out of the plane of the sky. inc_ld and inc_bd are
# each already "angle out of the sky plane," but in two different projections
# (the l-d plane and the b-d plane); this reconstructs the single physical
# inclination of the 3D spine vector relative to the sky-tangent plane at the
# cloud's own position. Reconstruction: spine_descriptors defines
#     inc_ld = atan2(dd, kl*dl),  inc_bd = atan2(dd, kb*db)
# so, taking dd = 1 as a reference unit, kl*dl = cot(inc_ld) and kb*db =
# cot(inc_bd) up to sign, giving a local Cartesian direction (cot(inc_ld),
# cot(inc_bd), 1) whose angle above the sky plane (z=0) is
#     i_sky = arcsin(1 / sqrt(cot(inc_ld)^2 + cot(inc_bd)^2 + 1)).
# Its uncertainty is propagated by Monte Carlo, drawing inc_ld/inc_bd as
# independent Gaussians from the CSV's mean +- sd (an approximation: the two
# are derived from the same 6 posterior-sample vectors, so not truly
# independent, but no per-sample raw vectors are saved to do better).
#
# Panel (b) -- orientation relative to the Galactic plane. pa_lb is the spine's
# position angle in the l-b plane, measured from the l-axis -- and the l-axis
# is, by construction, always the local Galactic-plane direction (a line of
# constant b), regardless of where the cloud sits on the sky. So pa_lb IS
# already the spine's tilt relative to the Galactic plane as seen in projection
# on the sky; it's plotted directly, no reconstruction needed. 0 deg = aligned
# with the Galactic plane, +-90 deg = perpendicular to it (pointed at a pole).
#
# Run through dust3d_pipeline.py --steps inclinations.

N_MC = 20000


RNG_SEED = 0


def sky_plane_inclination(inc_ld_deg, inc_bd_deg, inc_ld_sd_deg, inc_bd_sd_deg, n_mc=N_MC, seed=RNG_SEED):
    """Monte-Carlo mean +- sd of the reconstructed total out-of-sky-plane
    inclination, per cloud. See module docstring for the derivation."""
    rng = np.random.default_rng(seed)
    means, sds = [], []
    for ld, bd, ld_sd, bd_sd in zip(inc_ld_deg, inc_bd_deg, inc_ld_sd_deg, inc_bd_sd_deg):
        ld_draw = np.deg2rad(rng.normal(ld, max(ld_sd, 1e-6), n_mc))
        bd_draw = np.deg2rad(rng.normal(bd, max(bd_sd, 1e-6), n_mc))
        cot_ld = 1.0 / np.tan(ld_draw)
        cot_bd = 1.0 / np.tan(bd_draw)
        i_sky = np.arcsin(1.0 / np.sqrt(cot_ld**2 + cot_bd**2 + 1.0))
        i_sky_deg = np.degrees(np.abs(i_sky))
        means.append(i_sky_deg.mean())
        sds.append(i_sky_deg.std(ddof=1))
    return np.array(means), np.array(sds)


@_default_fonts
def plot_inclinations(csv_path, out_png):
    df = pd.read_csv(csv_path).sort_values('cloud').reset_index(drop=True)

    i_sky, i_sky_sd = sky_plane_inclination(
        df['inc_ld'].to_numpy(), df['inc_bd'].to_numpy(),
        df['inc_ld_sd'].to_numpy(), df['inc_bd_sd'].to_numpy())

    x = np.arange(len(df))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)

    axs[0].errorbar(x, i_sky, yerr=i_sky_sd, fmt='o', ms=6, color='tab:blue',
                     ecolor='tab:blue', elinewidth=1, capsize=3,
                     markeredgecolor='white', markeredgewidth=0.6)
    axs[0].set_ylabel('inclination out of the plane of the sky (deg)')
    axs[0].set_ylim(0, 90)
    axs[0].axhline(0, color='0.7', lw=0.8, zorder=0)
    axs[0].set_title('(a) tilt out of the sky plane\n'
                      r'0$^\circ$ = in the sky plane, 90$^\circ$ = along the line of sight',
                      fontsize=10)

    axs[1].errorbar(x, df['pa_lb'], yerr=df['pa_lb_sd'], fmt='o', ms=6, color='tab:orange',
                     ecolor='tab:orange', elinewidth=1, capsize=3,
                     markeredgecolor='white', markeredgewidth=0.6)
    axs[1].set_ylabel('position angle from the Galactic plane (deg)')
    axs[1].set_ylim(-90, 90)
    axs[1].axhline(0, color='0.7', lw=0.8, zorder=0)
    axs[1].set_title('(b) orientation relative to the Galactic plane\n'
                      r'0$^\circ$ = aligned with it, $\pm$90$^\circ$ = perpendicular (pointed at a pole)',
                      fontsize=10)

    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels(df['cloud'], rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.25)

    fig.suptitle('Spine orientation of each cloud', y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    print('saved', out_png)


@_default_fonts
def plot_inclination_scatter(csv_path, out_png):
    """Same two quantities as plot_inclinations, as one scatter: sky-plane
    tilt (x) vs Galactic-plane position angle (y), each cloud a labeled point
    with both error bars."""
    df = pd.read_csv(csv_path).sort_values('cloud').reset_index(drop=True)

    i_sky, i_sky_sd = sky_plane_inclination(
        df['inc_ld'].to_numpy(), df['inc_bd'].to_numpy(),
        df['inc_ld_sd'].to_numpy(), df['inc_bd_sd'].to_numpy())

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.errorbar(i_sky, df['pa_lb'], xerr=i_sky_sd, yerr=df['pa_lb_sd'],
                fmt='o', ms=6, color='tab:blue', ecolor='0.6', elinewidth=1,
                capsize=3, markeredgecolor='white', markeredgewidth=0.6, zorder=3)

    for xi, yi, name in zip(i_sky, df['pa_lb'], df['cloud']):
        ax.annotate(name, (xi, yi), textcoords='offset points', xytext=(6, 4), fontsize=8)

    ax.axhline(0, color='0.8', lw=0.8, zorder=0)
    ax.set_xlim(0, 90)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('inclination out of the sky plane (deg)\n'
                  r'0$^\circ$ = in the sky plane, 90$^\circ$ = along the line of sight')
    ax.set_ylabel('position angle from the Galactic plane (deg)\n'
                  r'0$^\circ$ = aligned with it, $\pm$90$^\circ$ = perpendicular (pointed at a pole)')
    ax.grid(True, alpha=0.25)
    ax.set_title('Spine orientation of each cloud', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    print('saved', out_png)


# ============================================================================
# Sky-coordinate (l-b, l-d, b-d) dust maps with the cloud contours and spines
#   (from plot_sky_projections.py)
# ============================================================================
# The sky-coordinate counterpart of plot_xyz_projections.py: three projections of
# the Edenhofer+2023 dust in (l, b, d), the fitted spines drawn over them, and under
# each panel a histogram of how far the clouds' spines tip out of that plane.
#
#   a  l-b   (the plane of the sky), integrated along d   (d <= DMAX)
#   b  l-d   integrated along b
#   c  b-d   integrated along l       (d on the horizontal axis, as in plot_moment_maps)
#
# "Integrated along" means the column of n_H through that coordinate direction, with
# the physical path length of each voxel: 1 pc along d, d*dB (rad) along b and
# d*cos(b)*dL (rad) along l. Only panel a is a true line-of-sight column, so only it
# is the observed A_V; b and c are columns along a meridian / a parallel of latitude.
# All are N_H / 2.2e21 (Guver & Ozel 2009) on one shared colour scale.
#
# Top row, per cloud, masked to its own catalog sky box (+PAD_DEG) and its fitted
# distance window (win_lo/win_hi from the spine_angles_and_fits CSV):
#     green  n_H = --n-iso cm^-3   (max along the projection axis)
#     black  A_V = 1 mag           (column along the projection axis)
#     cyan   the mean-map best-fit spine, from its l,b,d endpoints to where the
#            density along it falls below n_iso (the same stretch plot_cylinder_slice draws)
#
# Bottom row: over the clouds, the angle between the 3-D spine and that plane, in the
# local (e_l, e_b, e_d) frame at the cloud: l-b uses |e_d| (inclination to the plane
# of the sky), l-d uses |e_b|, b-d uses |e_l|. 0 deg = lying in the plane, 90 deg =
# perpendicular. Bars: the mean-map spines. Line: all per-sample spines, counts / 6.
# Dashed: what isotropically random orientations would put in each bin.
#
# Maps of the whole sky come from edenhofer_lbd_wrap.fits (6.8 GB, read once, ~2 min,
# cached); the per-cloud contours from the stored mean sample cubes (mean of the
# 12 posterior samples).
#
# Run through dust3d_pipeline.py --steps projections.

PC = 3.0856775814913673e18


AV = 2.2e21


CDR = np.deg2rad(CD)


DMAX = 600.0                     # pc: d range of the maps


BLIM = 45.0                      # deg: half-height of the b axis shown


NL = int(round(360.0 / CD))      # 1572 unique columns


X0 = int(np.ceil((L0 - 180.0) / CD))   # column where l = +180


BINS = np.arange(0, 91, 15.0)


FS = 8.0


def wrap_l(l):
    return (np.asarray(l) + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------
def sky_build_maps(cat, df, cache):
    key = np.concatenate([df.win_lo.values, df.win_hi.values,
                          cat[['l_min(deg)', 'l_max(deg)', 'b_min(deg)', 'b_max(deg)']].values.ravel()])
    if os.path.exists(cache):
        z = pickle.load(open(cache, 'rb'))
        if z['key'].shape == key.shape and np.allclose(z['key'], key):
            print('[cache]', cache)
            return z
        print('[cache] stale, rebuilding')

    t0 = time.time()
    cube = fits.open(WRAP, memmap=True)[0].data                  # [d, b, l]
    nb = cube.shape[1]
    b_c = -90.0 + CD * np.arange(nb)
    cosb = np.cos(np.deg2rad(b_c))
    nd = int(DMAX - D0) + 1
    d_c = D0 + np.arange(nd)
    lb = np.zeros((nb, NL)); cnt_lb = np.zeros((nb, NL), dtype=np.int32)
    ld = np.zeros((nd, NL)); bd = np.zeros((nb, nd))
    for k in range(nd):
        s = np.asarray(cube[k, :, X0:X0 + NL], dtype=np.float32)
        ok = np.isfinite(s); s0 = np.where(ok, s, 0.0)
        lb += s0; cnt_lb += ok
        ld[k, :] = s0.sum(axis=0) * d_c[k] * CDR
        bd[:, k] = s0.sum(axis=1) * cosb * d_c[k] * CDR
        if k % 100 == 0:
            print(f'  d {k}/{nd}  {time.time() - t0:.0f}s', flush=True)
    out = {'key': key, 'b_c': b_c, 'd_c': d_c,
           'lb': (lb * PC).astype(np.float32), 'ld': (ld * PC).astype(np.float32),
           'bd': (bd * PC).astype(np.float32)}
    out['lb'][cnt_lb == 0] = np.nan

    # ---- per cloud, from the stored mean cube, on box+PAD_DEG x window ----
    names = list(df.cloud)
    cl = []
    for i, nm in enumerate(names):
        row = cat.iloc[i]
        st = load_stored(cat, nm)
        mean = st['cube'].mean(axis=0)                           # [d, b, l], padded box
        lc, bc, _, (xo, yo) = box_grid(row, pad=PAD_DEG)
        y0, x0 = yo - st['yo'], xo - st['xo']
        sub = mean[:, y0:y0 + len(bc), x0:x0 + len(lc)]
        z0 = int(round(df.win_lo.values[i] - D0)); z1 = int(round(df.win_hi.values[i] - D0)) + 1
        sub = sub[z0:z1]; dz = ds[z0:z1]; cb = np.cos(np.deg2rad(bc))
        lb_max = sub.max(axis=0); lb_col = sub.sum(axis=0) * PC
        ld_max = sub.max(axis=1); ld_col = (sub.sum(axis=1) * (dz[:, None] * CDR)) * PC        # [d, l]
        bd_max = sub.max(axis=2).T                                                       # [b, d]
        bd_col = ((sub.sum(axis=2) * dz[:, None] * CDR) * cb[None, :]).T * PC            # [b, d]
        cl.append(dict(l=wrap_l(lc), b=bc, d=dz, lb_max=lb_max, lb_col=lb_col, ld_max=ld_max,
                       ld_col=ld_col, bd_max=bd_max, bd_col=bd_col))
        print(f'  {nm:18s} sub {sub.shape}', flush=True)
    out['clouds'] = cl
    os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
    pickle.dump(out, open(cache, 'wb'))
    print(f'[cache] wrote {cache}  ({time.time() - t0:.0f}s)')
    return out


# --------------------------------------------------------------------------
def local_frame(l_deg, b_deg):
    """e_l, e_b, e_d at (l, b) in the pipeline's xyz (l=0 along +y, l=90 along +x)."""
    l, b = np.deg2rad(l_deg), np.deg2rad(b_deg)
    e_l = np.array([np.cos(l), -np.sin(l), 0.0])
    e_b = np.array([-np.sin(b) * np.sin(l), -np.sin(b) * np.cos(l), np.cos(b)])
    e_d = np.array([np.cos(b) * np.sin(l), np.cos(b) * np.cos(l), np.sin(b)])
    return e_l, e_b, e_d


def sky_out_of_plane(p1, p2, l_deg, b_deg):
    """Angle (deg, 0-90) between the spine and the l-b, l-d and b-d planes."""
    u = (p2 - p1) / np.linalg.norm(p2 - p1)
    e_l, e_b, e_d = local_frame(l_deg, b_deg)
    a = np.abs([u @ e_d, u @ e_b, u @ e_l])
    return np.degrees(np.arcsin(np.clip(a, 0, 1)))


def place_labels(fig, ax, pts, names):
    """Greedy label placement: for each cloud try a ring of offsets and keep the
    first whose text box clears the boxes already placed, the other clouds'
    anchor points and the axes edge. Labels are annotations with a thin leader."""
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(rend)
    xy = ax.transData.transform(np.array(pts))
    sizes = []
    for nm in names:
        t = ax.text(0, 0, nm, fontsize=FS); bb = t.get_window_extent(rend)
        sizes.append((bb.width, bb.height)); t.remove()
    placed = []
    cand = [(dx, dy) for r in (16, 34, 60, 90, 130) for dx, dy in
            ((r, r * 0.45), (-r, r * 0.45), (r, -r * 0.45), (-r, -r * 0.45),
             (0, r * 0.7), (0, -r * 0.7), (r * 1.3, 0), (-r * 1.3, 0))]
    order = np.argsort(xy[:, 1])[::-1]
    res = {}
    for i in order:
        w, h = sizes[i]
        for dx, dy in cand:
            x0 = xy[i, 0] + dx - (w if dx < 0 else 0); y0 = xy[i, 1] + dy - h / 2
            box = (x0 - 3, x0 + w + 3, y0 - 2, y0 + h + 2)
            if box[0] < axbb.x0 or box[1] > axbb.x1 or box[2] < axbb.y0 or box[3] > axbb.y1:
                continue
            if any(box[0] < p[1] and p[0] < box[1] and box[2] < p[3] and p[2] < box[3] for p in placed):
                continue
            if any(box[0] - 4 < q[0] < box[1] + 4 and box[2] - 4 < q[1] < box[3] + 4
                   for j, q in enumerate(xy) if j != i):
                continue
            placed.append(box); res[i] = (dx, dy); break
        else:
            res[i] = cand[-1]; placed.append((xy[i, 0], xy[i, 0], xy[i, 1], xy[i, 1]))
    px = fig.dpi / 72.0
    for i, nm in enumerate(names):
        dx, dy = res[i]
        ax.annotate(nm, xy=pts[i], xytext=(dx / px, dy / px), textcoords='offset points',
                    fontsize=FS, color='black', ha='left' if dx >= 0 else 'right', va='center',
                    zorder=9, path_effects=[pe.withStroke(linewidth=2.6, foreground='white')],
                    arrowprops=dict(arrowstyle='-', color='#1d6f82', lw=0.5, alpha=0.85,
                                    shrinkA=0, shrinkB=0))


def place_labels_edges(fig, ax, pts, names, edge_frac=0.03, gap=6, leader_z=1,
                       leader_color='white'):
    """Labels in two rows, along the top edge for clouds at b > 0 and along the
    bottom edge for clouds at b < 0, each joined to its cloud by a thin leader.
    Within a row the labels sit at their cloud's l, pushed sideways only as far
    as needed to stop them overlapping. edge_frac: row height from the edge, as
    a fraction of the axis height. The leaders are drawn at leader_z, just above
    the map image, so contours, spines and scatter points sit on top of them."""
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(rend)
    pts = np.asarray(pts, float)
    xy = ax.transData.transform(pts)
    widths, height = [], 0.0
    for nm in names:
        t = ax.text(0, 0, nm, fontsize=FS); bb = t.get_window_extent(rend)
        widths.append(bb.width); height = max(height, bb.height); t.remove()
    widths = np.array(widths)
    for top in (True, False):
        idx = [i for i in range(len(names)) if (pts[i, 1] >= 0) == top]
        if not idx:
            continue
        idx = sorted(idx, key=lambda i: xy[i, 0])
        w = widths[idx]
        left = xy[idx, 0] - w / 2
        lo, hi = axbb.x0 + 4, axbb.x1 - 4
        for k in range(len(idx)):                     # push right past the previous label
            left[k] = max(left[k], lo if k == 0 else left[k - 1] + w[k - 1] + gap)
        for k in range(len(idx) - 1, -1, -1):         # then back left inside the axes
            left[k] = min(left[k], hi - w[k] if k == len(idx) - 1 else left[k + 1] - w[k] - gap)
        y = axbb.y1 - edge_frac * axbb.height if top else axbb.y0 + edge_frac * axbb.height
        inv = ax.transData.inverted()
        y_end = y - height / 2 - 2 if top else y + height / 2 + 2     # stop at the text's edge
        for k, i in enumerate(idx):
            xc = left[k] + w[k] / 2
            ax.text(*inv.transform((xc, y)), names[i], fontsize=FS, color='black',
                    ha='center', va='center', zorder=9,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground='white')])
            xe, ye = inv.transform((xc, y_end))
            ax.plot([pts[i, 0], xe], [pts[i, 1], ye], color=leader_color, lw=0.6, alpha=0.85,
                    zorder=leader_z)


# --------------------------------------------------------------------------
def sky_prepare(csv, fits_dir, cache, n_iso=N_ISO, fits_tag=None):
    """Maps (cached), per-cloud contours, spines, angles and the colour range."""
    NLEV = n_iso
    df = pd.read_csv(csv); names = list(df.cloud)
    cat = load_catalog().set_index('Name').loc[names].reset_index()
    G = sky_build_maps(cat, df, cache)
    b_c, d_c = G['b_c'], G['d_c']

    F = {nm: pickle.load(open(f'{fits_dir}/{nm}_spine_fits_{fits_tag or f"n{n_iso:g}_s40"}.pkl', 'rb'))
         for nm in names}
    ang_mean, ang_samp, curves = [], [], []
    for nm in names:
        f = F[nm]
        lbd = np.array(xyz_2_sph(f['mean']['start'])).ravel()     # (l, b, d) of the walk start
        ang_mean.append(sky_out_of_plane(f['mean']['p1'], f['mean']['p2'], lbd[0], lbd[1]))
        for s in f['samples']:
            sl = np.array(xyz_2_sph(s['start'])).ravel()
            ang_samp.append(sky_out_of_plane(s['p1'], s['p2'], sl[0], sl[1]))
        c = np.array(f['mean']['curve']).copy(); c[:, 0] = wrap_l(c[:, 0]); curves.append(c)
    ang_mean = np.array(ang_mean); ang_samp = np.array(ang_samp)
    ns_ = len(F[names[0]]['samples']); ncl = len(names)
    pos = np.column_stack([wrap_l(df.l.values), df.b.values, df.d.values])   # l, b, d

    # ---- maps in plotting orientation ----
    bsel = (b_c >= -BLIM) & (b_c <= BLIM)
    Alb = (G['lb'] / AV)[bsel]                                # [b, l]  (col 0 = l=+180)
    Ald = G['ld'] / AV                                        # [d, l]
    Abd = (G['bd'] / AV)[bsel]                                # [b, d]
    ext_lb = [180, -180, b_c[bsel][0] - CD / 2, b_c[bsel][-1] + CD / 2]
    ext_ld = [180, -180, d_c[0] - .5, d_c[-1] + .5]
    ext_bd = [d_c[0] - .5, d_c[-1] + .5, b_c[bsel][0] - CD / 2, b_c[bsel][-1] + CD / 2]
    VMIN = min(np.nanpercentile(A, 1) for A in (Alb, Ald, Abd))
    VMAX = max(np.nanpercentile(A, 99.9) for A in (Alb, Ald, Abd))
    norm = LogNorm(VMIN, VMAX); print('shared colour scale: %.3f - %.2f mag' % (VMIN, VMAX))

    return dict(locals())


def draw_maps(fig, S, norm, ax_lb=None, ax_ld=None, ax_bd=None, av_color='black',
              spine_color='#8fecff', niso_color='#39e639', labels='ring', cmap='magma',
              leader_color='white', lb_map=None, av_contour=True, gc_label=True,
              niso_contour=True):
    """Draw any of the l-b, l-d, b-d map panels into the axes given; returns an image.
    labels: 'ring' places each name next to its cloud (place_labels); 'edges' puts
    them in rows along the top / bottom edge by the sign of the cloud's b
    (place_labels_edges; l-b panel only, the others keep 'ring').
    lb_map: image for the l-b panel in place of S['Alb'] (same grid; norm must suit
    it). av_contour / niso_contour / gc_label: draw the A_V = 1 contours / the
    n_iso contours on box + PAD_DEG / the 'to GC' marker."""
    names, G, curves, pos, NLEV = S['names'], S['G'], S['curves'], S['pos'], S['NLEV']
    kw = dict(cmap=cmap, norm=norm, aspect='auto', interpolation='nearest', origin='lower')
    im = None
    if ax_lb is not None:
        im = ax_lb.imshow(S['Alb'] if lb_map is None else lb_map, extent=S['ext_lb'], **kw)
        ax_lb.set_xlim(180, -180); ax_lb.set_ylim(-BLIM, BLIM)
    if ax_ld is not None:
        im_ld = ax_ld.imshow(S['Ald'], extent=S['ext_ld'], **kw)
        im = im if im is not None else im_ld
        ax_ld.set_ylim(D0, DMAX)
    if ax_bd is not None:
        im_bd = ax_bd.imshow(S['Abd'], extent=S['ext_bd'], **kw)
        im = im if im is not None else im_bd
        ax_bd.set_xlim(D0, DMAX)
    lax = [a for a in (ax_lb, ax_ld) if a is not None]
    for ax in lax:
        ax.set_xticks(np.arange(-150, 181, 30)[::-1])
        for x in np.arange(-150, 181, 30):
            ax.axvline(x, color='white', lw=0.3, alpha=0.18, zorder=3)
    for ax in (ax_lb, ax_bd):
        if ax is not None:
            ax.axhline(0, color='white', lw=0.35, alpha=0.3, zorder=3)
    for ax in lax:
        ax.axvline(0, color='white', lw=0.8, alpha=0.45, zorder=3)
        if gc_label:
            ax.text(0, ax.get_ylim()[1] - (0.05 * np.diff(ax.get_ylim())[0] if ax is ax_lb else 12),
                    'to GC', color='black', fontsize=8, ha='center', va='top', zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])

    # per-cloud contours + spines
    for i, (nm, c) in enumerate(zip(names, G['clouds'])):
        Lg, Bg = np.meshgrid(c['l'], c['b'])                       # [b, l]
        Lg2, Dg2 = np.meshgrid(c['l'], c['d'])                     # [d, l]
        Dg3, Bg3 = np.meshgrid(c['d'], c['b'])                     # [b, d]
        for ax, X, Y, mx_, col_ in ((ax_lb, Lg, Bg, c['lb_max'], c['lb_col']),
                                    (ax_ld, Lg2, Dg2, c['ld_max'], c['ld_col']),
                                    (ax_bd, Dg3, Bg3, c['bd_max'], c['bd_col'])):
            if ax is None:
                continue
            A_ = np.nan_to_num(col_.astype(float) / AV, nan=0.0)
            if av_contour and A_.max() >= 1.0:
                ax.contour(X, Y, A_, levels=[1.0], colors=av_color, linewidths=1.0, zorder=5.6)
            if niso_contour:
                ax.contour(X, Y, np.nan_to_num(mx_.astype(float), nan=0.0), levels=[NLEV],
                           colors=niso_color, linewidths=0.8, zorder=5.8)
        cv = curves[i]
        for ax, x, y in ((ax_lb, cv[:, 0], cv[:, 1]), (ax_ld, cv[:, 0], cv[:, 2]),
                         (ax_bd, cv[:, 2], cv[:, 1])):
            if ax is None:
                continue
            ax.plot(x, y, color=spine_color, lw=2.1, solid_capstyle='round', zorder=5.2,
                    path_effects=[pe.withStroke(linewidth=3.6, foreground='black', alpha=0.6)])

    for ax, cols, xl, yl, ttl in (
            (ax_lb, [0, 1], 'Galactic longitude $\\ell$ [deg]', 'Galactic latitude $b$ [deg]',
             'a.  $\\ell$–$b$  (plane of the sky),  integrated along $d$,  $d\\leq$%d pc' % DMAX),
            (ax_ld, [0, 2], 'Galactic longitude $\\ell$ [deg]', 'distance $d$ [pc]',
             'b.  $\\ell$–$d$,  integrated along $b$'),
            (ax_bd, [2, 1], 'distance $d$ [pc]', 'Galactic latitude $b$ [deg]',
             'c.  $b$–$d$,  integrated along $\\ell$')):
        if ax is None:
            continue
        if labels == 'edges' and ax is ax_lb:
            place_labels_edges(fig, ax, pos[:, cols], names, leader_color=leader_color)
        else:
            place_labels(fig, ax, pos[:, cols], names)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=11, pad=8)
    return im


# ============================================================================
# Cartesian (X-Y, Y-Z, X-Z) dust maps with the cloud contours and spines
#   (from plot_xyz_projections.py)
# ============================================================================
# Three orthogonal projections of the Edenhofer+2023 xyz dust cube, with the
# fitted spines overlaid, and under each panel a histogram of how far the clouds'
# spines tip out of that plane.
#
#   a  face-on   (X-Y), integrated along Z  -- the Galactic plane from above
#   b  down X    (Y-Z), integrated along X  -- looking toward the Galactic centre
#   c  edge-on   (X-Z), integrated along Y  -- the disk from the side, Z vertical
#
# Panel a: horizontal = -Y, vertical = X (toward l=0 is UP).
# Panel b: horizontal = -Y, vertical = Z (viewing axis is X, into the page).
# Panel c: horizontal =  X, vertical = Z (toward l=0 is RIGHT; viewing axis is Y).
#
# Top row, per cloud, masked to its own catalog sky box (+PAD_DEG) and its fitted
# distance window (win_lo/win_hi from the spine_angles_and_fits CSV):
#     green  n_H = --n-iso cm^-3   (max along the projection axis)
#     black  A_V = 1 mag           (column along the projection axis)
#     cyan   the mean-map best-fit spine (40 seeds), projected into that plane
# A_V = N_H / 2.2e21 (Guver & Ozel 2009). All three are columns along a cartesian
# axis, NOT along the line of sight, so none is the observed A_V of these clouds.
#
# Bottom row: histogram over the clouds of the angle between the 3-D spine and
# that plane (0 deg = lying in the plane, 90 deg = perpendicular to it), from the
# same spines as the bars above: X-Y uses |Z|, Y-Z uses |X|, X-Z uses |Y| of the
# unit spine vector. Bars: the 14 mean-map spines. Line: all per-sample spines
# (6 independent posterior samples per cloud), counts divided by 6 to be
# comparable. Dashed: what isotropically random orientations would put in each bin.
#
# The spines come from plot_cylinder_slice.get_spine_fits (cached pickles). The
# pipeline's xyz has l=0 along +y; this figure has l=0 along +X, so x and y are
# swapped on the way in.
#
# Rebuilding the maps streams the 15.6 GB xyz cube once (~2 min) and is cached.
#
# Run through dust3d_pipeline.py --steps projections.

XYZ_K = 2.8 * 2.2e21 / PC            # E/pc -> n_H [cm^-3]


HALF = 760.0                      # half-width (pc) of the per-cloud subgrid; covers every win_hi


RMAX = 600.0; RLON = 627.0; RL = 520.0; LIM = 650.0; DPI = 300


RMAX_B = 400.0; RLON_B = 427.0; RL_B = 346.7; LIM_B = 450.0      # panel b zoom (r<=400 pc)


XYZ_SPINE_L = 29.0; FS, XYZ_PAD = 8.0, 14.0; MX, MY = 46.0, 10.0


Z_SPINE, Z_AV, Z_N5 = 5.2, 5.6, 5.8


# --------------------------------------------------------------------------
# maps: one pass over the cube
# --------------------------------------------------------------------------
def xyz_build_maps(cat, df, cache):
    """Global columns along Z, X, Y and, per cloud, max n_H and column along each
    axis inside the cloud's sky box and distance window. Cached in `cache`,
    keyed on the windows and boxes."""
    key = np.concatenate([df.win_lo.values, df.win_hi.values,
                          cat[['l_min(deg)', 'l_max(deg)', 'b_min(deg)', 'b_max(deg)']].values.ravel()])
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if z['key'].shape == key.shape and np.allclose(z['key'], key):
            print('[cache]', cache)
            return {k: z[k] for k in z.files}
        print('[cache] stale, rebuilding')

    D = fits.open(XYZ_CUBE, memmap=True)[1].data                  # axes [Z, Y, X]
    N = D.shape[0]
    gfull = (np.arange(N) - (N - 1) / 2) * 2.0
    j0 = int(round((-HALF + 1250) / 2)); j1 = int(round((HALF + 1250) / 2)) + 1
    gs = gfull[j0:j1]; ns = len(gs); nc = len(df)
    lmin = cat['l_min(deg)'].values - PAD_DEG; lmax = cat['l_max(deg)'].values + PAD_DEG
    bmin = cat['b_min(deg)'].values - PAD_DEG; bmax = cat['b_max(deg)'].values + PAD_DEG
    wlo = df.win_lo.values; whi = df.win_hi.values

    colZ = np.zeros((N, N)); cntZ = np.zeros((N, N), dtype=np.int32)    # [Y, X]
    colX = np.zeros((N, N)); cntX = np.zeros((N, N), dtype=np.int32)    # [Z, Y]
    colY = np.zeros((N, N)); cntY = np.zeros((N, N), dtype=np.int32)    # [Z, X]
    # per cloud, each in the plane's own [row, col]:  xy [Y,X], yz [Z,Y], xz [Z,X]
    mx = {p: np.full((nc, ns, ns), -np.inf, dtype=np.float32) for p in ('xy', 'yz', 'xz')}
    cl = {p: np.zeros((nc, ns, ns)) for p in ('xy', 'yz', 'xz')}
    hit = {p: np.zeros((nc, ns, ns), dtype=bool) for p in ('xy', 'yz', 'xz')}

    YY, XX = np.meshgrid(gs, gs, indexing='ij'); RYX = np.hypot(XX, YY)
    lon = np.degrees(np.arctan2(YY, XX))
    t0 = time.time()
    for k in range(N):
        Z = gfull[k]
        s = np.asarray(D[k], dtype=np.float32) * XYZ_K            # (Y, X)
        ok = np.isfinite(s); s0 = np.where(ok, s, 0.0)
        colZ += s0; cntZ += ok
        colX[k, :] = s0.sum(axis=1); cntX[k, :] = ok.sum(axis=1)
        colY[k, :] = s0.sum(axis=0); cntY[k, :] = ok.sum(axis=0)
        if j0 <= k < j1:
            ss = s[j0:j1, j0:j1]; okk = ok[j0:j1, j0:j1]; kk = k - j0
            r = np.sqrt(RYX**2 + Z * Z)
            lat = np.degrees(np.arcsin(np.clip(Z / np.maximum(r, 1e-9), -1, 1)))
            for c in range(nc):
                lo, hi = lmin[c], lmax[c]
                m = (okk & (((lon - lo) % 360.0) <= ((hi - lo) % 360.0)) & (lat >= bmin[c])
                     & (lat <= bmax[c]) & (r >= wlo[c]) & (r <= whi[c]))
                if not m.any():
                    continue
                v = np.where(m, ss, -np.inf); v0 = np.where(m, ss, 0.0)
                mx['xy'][c] = np.maximum(mx['xy'][c], v); cl['xy'][c] += v0; hit['xy'][c] |= m
                mx['yz'][c, kk, :] = v.max(axis=1); cl['yz'][c, kk, :] = v0.sum(axis=1)
                hit['yz'][c, kk, :] = m.any(axis=1)
                mx['xz'][c, kk, :] = v.max(axis=0); cl['xz'][c, kk, :] = v0.sum(axis=0)
                hit['xz'][c, kk, :] = m.any(axis=0)
        if k % 150 == 0:
            print(f'  z {k}/{N}  {time.time() - t0:.0f}s', flush=True)

    out = {'key': key, 'g': gs, 'gfull': gfull, 'names': np.array(list(df.cloud))}
    for nm, col, cnt in (('xy', colZ, cntZ), ('yz', colX, cntX), ('xz', colY, cntY)):
        NH = (col * 2.0 * PC).astype(np.float32); NH[cnt == 0] = np.nan
        out['NH_' + nm] = NH; out['cnt_' + nm] = cnt
        a = mx[nm]; a[~np.isfinite(a) | ~hit[nm]] = np.nan
        out['acc_' + nm] = a
        c_ = (cl[nm] * 2.0 * PC).astype(np.float32); c_[~hit[nm]] = np.nan
        out['col_' + nm] = c_
    os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
    np.savez_compressed(cache, **out)
    print(f'[cache] wrote {cache}  ({time.time() - t0:.0f}s)')
    return out


# --------------------------------------------------------------------------
# spines
# --------------------------------------------------------------------------
def unit_fig(p1, p2):
    """Unit spine vector in THIS figure's frame (X toward l=0, Y toward l=90, Z
    north); the pipeline's xyz has l=0 along +y, so swap x and y."""
    u = (p2 - p1) / np.linalg.norm(p2 - p1)
    return np.array([u[1], u[0], u[2]])


def xyz_out_of_plane(u):
    """Angle (deg, 0-90) between a unit vector and each plane: X-Y, Y-Z, X-Z."""
    a = np.abs(u)
    return np.degrees(np.arcsin(np.clip([a[2], a[0], a[1]], 0, 1)))


# --------------------------------------------------------------------------
def xyz_prepare(csv, fits_dir, cache, n_iso=N_ISO, fits_tag=None):
    """Everything the panels need: maps (cached), spines, angles, colour range."""
    NLEV = n_iso
    df = pd.read_csv(csv); names = list(df.cloud)
    cat = load_catalog().set_index('Name').loc[names].reset_index()
    M = xyz_build_maps(cat, df, cache)
    g, gfull = M['g'], M['gfull']

    # ---- spines ----
    fits_ = {nm: pickle.load(open(f'{fits_dir}/{nm}_spine_fits_{fits_tag or f"n{n_iso:g}_s40"}.pkl', 'rb'))
             for nm in names}
    u_mean = np.array([unit_fig(f['mean']['p1'], f['mean']['p2']) for f in fits_.values()])
    u_samp = [np.array([unit_fig(s['p1'], s['p2']) for s in f['samples']]) for f in fits_.values()]
    ang_mean = np.array([xyz_out_of_plane(u) for u in u_mean])                       # (ncl, 3)
    ang_samp = np.concatenate([[xyz_out_of_plane(u) for u in us] for us in u_samp])  # (ncl*6, 3)

    l = np.deg2rad(df.l.values); b = np.deg2rad(df.b.values); d = df.d.values
    Xc = d * np.cos(b) * np.cos(l); Yc = d * np.cos(b) * np.sin(l); Zc = d * np.sin(b)
    dX, dY, dZ = u_mean[:, 0], u_mean[:, 1], u_mean[:, 2]

    # ---- map arrays in plotting orientation ----
    NHa = M['NH_xy'].astype(float); NHa[M['cnt_xy'] == 0] = np.nan
    Aa = np.rot90(NHa / AV, -1)                                   # -> [V=X, H=-Y]
    na = Aa.shape[0]; ga = (np.arange(na) - (na - 1) / 2) * 2.0
    aGY, aGX = np.meshgrid(ga, ga, indexing='ij')
    Aa = np.where(np.hypot(aGX, aGY) <= RMAX, Aa, np.nan)
    aY, aX = np.meshgrid(g, g, indexing='ij'); caH, caV = -aY, aX
    Ha, Va = -Yc, Xc; dHa, dVa = -dY, dX

    NHb = M['NH_yz'].astype(float) / AV; NHb[M['cnt_yz'] == 0] = np.nan      # [Z, Y]
    bVc, bYc = np.meshgrid(gfull, gfull, indexing='ij')
    Ab = np.where(np.hypot(-bYc, bVc) <= RMAX_B, NHb, np.nan)[:, ::-1]
    bZg, bYg = np.meshgrid(g, g, indexing='ij'); bHg, bVg = -bYg, bZg
    Hb, Vb = -Yc, Zc; dHb, dVb = -dY, dZ

    NHc = M['NH_xz'].astype(float) / AV; NHc[M['cnt_xz'] == 0] = np.nan      # [Z, X]
    cVc_, cXc_ = np.meshgrid(gfull, gfull, indexing='ij')
    Ac = np.where(np.hypot(cXc_, cVc_) <= RMAX, NHc, np.nan)
    cZg, cXg = np.meshgrid(g, g, indexing='ij'); cHg, cVg = cXg, cZg
    Hc, Vc = Xc, Zc; dHc, dVc = dX, dZ

    VMIN = min(np.nanpercentile(A, 1) for A in (Aa, Ab, Ac))
    VMAX = max(np.nanpercentile(A, 99.9) for A in (Aa, Ab, Ac))
    norm = LogNorm(VMIN, VMAX); print('shared colour scale: %.3f - %.2f mag' % (VMIN, VMAX))

    return dict(locals())


def panel_cfgs(S):
    """The three panels' settings: a face-on X-Y, b Y-Z, c edge-on X-Z."""
    Aa, Ab, Ac, M, gfull = S['Aa'], S['Ab'], S['Ac'], S['M'], S['gfull']
    caH, caV, bHg, bVg, cHg, cVg = S['caH'], S['caV'], S['bHg'], S['bVg'], S['cHg'], S['cVg']
    Ha, Va, dHa, dVa = S['Ha'], S['Va'], S['dHa'], S['dVa']
    Hb, Vb, dHb, dVb = S['Hb'], S['Vb'], S['dHb'], S['dVb']
    Hc, Vc, dHc, dVc = S['Hc'], S['Vc'], S['dHc'], S['dVc']
    cfg = [dict(A=Aa, ext=[-1250, 1250, -1250, 1250], cH=caH, cV=caV, acc=M['acc_xy'],
                col=M['col_xy'], H=Ha, V=Va, dH=dHa, dV=dVa, spokes=True, mid=False, gc_axis='y',
                rmax=RMAX, rlon=RLON, rl=RL, lim=LIM,
                xlab='$-$Y  [pc]        ($\\ell=90°$ left,  $\\ell=270°$ right)',
                ylab='X  [pc]        (toward $\\ell=0°$, Galactic centre)',
                ttl='a.  face-on  (X$-$Y),  integrated along Z'),
           dict(A=Ab, ext=[gfull[0], gfull[-1], gfull[0], gfull[-1]], cH=bHg, cV=bVg,
                acc=M['acc_yz'], col=M['col_yz'], H=Hb, V=Vb, dH=dHb, dV=dVb, spokes=False, mid=True,
                gc_axis=None, rmax=RMAX_B, rlon=RLON_B, rl=RL_B, lim=LIM_B,
                xlab='$-$Y  [pc]        ($\\ell=90°$ left,  $\\ell=270°$ right)',
                ylab='Z  [pc]        (toward Galactic north)',
                ttl='b.  down the X axis  (Y$-$Z),  integrated along X,  $r\\leq400$ pc'),
           dict(A=Ac, ext=[gfull[0], gfull[-1], gfull[0], gfull[-1]], cH=cHg, cV=cVg,
                acc=M['acc_xz'], col=M['col_xz'], H=Hc, V=Vc, dH=dHc, dV=dVc, spokes=False, mid=True,
                gc_axis='x', rmax=RMAX, rlon=RLON, rl=RL, lim=LIM,
                xlab='X  [pc]        (toward $\\ell=0°$, Galactic centre)',
                ylab='Z  [pc]        (toward Galactic north)',
                ttl='c.  edge-on  (X$-$Z),  integrated along Y')]
    return cfg


def draw_panel(fig, ax, P, S, norm, cmap='magma', cloud_contours=True):
    """Draw one map panel (an entry of panel_cfgs) into `ax`; returns the image.
    cloud_contours=False skips the per-cloud A_V = 1 and n_iso contours."""
    names, NLEV = S['names'], S['NLEV']
    th = np.linspace(0, 2 * np.pi, 400)
    ax.set_xlim(-P['lim'], P['lim']); ax.set_ylim(-P['lim'], P['lim']); ax.set_aspect(1)
    fig.canvas.draw(); rend = fig.canvas.get_renderer()

    def solve_labels(ax, H, V, RL):
        inv = ax.transData.inverted(); W = []; HT = []
        for nm_ in names:
            t = ax.text(0, 0, nm_, fontsize=FS); bb = t.get_window_extent(renderer=rend)
            (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            W.append(abs(x1 - x0)); HT.append(abs(y1 - y0)); t.remove()
        W = np.array(W); HT = np.array(HT)
        a0 = np.degrees(np.arctan2(V, H)) % 360

        def boxes(a, mx=0., my=0.):
            lx, ly = RL * np.cos(np.deg2rad(a)), RL * np.sin(np.deg2rad(a)); out = []
            for x, y, w, h in zip(lx, ly, W, HT):
                x0 = x + XYZ_PAD if x >= 0 else x - XYZ_PAD - w
                out.append((x0 - mx / 2, x0 + w + mx / 2, y - h * 0.62 - my / 2, y + h * 0.62 + my / 2))
            return out, lx, ly
        a = a0.copy()
        for it in range(40000):
            bx, _, _ = boxes(a, MX, MY); moved = False
            for i in range(len(a)):
                for j in range(i + 1, len(a)):
                    A, B = bx[i], bx[j]
                    if A[0] < B[1] and B[0] < A[1] and A[2] < B[3] and B[2] < A[3]:
                        s = 1.0 if (((a[j] - a[i] + 180) % 360) - 180) >= 0 else -1.0
                        a[i] = (a[i] - 0.15 * s) % 360; a[j] = (a[j] + 0.15 * s) % 360; moved = True
            if not moved:
                break
        bx0, LX, LY = boxes(a)
        clash = [(names[i], names[j]) for i in range(len(a)) for j in range(i + 1, len(a))
                 if bx0[i][0] < bx0[j][1] and bx0[j][0] < bx0[i][1]
                 and bx0[i][2] < bx0[j][3] and bx0[j][2] < bx0[i][3]]
        return LX, LY, clash

    rmax, rlon, rl_, lim_ = P['rmax'], P['rlon'], P['rl'], P['lim']
    im = ax.imshow(P['A'], origin='lower', extent=P['ext'], cmap=cmap, norm=norm,
                   interpolation='nearest')
    rrings = (200, 400) if rmax > 500 else (100, 200, 300)
    for r in rrings:
        if r >= rmax:
            continue
        ax.plot(r * np.cos(th), r * np.sin(th), color='white', lw=0.4, alpha=0.22, zorder=3)
        ax.text(r * np.cos(np.deg2rad(20)), r * np.sin(np.deg2rad(20)), f'{r}', color='black',
                fontsize=6.5, ha='center', va='center', zorder=4,
                path_effects=[pe.withStroke(linewidth=1.8, foreground='white')])
    ax.plot(rmax * np.cos(th), rmax * np.sin(th), color='0.35', lw=0.8, alpha=0.9, zorder=3)
    if P['spokes']:
        for ll in range(0, 360, 30):
            aa = np.deg2rad(90 + ll)
            ax.plot([0, rmax * np.cos(aa)], [0, rmax * np.sin(aa)], color='white', lw=0.25,
                    alpha=0.12, zorder=3)
            ax.text(rlon * np.cos(aa), rlon * np.sin(aa), f'{ll}°', color='black', fontsize=7,
                    ha='center', va='center', clip_on=False)
    if P['mid']:
        ax.plot([-rmax, rmax], [0, 0], color='white', lw=0.35, alpha=0.30, zorder=3)
        ax.text(rmax * 0.78, 16, 'Z = 0  (midplane)', color='black', fontsize=7,
                ha='center', va='bottom', zorder=4,
                path_effects=[pe.withStroke(linewidth=1.8, foreground='white')])
    for h, v, uh, uv in zip(P['H'], P['V'], P['dH'], P['dV']):
        nn = np.hypot(uh, uv)
        if nn < 1e-6:                      # spine points straight along the view axis
            ax.plot(h, v, 'o', color='#8fecff', ms=4, mec='black', mew=0.6, zorder=Z_SPINE)
            continue
        uh, uv = uh / nn, uv / nn
        ax.plot([h - XYZ_SPINE_L * uh, h + XYZ_SPINE_L * uh], [v - XYZ_SPINE_L * uv, v + XYZ_SPINE_L * uv], color='#8fecff', lw=2.1,
                solid_capstyle='round', zorder=Z_SPINE,
                path_effects=[pe.withStroke(linewidth=3.6, foreground='black', alpha=0.6)])
    for c in (range(len(names)) if cloud_contours else ()):
        Acol = np.nan_to_num(P['col'][c].astype(float) / AV, nan=0.0)
        if Acol.max() >= 1.0:
            ax.contour(P['cH'], P['cV'], Acol, levels=[1.0], colors='black', linewidths=1.1,
                       zorder=Z_AV)
        Amx = P['acc'][c].astype(float)
        if np.isfinite(Amx).any():
            ax.contour(P['cH'], P['cV'], np.nan_to_num(Amx, nan=0.0), levels=[NLEV],
                       colors='#39e639', linewidths=0.8, zorder=Z_N5)
    LX, LY, clash = solve_labels(ax, P['H'], P['V'], rl_)
    print('%s label overlaps: %s' % (P['ttl'][:2], clash or 'none'))
    for h, v, lx, ly, nm_ in zip(P['H'], P['V'], LX, LY, names):
        ax.plot([h, lx], [v, ly], color='#1d6f82', lw=0.5, alpha=0.85, zorder=5)
        ax.text(lx + (XYZ_PAD if lx >= 0 else -XYZ_PAD), ly, nm_, fontsize=FS, color='black',
                ha='left' if lx >= 0 else 'right', va='center', zorder=7,
                path_effects=[pe.withStroke(linewidth=2.6, foreground='white')])
    ax.plot(0, 0, marker='$\\odot$', color='white', ms=10, zorder=8)
    gcax = P['gc_axis']
    if gcax == 'y':
        gc_y0, gc_y1, gc_txt = rmax * 0.68, rmax * 0.90, rmax * 0.927
        ax.annotate('', xy=(0, gc_y1), xytext=(0, gc_y0),
                    arrowprops=dict(arrowstyle='-|>', color='white', lw=1.1), zorder=8)
        ax.text(0, gc_txt, 'to GC', color='black', fontsize=8.5, ha='center', va='bottom',
                zorder=8, path_effects=[pe.withStroke(linewidth=2.4, foreground='white')])
    elif gcax == 'x':
        gc_v = rmax * 0.75; gc_x0, gc_x1 = -rmax * 0.09, rmax * 0.09
        ax.annotate('', xy=(gc_x1, gc_v), xytext=(gc_x0, gc_v),
                    arrowprops=dict(arrowstyle='-|>', color='white', lw=1.1), zorder=8)
        ax.text(0, gc_v + rmax * 0.035, 'to GC', color='black', fontsize=8.5, ha='center',
                va='bottom', zorder=8,
                path_effects=[pe.withStroke(linewidth=2.4, foreground='white')])
    else:
        ax.text(0, rmax * 0.927, 'GC into the page', color='black', fontsize=8.5, ha='center',
                va='bottom', zorder=8,
                path_effects=[pe.withStroke(linewidth=2.4, foreground='white')])
    ax.set_xlabel(P['xlab']); ax.set_ylabel(P['ylab']); ax.set_title(P['ttl'], fontsize=11, pad=8)
    ax.set_xlim(-lim_, lim_); ax.set_ylim(-lim_, lim_); ax.set_aspect(1); ax.set_facecolor('white')
    return im


# ============================================================================
# 1 x 3 spine-inclination histograms
#   (from plot_inclination_panels.py)
# ============================================================================
# 1 x 3 summary of how the fitted spines are inclined, collecting three of the
# histograms from the projection figures:
#
#   a  inclination to the plane of the sky (l-b)   -- plot_sky_projections, panel 1
#   b  inclination to the Galactic plane (X-Y)     -- plot_xyz_projections, panel 1
#   c  inclination to the b-d plane                -- plot_sky_projections, panel 3
#
# Same angles, bins and conventions as those scripts (their out_of_plane functions
# are imported, not re-derived): 0 deg = spine lies in the plane, 90 = perpendicular.
# Bars: the mean-map spine of each cloud. Line: every posterior-sample spine,
# counts divided by the number of samples. Dashed: isotropic expectation.
#
# Run through dust3d_pipeline.py --steps inclinations.

def inclination_angles(f):
    """(sky, galactic, b-d) inclination in deg for one spine-fit record."""
    lbd = np.array(xyz_2_sph(f['start'])).ravel()
    sky = sky_out_of_plane(f['p1'], f['p2'], lbd[0], lbd[1])      # l-b, l-d, b-d
    gal = xyz_out_of_plane(unit_fig(f['p1'], f['p2']))         # X-Y, Y-Z, X-Z
    return np.array([sky[0], gal[0], sky[2]])


# ============================================================================
# [10-14 experiment] peak-to-saddle distance windows
#   (from plot_saddle_windows.py)
# ============================================================================
# A dendrogram-free distance window, compared with the current one.
#
# New window, per cloud, on the mass-per-slice profile M(d) of the MEAN map summed
# over the unpadded catalog box (mass_profile, Msun/pc):
#   1. start at the slice nearest the catalog d_mean;
#   2. hill-climb to the local maximum of M(d) that d_mean sits on -- "the
#      structure associated with d_mean";
#   3. from that peak, step outward one 1 pc slice at a time in both directions
#      while M(d) keeps falling (or stays flat); stop at the first slice where it
#      would rise again, i.e. at the saddle on each side.
#   4. floor: a side also stops once M(d) drops below --floor x M(peak), so a
#      monotonic flank (no saddle before the grid edge) ends where the cloud has
#      faded instead of running to 69 pc;
#   5. balance: if one side's stopping level is more than --tol times the other's,
#      the HIGHER side keeps going -- over the next bump to the next saddle (or
#      floor crossing) -- and the comparison repeats, until the two levels agree to
#      within --tol, a side hits the grid edge, or a side has gone --max-ext pc
#      past its first saddle.
# The window is the final saddle-to-saddle range. No dendrogram, no padding.
#
# Drawn against the current pipeline's windows (sample_summary):
#   grey dotted  the dendrogram leaf select_leaf picks
#   red dashed   the adopted n_iso window (win_lo/win_hi from --csv): the run of
#                slices whose padded sky box reaches n_iso, grown from the densest
#                slice in the leaf, +-10 pc
#
# Run through dust3d_pipeline.py --steps saddle.

C_NEW, C_OLD, C_LEAF = '#0571b0', 'tab:red', '0.45'


def climb(prof, k):
    """Index of the local maximum reached by stepping uphill from k."""
    while True:
        nb = [j for j in (k - 1, k + 1) if 0 <= j < len(prof)]
        j = max(nb, key=lambda j: prof[j])
        if prof[j] > prof[k]:
            k = j
        else:
            return k


def to_saddle(prof, k, step):
    """Step from k in direction step (+-1) while prof does not rise."""
    j = k
    while 0 <= j + step < len(prof) and prof[j + step] <= prof[j]:
        j += step
    return j


def to_stop(prof, k, step, floor_val):
    """Like to_saddle, but also stop at the first slice below floor_val."""
    j = k
    while 0 <= j + step < len(prof) and prof[j + step] <= prof[j]:
        j += step
        if prof[j] < floor_val:
            break
    return j


def next_saddle(prof, k, step, floor_val):
    """From a saddle at k, climb the next bump in direction step, then descend
    to the following saddle (or floor crossing). None if the edge comes first."""
    j = k
    while 0 <= j + step < len(prof) and prof[j + step] > prof[j]:
        j += step
    if not 0 <= j + step < len(prof):
        return None
    return to_stop(prof, j, step, floor_val)


def saddle_window(prof, d_mean, floor=0.0, tol=np.inf, max_ext=np.inf):
    """Returns k0, kpeak, first (lo, hi), balanced (lo, hi), and a note."""
    k0 = int(np.argmin(np.abs(ds - d_mean)))
    kp = climb(prof, k0)
    fv = floor * prof[kp]
    lo1, hi1 = to_stop(prof, kp, -1, fv), to_stop(prof, kp, +1, fv)
    lo, hi, note = lo1, hi1, ''
    edge = {-1: False, +1: False}
    for _ in range(100):
        a, b = prof[lo], prof[hi]
        if max(a, b) <= tol * min(a, b):
            break
        side = -1 if a > b else +1
        if edge[side]:
            note = f'{"near" if side < 0 else "far"} side cannot reach a matching level'
            break
        cur = lo if side < 0 else hi
        first = lo1 if side < 0 else hi1
        nxt = next_saddle(prof, cur, side, fv)
        if nxt is None or abs(ds[nxt] - ds[first]) > max_ext:
            edge[side] = True
            continue
        if side < 0:
            lo = nxt
        else:
            hi = nxt
    return k0, kp, (lo1, hi1), (lo, hi), note


def plot_saddle_one(ax, name, prof, d_mean, rec, with_labels=True):
    lo, hi = rec['new_lo'], rec['new_hi']
    ax.plot(ds, prof, color='k', lw=1.1)
    ax.set_yscale('log')
    ax.axvspan(lo, hi, color=C_NEW, alpha=0.12, lw=0)
    for x in (lo, hi):
        ax.axvline(x, color=C_NEW, lw=1.6)
    if rec['first_lo'] != lo or rec['first_hi'] != hi:
        for x in (rec['first_lo'], rec['first_hi']):
            ax.axvline(x, color=C_NEW, lw=1.0, ls=(0, (1, 1.5)))
    # the two final saddle levels, so their match (or not) is visible
    for x in (lo, hi):
        y = prof[int(x - ds[0])]
        ax.plot(x, y, marker='D', color=C_NEW, mec='white', ms=5.5, zorder=6)
    ax.axhline(prof[int(lo - ds[0])], color=C_NEW, lw=0.6, alpha=0.5)
    ax.axhline(prof[int(hi - ds[0])], color=C_NEW, lw=0.6, alpha=0.5)
    for x in (rec['old_lo'], rec['old_hi']):
        ax.axvline(x, color=C_OLD, lw=1.4, ls='--')
    if np.isfinite(rec['leaf_lo']):
        for x in (rec['leaf_lo'], rec['leaf_hi']):
            ax.axvline(x, color=C_LEAF, lw=1.0, ls=':')
    ax.axvline(d_mean, color='0.6', lw=0.9, ls=(0, (6, 3)))
    ax.plot(rec['peak'], prof[int(rec['peak'] - ds[0])], 'o', color=C_NEW, ms=5, zorder=5)

    xs = [lo, hi, rec['first_lo'], rec['first_hi'], rec['old_lo'], rec['old_hi'], d_mean]
    if np.isfinite(rec['leaf_lo']):
        xs += [rec['leaf_lo'], rec['leaf_hi']]
    span = max(xs) - min(xs)
    ax.set_xlim(max(ds[0], min(xs) - 0.4 * span), min(ds[-1], max(xs) + 0.4 * span))
    sel = (ds >= ax.get_xlim()[0]) & (ds <= ax.get_xlim()[1])
    ax.set_ylim(np.nanmin(prof[sel]) * 0.7, np.nanmax(prof[sel]) * 1.6)

    flag = '' if abs(rec['peak'] - d_mean) <= 30 else f'  [peak {rec["peak"]:.0f} pc, {rec["peak"] - d_mean:+.0f} from d_mean]'
    ratio = f"  saddles {rec['ratio']:.1f}x" + ('' if not rec['note'] else ' (!)')
    ax.set_title(f'{name}{flag}{ratio}', fontsize=10)
    if with_labels:
        ax.set_xlabel('heliocentric distance (pc)')
        ax.set_ylabel(r'mass per slice ($M_\odot\,\mathrm{pc}^{-1}$)')


SADDLE_LEGEND = [Patch(facecolor=C_NEW, alpha=0.25, edgecolor=C_NEW, label='new: balanced saddle window'),
          Line2D([], [], color=C_NEW, ls=(0, (1, 1.5)), lw=1.0, label='new: first saddles (before balancing)'),
          Line2D([], [], color=C_NEW, marker='D', mec='white', ls='', ms=5.5, label='saddle levels compared'),
          Line2D([], [], color=C_OLD, ls='--', lw=1.4, label=r'old: adopted $n_{iso}$ window'),
          Line2D([], [], color=C_LEAF, ls=':', lw=1.0, label='old: dendrogram leaf'),
          Line2D([], [], color='0.6', ls=(0, (6, 3)), lw=0.9, label=r'catalog $d_{mean}$'),
          Line2D([], [], color=C_NEW, marker='o', ls='', ms=5, label='peak climbed to')]


# ============================================================================
# [10-14 experiment] A_V = 1 mag aperture on the mass-per-slice profile
#   (from plot_av1_aperture.py)
# ============================================================================
# Mass-per-slice with an A_V = 1 mag aperture instead of the whole catalog box.
#
# Aperture, per cloud: the pixels of the unpadded catalog box where the Edenhofer
# column summed over the cloud's CURRENT distance window (win_lo..win_hi from
# --csv) reaches A_V >= 1 mag, A_V = sum_d n_H * 1 pc / 2.2e21.
#
# Then, exactly as sample_summary does it but on the masked profile:
#   leaf    select_leaf on the masked M(d) (robust floor, distance-scaled min_npix)
#   window  grown from the densest slice of the leaf while the max n_H over the
#           catalog box + lb_pad (1 deg) stays >= n_iso, search capped at d_search
#           (400 pc) a side, then +-d_edge_pad (10 pc).
# The window step is reproduced here (not imported), so it is checked first:
# run on the BOX aperture it must give back win_lo/win_hi from --csv.
#
# Note the circularity: the aperture is built from the old window, and the new
# window from the aperture. This is one pass, not iterated to convergence.
#
# Run through dust3d_pipeline.py --steps aperture.

C_BOX, C_AP, C_OLD = '0.55', '#0571b0', 'tab:red'


def grow_window(i, table1, leaf_mask, mproxy, n_iso, lb_pad=1.0, d_edge_pad=10.0, d_search=400.0):
    """sample_summary's distance window (choosing_distances_from_healpix.py:769-794)."""
    idx = np.flatnonzero(leaf_mask)
    lo_i, hi_i = int(idx[0]), int(idx[-1])
    cap = int(round(d_search))
    lo_s, hi_s = max(0, lo_i - cap), min(len(leaf_mask) - 1, hi_i + cap)
    m_s = np.zeros_like(leaf_mask); m_s[lo_s:hi_s + 1] = True
    big, _ = get_cloud_info(i, mproxy, table1, 'full', dmask=m_s, pad=lb_pad)
    nmax = np.nanmax(big, axis=(1, 2))
    lo_c, hi_c = lo_i, hi_i
    inside = nmax[lo_i - lo_s:hi_i - lo_s + 1]
    if np.nanmax(inside) >= n_iso:
        k = int(np.nanargmax(inside)) + (lo_i - lo_s)
        a_, b_ = k, k
        while a_ - 1 >= 0 and nmax[a_ - 1] >= n_iso:
            a_ -= 1
        while b_ + 1 < len(nmax) and nmax[b_ + 1] >= n_iso:
            b_ += 1
        lo_c, hi_c = a_ + lo_s, b_ + lo_s
    pad_i = int(round(d_edge_pad))
    lo_d, hi_d = max(0, lo_c - pad_i), min(len(leaf_mask) - 1, hi_c + pad_i)
    return ds[lo_d], ds[hi_d]


def masked_profile(cube, mask):
    return mass_profile(np.where(mask[None, :, :], cube, 0.0))


def aperture_plot_cloud(axm, axp, row, av, mask, prof_box, prof_ap, rec):
    lc, bc, _, _ = box_grid(row)
    ext = [lc[0] + CD / 2, lc[-1] - CD / 2, bc[0] - CD / 2, bc[-1] + CD / 2]
    pos = av[av > 0]
    axm.imshow(av, origin='lower', extent=ext, cmap='gray_r', aspect='equal',
               norm=LogNorm(vmin=max(0.02, np.nanpercentile(pos, 1)), vmax=np.nanpercentile(pos, 99.5)))
    if mask.any() and not mask.all():
        axm.contour(lc, bc, mask.astype(float), levels=[0.5], colors=C_AP, linewidths=1.3)
    axm.set_title(f"{row['Name']}: $A_V$ over {rec['old_lo']:.0f}-{rec['old_hi']:.0f} pc\n"
                  f"aperture = {100 * rec['ap_frac']:.0f}% of box", fontsize=9)
    axm.tick_params(labelsize=7)

    axp.plot(ds, prof_box, color=C_BOX, lw=1.1)
    axp.plot(ds, prof_ap, color=C_AP, lw=1.3)
    axp.set_yscale('log')
    for x in (rec['old_lo'], rec['old_hi']):
        axp.axvline(x, color=C_OLD, ls='--', lw=1.3)
    for x in (rec['new_lo'], rec['new_hi']):
        axp.axvline(x, color=C_AP, ls='-', lw=1.6)
    axp.axvline(rec['d_mean'], color='0.6', ls=(0, (6, 3)), lw=0.9)
    xs = [rec['old_lo'], rec['old_hi'], rec['new_lo'], rec['new_hi']]
    span = max(xs) - min(xs)
    axp.set_xlim(max(ds[0], min(xs) - 0.6 * span), min(ds[-1], max(xs) + 0.6 * span))
    sel = (ds >= axp.get_xlim()[0]) & (ds <= axp.get_xlim()[1])
    top = np.nanmax(prof_box[sel])
    pos = prof_ap[sel][prof_ap[sel] > 0]
    lo_y = max(pos.min(), top * 1e-4) if pos.size else top * 1e-3   # empty aperture: box only
    axp.set_ylim(lo_y * 0.7, top * 1.6)
    axp.set_title(f"window {rec['old_lo']:.0f}-{rec['old_hi']:.0f} -> {rec['new_lo']:.0f}-{rec['new_hi']:.0f} pc;"
                  f"  M(window) {rec['M_ap'] / rec['M_box']:.2f}x box", fontsize=9)
    axp.tick_params(labelsize=7)


APERTURE_LEGEND = [Line2D([], [], color=C_BOX, lw=1.1, label='M(d), whole catalog box'),
          Line2D([], [], color=C_AP, lw=1.3, label=r'M(d), $A_V \geq 1$ aperture'),
          Line2D([], [], color=C_OLD, ls='--', lw=1.3, label='current window (box aperture)'),
          Line2D([], [], color=C_AP, lw=1.6, label='window from aperture profile'),
          Line2D([], [], color='0.6', ls=(0, (6, 3)), lw=0.9, label=r'catalog $d_{mean}$')]


# ============================================================================
# [10-14 experiment] leak test on the projected n_H >= threshold region
#   (from plot_leak_niso.py)
# ============================================================================
# Leak test on the Edenhofer cube: for each cloud and each threshold n_thr, how much
# of the cloud's projected n_H >= n_thr region lies OUTSIDE its catalog box.
#
# Map, per cloud: P(l, b) = max over d of n_H(l, b, d), the max taken over the
# cloud's distance window (win_lo..win_hi from --csv, the same window for every
# threshold), on the stored PAD_STORED (8 deg) padded mean cube, so a region can
# spill out of the catalog box and still be counted.
#
# For each threshold:
#   1. label the connected regions of P >= n_thr (8-connectivity);
#   2. seed = the region with the largest sum of P over its in-box pixels;
#   3. leak = 1 - (seed pixels inside the box) / (all seed pixels);
#   4. closed = the seed region does not touch the edge of the padded cube. If it
#      does, the outside part is truncated at 8 deg and the leak is a LOWER bound.
# Pixel counts, not solid angle (cos b weighting not applied).
#
# Run through dust3d_pipeline.py --steps leak.

S8 = np.ones((3, 3), dtype=bool)


LEAK_COLORS = ['#e66101', '#0571b0', '#1b9e77', '#7570b3']


def leak_one(P, cat, thr):
    m = np.nan_to_num(P, nan=-np.inf) >= thr
    lab, k = ndi.label(m, S8)
    if k == 0:
        return None
    ids = np.arange(1, k + 1)
    insum = ndi.sum(np.where(cat, np.nan_to_num(P), 0.0), lab, ids)
    if insum.max() <= 0:
        return None
    c = lab == ids[int(np.argmax(insum))]
    edge = np.zeros_like(c)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    edge |= ndi.binary_dilation(~np.isfinite(P), S8)
    return dict(region=c, leak=float(1 - (c & cat).sum() / c.sum()),
                closed=not bool((c & edge).any()),
                box_cover=float((c & cat).sum() / cat.sum()),
                area_deg2=float(c.sum() * CD * CD))


# ============================================================================
# [10-13/14 experiment] Edenhofer vs NICEST (or projected n_iso) A_V contours
#   (from plot_av1_contours.py)
# ============================================================================
# Zeroth-moment (A_V) maps of each cloud with two A_V = 1 mag contours overplotted:
#
#     orange  Edenhofer+2024 dust cube, summed over the FULL distance axis (69-1250 pc)
#     blue    --blue nicest: NICEST A_V = 1 mag (iNICEST, 0.229 deg effective resolution)
#             --blue niso:   projected n_H = --n-iso cm^-3 isocontour, i.e. the contour of
#                            max_d n_H(l, b, d), the max taken over the cloud's fitted
#                            distance window (win_lo..win_hi from --csv) or, with
#                            --niso-range full, over all 69-1250 pc
#
# Edenhofer column: N_H = sum_d n_H(l,b,d) * 1 pc, A_V = N_H / 2.2e21 cm^-2 (same
# convention as plot_xyz/plot_sky_projections).
# NICEST: the maps are A_J (mag); A_V = 8.9 * A_J, as in load_spectral_and_nicest.
# The NICEST contour is drawn on NICEST's own 0.1145 deg CAR grid (no regridding);
# both grids are linear in l/b, so the two sit on the same axes directly.
#
# Run through dust3d_pipeline.py --steps contours.

AV_PER_AJ = 8.9


C_EDEN, C_NIC = '#e66101', '#0571b0'


def nicest_av(name):
    """(l, b, A_V) of one NICEST map; l, b are 2D pixel-centre arrays in deg."""
    path = f'{NICEST_DIR}/{ALIASES.get(name, name)}_nicest.fits'
    if not os.path.exists(path):
        return None
    hdu = fits.open(path)[0]
    ny, nx = hdu.data.shape
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    l, b = WCS(hdu.header).celestial.all_pix2world(xx, yy, 0)
    return l, b, np.asarray(hdu.data, float) * AV_PER_AJ


def edenhofer_av(cube):
    """Full-distance column, A_V mag. cube is (nd, nb, nl) in cm^-3, 1 pc slices."""
    return np.nansum(cube, axis=0) * PC_CM / NH_PER_AV


def niso_projection(cube, window=None):
    """max_d n_H over the distance slices in window=(lo, hi) pc, or all of them."""
    if window is not None:
        sel = (ds >= window[0]) & (ds <= window[1])
        cube = cube[sel]
    return np.nanmax(cube, axis=0)


def contours_plot_cloud(ax, row, cube, with_labels=True, blue="nicest", n_iso=N_ISO, window=None):
    lc, bc, _, _ = box_grid(row)
    av = edenhofer_av(cube)
    ext = [lc[0] + CD / 2, lc[-1] - CD / 2, bc[0] - CD / 2, bc[-1] + CD / 2]
    vmax = np.nanpercentile(av, 99.5)
    im = ax.imshow(av, origin='lower', extent=ext, cmap='gray_r', aspect='equal',
                   norm=LogNorm(vmin=max(0.05, np.nanpercentile(av, 1)), vmax=vmax))
    stats = {'frac_eden': float(np.nanmean(av >= 1))}
    if av.max() >= 1:
        ax.contour(lc, bc, av, levels=[1.0], colors=C_EDEN, linewidths=1.4)

    if blue == 'niso':
        nmax = niso_projection(cube, window)
        if np.nanmax(nmax) >= n_iso:
            ax.contour(lc, bc, nmax, levels=[n_iso], colors=C_NIC, linewidths=1.4)
        stats['frac_niso'] = float(np.nanmean(nmax >= n_iso))
        nic = None
    else:
        nic = nicest_av(row['Name'])
    if blue == 'niso':
        pass
    elif nic is not None:
        l, b, a = nic
        lmid = 0.5 * (lc[0] + lc[-1])
        l = lmid + ((l - lmid + 180.0) % 360.0 - 180.0)    # same l branch as the cube
        keep = (l.min() <= lc[0] + 1) & (l.max() >= lc[-1] - 1)
        if keep and np.nanmax(a) >= 1:
            ax.contour(l, b, a, levels=[1.0], colors=C_NIC, linewidths=1.4)
        # fraction of the catalog box above 1 mag, on NICEST's own pixels
        inbox = (l <= lc[0] + CD / 2) & (l >= lc[-1] - CD / 2) & \
                (b >= bc[0] - CD / 2) & (b <= bc[-1] + CD / 2)
        stats['frac_nicest'] = float(np.nanmean(a[inbox] >= 1)) if inbox.any() else np.nan
    else:
        stats['frac_nicest'] = np.nan

    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_title(f"{row['Name']}  (d = {row['d_mean(pc)']:.0f} pc)", fontsize=10)
    if with_labels:
        ax.set_xlabel('l (deg)')
        ax.set_ylabel('b (deg)')
    return im, stats


def contours_legend(blue, n_iso, niso_range):
    eden = Line2D([], [], color=C_EDEN, lw=1.6, label=r'Edenhofer $A_V$ = 1 mag (69-1250 pc)')
    if blue == 'niso':
        rng = 'fitted window' if niso_range == 'window' else '69-1250 pc'
        lab = rf'max$_d\ n_H$ = {n_iso:g} cm$^{{-3}}$ ({rng})'
    else:
        lab = r'NICEST $A_V$ = 1 mag'
    return [eden, Line2D([], [], color=C_NIC, lw=1.6, label=lab)]


# ============================================================================
# Steps: the scripts' main() functions, with their command-line options as
# arguments. dust3d_pipeline.py calls these.
# ============================================================================
def _dir(path):
    path = path.rstrip('/') + '/'
    os.makedirs(path, exist_ok=True)
    return path


def _sample_profiles(table1):
    """Mass-per-slice profiles of the independent posterior samples, per cloud."""
    allp, indep = build_sample_profiles(table1)
    return allp, indep, [pr[indep] for pr in allp]


# ---- cubes -----------------------------------------------------------------
def extract_padded_sample_cubes(outdir=CUBE_DIR, pad=PAD_STORED, d_max=D_MAX, catalog=CAT,
                                clouds=None):
    """(from extract_padded_sample_cubes.py)

    Extract one padded (distance, b, l) cube per cloud, for EVERY posterior
    sample, straight from the native Edenhofer+2023 samples release, in ONE pass
    over the 19.5 GB samples file for all clouds at once.

    Boxes: the l/b bounds in `catalog` grown by `pad` degrees on all four sides.
    Distance: the full 1 pc axis (69-1250 pc, 1182 slices). Writes
    <outdir>/<cloud>.npz with
        cube   (nsamp, ND, nb, nl) float32, cm^-3
        indep  indices of one sample per antithetic pair
        xo, yo pixel offset of the box on the full-sky grid (as box_grid returns)
        lc, bc pixel-centre l and b;  ds  distance of each slice
        row    the catalog row (l_max, l_min, b_min, b_max, v_min, v_max, d_mean), unpadded
        pad    degrees of padding
    """
    cat = pd.read_csv(catalog)
    cat = cat[cat['d_mean(pc)'] < d_max].reset_index(drop=True)
    if clouds:
        cat = cat[cat['Name'].isin(clouds)].reset_index(drop=True)
    print(f'{len(cat)} clouds (d_mean < {d_max:g} pc), pad {pad:g} deg')

    grids, weights, sizes = [], [], []
    for _, row in cat.iterrows():
        lc, bc, _, (xo, yo) = box_grid(row, pad=pad)
        LL, BB = np.meshgrid(lc, bc)
        idx, wgt = bilinear_interpolation_weights((LL.ravel() % 360.0) * u.deg,
                                                  BB.ravel() * u.deg,
                                                  nside=NSIDE, order='nested')
        grids.append((lc, bc, xo, yo, LL.shape))
        weights.append((idx.astype(np.int64), wgt.astype(np.float32)))
        sizes.append(LL.size)
        print(f'  {row["Name"]:18s} {LL.shape[0]:4d} x {LL.shape[1]:4d} px')

    h = fits.open(DATA + SAMPLES, memmap=True)
    rc = h[2].data['radial pixel centers'].astype(float)
    nsamp = h[1].shape[0]
    gb = sum(sizes) * ND * nsamp * 4 / 1e9
    print(f'{nsamp} samples x {rc.size} shells; output {gb:.1f} GB in total')

    stacks = [np.empty((nsamp, rc.size, n), dtype=np.float32) for n in sizes]
    pairs = None
    for k in range(rc.size):
        shell = np.asarray(h[1].data[:, k, :])        # (nsamp, npix), read once
        if pairs is None:
            pairs = antithetic_pairs(shell)
        for st, (idx, wgt) in zip(stacks, weights):
            st[:, k, :] = (shell[:, idx] * wgt).sum(axis=1)
        if k % 50 == 0:
            print(f'  shell {k}/{rc.size}', flush=True)
    h.close()
    indep = np.array([a for a, b in pairs])
    print('independent samples:', indep.tolist())

    os.makedirs(outdir, exist_ok=True)
    lo_r, lo_d = np.log(rc), np.log(ds)
    for (_, row), st, (lc, bc, xo, yo, shape) in zip(cat.iterrows(), stacks, grids):
        cube = np.empty((nsamp, ND) + shape, dtype=np.float32)
        for s in range(nsamp):
            flat = np.empty((ND, st.shape[2]), dtype=np.float32)
            for j in range(st.shape[2]):
                flat[:, j] = np.interp(lo_d, lo_r, st[s, :, j])
            cube[s] = (flat * K_E_TO_N).reshape(ND, *shape)
        out = os.path.join(outdir, f'{row["Name"]}.npz')
        np.savez(out, cube=cube, indep=indep, xo=xo, yo=yo, lc=lc, bc=bc, ds=ds,
                 pad=pad, row=row[['l_max(deg)', 'l_min(deg)', 'b_min(deg)',
                                   'b_max(deg)', 'v_min(km/s)', 'v_max(km/s)',
                                   'd_mean(pc)']].to_numpy().astype(float))
        print(f'saved {out}  {cube.shape}  {cube.nbytes / 1e6:.0f} MB', flush=True)


# ---- windows + Plummer fits (the n_iso table) -------------------------------
def sample_summary_table(table1, cubes, sprofs, outdir, clouds=None, n_iso=N_ISO,
                         min_value=None, save_figures=True, spine_slab_cut=True):
    """(from choosing_distances_from_healpix.py --sample-summary)

    Runs sample_summary for each cloud and writes
    <outdir>/spine_angles_and_fits_samples_n<n_iso>.csv (FIT_COLS): the adopted
    distance window, window mass, spine point + angles and the per-sample
    Plummer fits. save_figures also writes <cloud>_projmaps_profiles_samples_n<N>.png.
    """
    global FIG_LOC
    FIG_LOC = _dir(outdir)
    want = list(table1['Name']) if not clouds else list(clouds)
    rows = []
    for cl in want:
        print(f'== {cl}')
        _, _, st = sample_summary(table1, cl, cubes, sprofs, min_value=min_value,
                                  n_iso=n_iso, save=save_figures,
                                  spine_slab_cut=spine_slab_cut,
                                  suffix=f'_projmaps_profiles_samples_n{n_iso:g}')
        rows.append(st)
        plt.close('all')
    df = pd.DataFrame(rows)[FIT_COLS]
    out = f'{FIG_LOC}spine_angles_and_fits_samples_n{n_iso:g}.csv'
    df.to_csv(out, index=False, float_format='%.4f')
    print(f'\n{df.to_string(index=False, float_format=lambda v: f"{v:.2f}")}')
    print(f'\nwrote {out}')
    return df


def add_masked_mass_columns(table1, csv_path, n_thresh, col):
    """(from add_masked_mass.py) Add <col>, <col>_sd: per-sample mass of the
    voxels with n >= n_thresh inside each cloud's window, mean +- sd over the
    independent samples."""
    df = pd.read_csv(csv_path)
    means, sds = [], []
    t0 = time.time()
    for _, row in df.iterrows():
        m, sd = masked_mass_per_cloud(table1, row.cloud, row.win_lo, row.win_hi, n_thresh)
        means.append(m); sds.append(sd)
        print(f'  {col} {row.cloud:18s} {m:10.1f} +- {sd:8.1f}  (mean-map M={row.M:10.1f})',
              flush=True)
    df[col] = means
    df[f'{col}_sd'] = sds
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f'wrote {csv_path}  ({time.time() - t0:.0f}s total)')
    return df


# ---- cylinder figures, spine windows ----------------------------------------
def cylinder_figures(table1, win_csv, outdir, clouds=None, n_iso=N_ISO, n_seeds=40,
                     lb_pad=1.0, lb_search_pad=PAD_STORED, refit=False, perp_nmin=0.0,
                     perp_fit='lin_dip', perp_slab='spine', layout='three', spine_cache=None,
                     per_sample=True):
    """(from plot_cylinder_slice.py)

    Per cloud: the 40-seed spines (mean map + each independent sample, cached in
    `spine_cache`, default <outdir>/_spine_fits), the cylinder profiles, and
    <cloud>_cylinder_three_row_n<N>.png (layout 'three') or
    <cloud>_cylinder_slice_n<N>.png ('two'). Updates the clouds' rows of
    <outdir>/radial_profile_fits_n<N>.csv (rho0, r0, alpha, r_dip of the
    figure's own fit, and M_cyl, the mass between the spine caps out to r_dip).
    per_sample: also run that chain on each independent posterior sample, with
    the sample's own spine and caps (cylinder_samples), adding <col>_sd and
    <col>_smean (spread and mean over the samples) for rho0, r0, alpha, r_dip and
    M_cyl, and writing every sample's values to
    <outdir>/radial_profile_fits_samples_n<N>.csv.
    """
    outdir = _dir(outdir)
    spine_cache = spine_cache or outdir + '_spine_fits'
    if layout == 'three':
        cubes = build_cubes(table1)
        allp, indep_p = build_sample_profiles(table1)
        names = list(table1['Name'])
    win = pd.read_csv(win_csv).set_index('cloud')
    rows, sample_rows = [], []
    for cloud in (clouds or list(table1['Name'])):
        print(f'== {cloud}')
        row = win.loc[cloud]
        lo, hi = float(row['win_lo']), float(row['win_hi'])
        fits_ = get_spine_fits(cloud, table1, lo, hi, n_iso, n_seeds, lb_search_pad,
                               cache_dir=spine_cache, refit=refit)
        out = build_cylinder_data(cloud, table1, lo, hi, n_iso, n_seeds, lb_pad, lb_search_pad,
                                  fits=fits_, spine_slab=(perp_slab == 'spine'),
                                  perp_nmin=perp_nmin, perp_fit=perp_fit)
        mass, tag = None, 'cylinder_slice'
        if layout == 'three':
            if list(fits_['indep']) != [int(k) for k in indep_p]:
                raise SystemExit(f'sample indices differ: spines {fits_["indep"]} vs '
                                 f'profiles {list(indep_p)}')
            i = names.index(cloud)
            dmean = float(table1.iloc[i]['d_mean(pc)'])
            mass = (mass_profile(cubes[i]), allp[i][indep_p], (lo, hi), dmean,
                    (fits_['mean']['d_lo'], fits_['mean']['d_hi']))
            tag = 'cylinder_three_row'
        dips = plot_cylinder_slice(cloud, *out, f'{outdir}{cloud}_{tag}_n{n_iso:g}.png',
                                   n_iso=n_iso, mass=mass)
        plt.close('all')
        rd, nd, fl = dips['profile']
        res = out[1]
        rho0_, r0_, al_ = res['fit']
        rec = {'cloud': cloud, 'slab': perp_slab, 'perp_nmin': perp_nmin,
               'perp_fit': perp_fit,
               'rho0': rho0_, 'r0': abs(r0_), 'alpha': al_,
               'r_dip': rd, 'n_dip': nd, 'dip_flag': fl, 'M_cyl': res['M_cyl']}
        print(f"  fit rho0 {rho0_:.1f} r0 {abs(r0_):.2f} alpha {al_:.2f} | "
              f"dip {rd:.1f} pc, n {nd:.2f} ({fl}) | M_cyl {res['M_cyl']:.0f} Msun")
        if np.isfinite(res['r_dip']) and abs(res['r_dip'] - rd) > 1e-6:
            print(f"  !! fit r_dip {res['r_dip']:.2f} != plotted r_dip {rd:.2f}")

        # the same chain on each posterior sample, with its own spine and caps
        if per_sample and perp_fit == 'lin_dip':
            smp = cylinder_samples(cloud, table1, lo, hi, fits_, lb_pad, lb_search_pad,
                                   spine_slab=(perp_slab == 'spine'), perp_nmin=perp_nmin,
                                   perp_fit=perp_fit)
            sdf = pd.DataFrame(smp)
            sdf.insert(0, 'cloud', cloud)
            sample_rows.append(sdf)
            rec['n_samp'] = len(sdf)
            for col in ('rho0', 'r0', 'alpha', 'r_dip', 'M_cyl'):
                v = sdf[col].to_numpy(float) if len(sdf) else np.array([])
                rec[col + '_sd'] = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
                rec[col + '_smean'] = float(np.mean(v)) if len(v) else np.nan
            for _, s in sdf.iterrows():
                print(f"    sample {s['sample']:2d}: rho0 {s['rho0']:6.1f} r0 {s['r0']:5.2f} "
                      f"alpha {s['alpha']:4.2f} | dip {s['r_dip']:5.1f} pc ({s['dip_flag']}) "
                      f"| M_cyl {s['M_cyl']:8.0f} | pad {s['pad']:.2f}")
        rows.append(rec)
    path = f'{outdir}radial_profile_fits_n{n_iso:g}.csv'
    if rows:
        # update the rows for the clouds just run; keep any others already there
        new = pd.DataFrame(rows)
        if os.path.exists(path):
            old = pd.read_csv(path)
            new = pd.concat([old[~old['cloud'].isin(new['cloud'])], new], ignore_index=True)
        new.to_csv(path, index=False)
        print(f'wrote {path}')
    if sample_rows:
        spath = f'{outdir}radial_profile_fits_samples_n{n_iso:g}.csv'
        new = pd.concat(sample_rows, ignore_index=True)
        if os.path.exists(spath):
            old = pd.read_csv(spath)
            new = pd.concat([old[~old['cloud'].isin(new['cloud'])], new], ignore_index=True)
        new.to_csv(spath, index=False)
        print(f'wrote {spath}')
    return path


def spine_window_figures(table1, win_csv, outdir, clouds=None, n_iso=N_ISO, n_seeds=40,
                         lb_search_pad=PAD_STORED, refit=False, spine_cache=None):
    """(from plot_sample_spine_windows.py) <cloud>_sample_spine_windows.png:
    mass spectra with the distance range each sample's spine covers."""
    outdir = _dir(outdir)
    spine_cache = spine_cache or outdir + '_spine_fits'
    cubes = build_cubes(table1)
    allp, indep_p = build_sample_profiles(table1)
    win = pd.read_csv(win_csv).set_index('cloud')
    names = list(table1['Name'])
    for cloud in (clouds or names):
        print(f'== {cloud}')
        i = names.index(cloud)
        row = win.loc[cloud]
        fits_ = get_spine_fits(cloud, table1, float(row['win_lo']), float(row['win_hi']),
                               n_iso, n_seeds, lb_search_pad, cache_dir=spine_cache, refit=refit)
        if list(fits_['indep']) != [int(k) for k in indep_p]:
            raise SystemExit(f'sample indices differ: spines {fits_["indep"]} vs '
                             f'profiles {list(indep_p)}')
        plot_sample_spine_windows(cloud, mass_profile(cubes[i]), allp[i][indep_p], fits_,
                                  f'{outdir}{cloud}_sample_spine_windows.png')
        plt.close('all')


# ---- the rho0 input table, masses, correlations, IF23 -----------------------
def spine_inclination(rec):
    """Inclination (deg) of one spine-fit record to the plane of the sky, at its
    walk start: sin i = |u . r_hat|. Same angle as sky_out_of_plane()[0] and
    spine_sky()'s cos i."""
    lbd = np.array(xyz_2_sph(rec['start'])).ravel()
    return float(sky_out_of_plane(rec['p1'], rec['p2'], lbd[0], lbd[1])[0])


def rho0_inputs_table(win_csv, radial_csv, fits_dir, out_csv, n_iso=N_ISO, n_seeds=40):
    """Per-cloud inputs of the rho0 correlation and IF23 figures, written to
    out_csv: rho0, r0, alpha, r_dip (and M_cyl and the sample spreads, when
    present) from the cylinder figures' fits (radial_profile_fits_n<N>.csv),
    M (window mass) from the n_iso table, and the
    mean spine's inclination to the sky (inc_deg, cos_i). av1_mass_columns()
    then adds the A_V-contour masses.

    (Added in the consolidation: on 2026-10-14 this table was assembled
    inline in the session, not by a script.)
    """
    rad = pd.read_csv(radial_csv).set_index('cloud')
    win = pd.read_csv(win_csv).set_index('cloud')
    rows = []
    for cl in win.index:
        if cl not in rad.index:
            continue
        f = pickle.load(open(f'{fits_dir}/{cl}_spine_fits_n{n_iso:g}_s{n_seeds}.pkl', 'rb'))
        inc = spine_inclination(f['mean'])
        rec = dict(cloud=cl, rho0=rad.loc[cl, 'rho0'], r0=rad.loc[cl, 'r0'],
                   alpha=rad.loc[cl, 'alpha'], r_dip=rad.loc[cl, 'r_dip'],
                   M=win.loc[cl, 'M'], inc_deg=inc, cos_i=np.cos(np.deg2rad(inc)))
        # the cylinder mass and the posterior-sample spreads, when the cylinder
        # step wrote them (cylinder_figures(per_sample=True))
        for col in ('M_cyl', 'rho0_sd', 'r0_sd', 'alpha_sd', 'r_dip_sd', 'M_cyl_sd', 'n_samp'):
            if col in rad.columns:
                rec[col] = rad.loc[cl, col]
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print('wrote', out_csv)
    return df


def av1_mass_columns(table1, win_csv, out_csv, av=1.0, av_rule='leak', leak_cut=0.3):
    """(from compute_av1_masses.py) Add M_av1, M_av1_sd, av1_frac, av_thr,
    leak_05 to out_csv: mass inside the A_V = 0.5 or 1 mag contour of the window
    column (threshold from the leak test, or `av` for every cloud with
    av_rule='fixed'), on the mean map and on each independent sample."""
    table1 = table1.reset_index(drop=True)
    win = pd.read_csv(win_csv).set_index('cloud')
    cubes = build_cubes(table1)
    out = pd.read_csv(out_csv).set_index('cloud')

    for i, row in table1.iterrows():
        name = row['Name']
        if name not in out.index:
            continue
        w = (ds >= win.loc[name, 'win_lo']) & (ds <= win.loc[name, 'win_hi'])
        if av_rule == 'leak':
            lk = leak_05(table1, row, w)
            thr = 0.5 if (np.isfinite(lk) and lk < leak_cut) else 1.0
        else:
            lk, thr = np.nan, av
        m_mean, frac = av1_mass(cubes[i], w, thr)

        st = load_stored(table1, name)
        ms = [av1_mass(_unpadded(st['cube'][k], row, st), w, thr)[0]
              for k in st['indep']]
        out.loc[name, 'M_av1'] = m_mean
        out.loc[name, 'M_av1_sd'] = float(np.std(ms, ddof=1))
        out.loc[name, 'av1_frac'] = frac
        out.loc[name, 'av_thr'] = thr
        out.loc[name, 'leak_05'] = lk
        print(f'{name:16s} leak {lk:4.2f} -> A_V >= {thr:g}  aperture {100 * frac:5.1f}% of box  '
              f'M = {m_mean:9.0f} '
              f'+- {out.loc[name, "M_av1_sd"]:7.0f} Msun   (window mass {out.loc[name, "M"]:9.0f})',
              flush=True)

    out.reset_index().to_csv(out_csv, index=False)
    print('wrote', out_csv)
    return out


def rho0_correlation_figures(fig_dir, csv, ifb_csv=IFB_DEFAULT, n_iso=10.0,
                             mass_col='M_cyl', tag=''):
    """(from plot_rho0_correlations.py)

    rho0_correlations_n<N><tag>.png from `csv` (cloud, rho0, r0, alpha, cos_i,
    <mass_col> ... columns): ours vs IF23, x = rho0 / cos i, each set with its
    own power-law fit. Returns the fits (see plot_single).
    """
    fits = plot_single(csv, ifb_csv, _dir(fig_dir) + f'rho0_correlations_n{n_iso:g}{tag}.png',
                       n_iso=n_iso, mass_col=mass_col)
    plt.close('all')
    return fits


def if23_fig7_figure(inputs, fits_dir, out, tag=f'n{N_ISO:g}_s40', ifb=IFB_FIG7):
    """(from plot_if23_fig7.py) Our version of Imara & Forbes (2023) Fig. 7.

      left   M (inside the A_V = 0.5 or 1 mag contour, M_av1) vs r0   [IF23: M_1AV vs r0]
      right  filled: projected spine length L cos i, L = the walked spine length
             (t_neg + t_pos of the mean 40-seed spine)              [IF23: b_par,max]
             open:   r_dip                                          [IF23: b_perp,max]
    r0, r_dip, M_av1, cos_i come from `inputs` (rho0_inputs table); L and its
    spread over the posterior-sample spines from the _spine_fits pickles.
    """
    df = pd.read_csv(inputs).set_index('cloud')
    Lp, Lp_sd = [], []
    for c_ in df.index:
        f = pickle.load(open(f'{fits_dir}/{c_}_spine_fits_{tag}.pkl', 'rb'))
        L, ci = spine_sky(f['mean'])
        Lp.append(L * ci)
        Lp_sd.append(np.std([np.prod(spine_sky(s)) for s in f['samples']], ddof=1))
    df['L_proj'], df['L_proj_sd'] = Lp, Lp_sd

    ifb = pd.read_csv(ifb)
    B, R, MS = 'tab:blue', 'tab:red', 8
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=150, constrained_layout=True)
    # this work: blue circles; IF23 (digitized from their Fig. 7): red diamonds.
    # Right panel: filled = length along the spine, open = perpendicular extent.
    a1.errorbar(df['r0'], df['M_av1'], yerr=df['M_av1_sd'], fmt='o', ms=MS, color=B,
                mec='white', mew=0.6, ecolor=B, elinewidth=1.1, zorder=3)
    a1.plot(ifb['r0_pc'], ifb['M_1AV'], 'D', ms=MS, color=R, mec='white', mew=0.6, zorder=3)
    a2.errorbar(df['r0'], df['L_proj'], yerr=df['L_proj_sd'], fmt='o', ms=MS, color=B,
                mec='white', mew=0.6, ecolor=B, elinewidth=1.1, zorder=3)
    a2.plot(df['r0'], df['r_dip'], 'o', ms=MS, mfc='none', mec=B, mew=1.5, zorder=3)
    a2.plot(ifb['r0_pc'], ifb['b_par_max'], 'D', ms=MS, color=R, mec='white', mew=0.6, zorder=3)
    a2.plot(ifb['r0_pc'], ifb['b_perp_max'], 'D', ms=MS, mfc='none', mec=R, mew=1.5, zorder=3)

    a1.legend(handles=[
        Line2D([], [], marker='o', ls='', ms=MS, color=B, mec='white',
               label=r'this work ($A_V$ > 0.5 or 1 mag)'),
        Line2D([], [], marker='D', ls='', ms=MS, color=R, mec='white',
               label=r'Imara & Forbes 2023 ($A_V$ > 1 mag)')],
        loc='lower right', fontsize=8.5, framealpha=0.9)
    a2.legend(handles=[
        Line2D([], [], marker='o', ls='', ms=MS, color=B, mec='white',
               label=r'this work: $L\cos i$ (projected spine length)'),
        Line2D([], [], marker='o', ls='', ms=MS, mfc='none', mec=B, mew=1.5,
               label=r'this work: $r_{\rm dip}$'),
        Line2D([], [], marker='D', ls='', ms=MS, color=R, mec='white',
               label=r'IF23: $b_{\parallel,\rm max}$'),
        Line2D([], [], marker='D', ls='', ms=MS, mfc='none', mec=R, mew=1.5,
               label=r'IF23: $b_{\perp,\rm max}$')],
        loc='lower right', fontsize=8.5, framealpha=0.9)

    mt = mticker
    for ax in (a1, a2):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$r_0$ (pc)')
        ax.xaxis.set_major_locator(mt.FixedLocator([2, 3, 5, 10, 20, 30]))
        ax.xaxis.set_major_formatter(mt.FormatStrFormatter('%g'))
        ax.xaxis.set_minor_formatter(mt.NullFormatter())
        ax.grid(True, which='major', alpha=0.25)
    a1.xaxis.set_major_locator(mt.FixedLocator([0.1, 0.3, 1, 3, 10, 30]))
    a2.xaxis.set_major_locator(mt.FixedLocator([0.1, 0.3, 1, 3, 10, 30]))
    a2.yaxis.set_major_locator(mt.FixedLocator([2, 5, 10, 20, 50, 100, 200]))
    a2.yaxis.set_major_formatter(mt.FormatStrFormatter('%g'))
    a2.yaxis.set_minor_formatter(mt.NullFormatter())
    a1.set_ylabel(r'$M\ (M_\odot)$')
    a2.set_ylabel(r'size along / across the spine (pc)')

    m1 = slope(df['r0'], df['M_av1'])
    mL = slope(df['r0'], df['L_proj'])
    mR = slope(df['r0'], df['r_dip'])
    a1.text(0.03, 0.97, f'log-log slope {m1:.2f}', transform=a1.transAxes, va='top', fontsize=9)
    a2.text(0.03, 0.97, f'log-log slope: $L\\cos i$ {mL:.2f},  $r_{{\\rm dip}}$ {mR:.2f}',
            transform=a2.transAxes, va='top', fontsize=9)
    a1.set_title(r'mass vs $r_0$   [IF23: $M_{1A_V}$]', fontsize=10.5)
    a2.set_title(r'projected spine length and dip radius vs $r_0$   '
                 r'[IF23: $b_{\parallel,\rm max}$, $b_{\perp,\rm max}$]', fontsize=10.5)
    fig.suptitle(f'After Imara & Forbes (2023) Fig. 7  --  this work ({len(df)} clouds, '
                 r'$n_{\rm iso}$ = ' + f'{tag.split("_")[0][1:]}' + r' cm$^{-3}$) vs IF23 (8 clouds)',
                 fontsize=12)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)
    print(df[['r0', 'M_av1', 'L_proj', 'L_proj_sd', 'r_dip']].round(2).to_string())


def spine_inclination_figures(csv, out, scatter_out=None):
    """(from plot_spine_inclinations.py) Sky-plane tilt (Monte Carlo from
    inc_ld/inc_bd) and Galactic position angle per cloud; optional scatter."""
    plot_inclinations(csv, out)
    if scatter_out:
        plot_inclination_scatter(csv, scatter_out)
    plt.close('all')


# ---- the projection figures ------------------------------------------------
def _map_legend(n_iso, spine_label='fitted spine'):
    return [Line2D([], [], color='#39e639', lw=1.2, label=f'$n_H={n_iso:g}$ cm$^{{-3}}$'),
            Line2D([], [], color='black', lw=1.4, label='$A_V=1$ mag'),
            Line2D([], [], color='#4fd0e7', lw=2.0, label=spine_label)]


def _inclination_hists(hax, ang_mean, ang_samp, ns_, ncl, planes):
    """The bottom-row inclination histograms shared by the sky and xyz figures."""
    ctr = 0.5 * (BINS[:-1] + BINS[1:]); wid = np.diff(BINS)
    iso = ncl * (np.sin(np.deg2rad(BINS[1:])) - np.sin(np.deg2rad(BINS[:-1])))   # isotropic expectation
    ymax = 0
    for i, ax in enumerate(hax):
        hm, _ = np.histogram(ang_mean[:, i], BINS); hs, _ = np.histogram(ang_samp[:, i], BINS)
        ax.bar(ctr, hm, width=wid * 0.96, color='#4fd0e7', edgecolor='#1d6f82', lw=0.9,
               label='mean-map spines (%d clouds)' % ncl, zorder=2)
        ax.stairs(hs / ns_, BINS, color='black', lw=1.4,
                  label='per-sample spines (counts / %d)' % ns_, zorder=3)
        ax.stairs(iso, BINS, color='tab:orange', lw=1.2, ls='--', label='isotropic expectation', zorder=3)
        ax.set_xlim(0, 90); ax.set_xticks(BINS)
        ax.set_xlabel(f'angle out of the {planes[i]} plane  (°;  0 = in the plane, 90 = perpendicular)')
        ax.set_ylabel('number of clouds')
        ax.set_title(f'inclination to the {planes[i]} plane   (median {np.median(ang_mean[:, i]):.0f}°)',
                     fontsize=10.5)
        ax.grid(True, axis='y', alpha=0.25)
        ymax = max(ymax, hm.max(), (hs / ns_).max(), iso.max())
    for ax in hax:
        ax.set_ylim(0, ymax * 1.15)
    hax[0].legend(loc='upper left', fontsize=8, framealpha=0.9)


def sky_projection_figure(csv, fits_dir, outdir, n_iso=N_ISO, fits_tag=None, cache=None):
    """(from plot_sky_projections.py) edenhofer_sky_projections_inclination.png:
    l-b / l-d / b-d dust maps with each cloud's n_iso and A_V = 1 contours and
    spine, and the inclination histograms to each plane."""
    outdir = _dir(outdir)
    cache = cache or outdir + '_sky_maps/sky_projection_maps.pkl'
    S = sky_prepare(csv, fits_dir, cache, n_iso, fits_tag)
    names, ang_mean, ang_samp = S['names'], S['ang_mean'], S['ang_samp']
    ns_, ncl, norm = S['ns_'], S['ncl'], S['norm']

    fig = plt.figure(figsize=(30, 10.6))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 0.42, 0.028], height_ratios=[1, 0.62],
                  hspace=0.30, wspace=0.14)
    ax_lb = fig.add_subplot(gs[0, 0]); ax_ld = fig.add_subplot(gs[0, 1], sharex=ax_lb)
    ax_bd = fig.add_subplot(gs[0, 2], sharey=ax_lb)
    hax = [fig.add_subplot(gs[1, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    im = draw_maps(fig, S, norm, ax_lb, ax_ld, ax_bd)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('A$_V$  [mag]   (column along the projection axis)', labelpad=8, fontsize=10)
    ax_lb.legend(handles=_map_legend(n_iso), loc='lower left', fontsize=7.5, framealpha=0.9,
                 facecolor='white', edgecolor='0.45')

    _inclination_hists(hax, ang_mean, ang_samp, ns_, ncl, ['$\\ell$–$b$ (sky)', '$\\ell$–$d$', '$b$–$d$'])
    fig.canvas.draw()
    for a, h in zip((ax_lb, ax_ld, ax_bd), hax):
        p = a.get_position(); q = h.get_position()
        h.set_position([p.x0 + 0.03 * p.width, q.y0, p.width * 0.94, q.height])

    fig.suptitle('Edenhofer+2023 3D dust in sky coordinates  —  dust column in visual magnitudes,'
                 '  and the inclination of the fitted spines to each plane', fontsize=13, y=0.96)
    out = outdir + 'edenhofer_sky_projections_inclination.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', out)
    print('%-18s %s' % ('cloud', 'out-of-plane angle to l-b / l-d / b-d (deg), mean spine'))
    for nm, a in zip(names, ang_mean):
        print('%-18s %5.1f %5.1f %5.1f' % (nm, *a))


def xyz_projection_figure(csv, fits_dir, outdir, n_iso=N_ISO, fits_tag=None, cache=None):
    """(from plot_xyz_projections.py) edenhofer_xyz_topdown_mom0_inclination.png:
    face-on X-Y, Y-Z and edge-on X-Z dust maps with the clouds' contours and
    spines, and the inclination histograms to each plane."""
    outdir = _dir(outdir)
    cache = cache or outdir + '_xyz_maps/xyz_projection_maps.npz'
    S = xyz_prepare(csv, fits_dir, cache, n_iso, fits_tag)
    names, fits_, ang_mean, ang_samp = S['names'], S['fits_'], S['ang_mean'], S['ang_samp']
    norm = S['norm']

    fig = plt.figure(figsize=(29.5, 13.3))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.035], height_ratios=[1, 0.36],
                  hspace=0.22, wspace=0.16)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    hax = [fig.add_subplot(gs[1, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    for ax, lim in zip(axs, (LIM, LIM_B, LIM)):
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect(1)
    for ax, P in zip(axs, panel_cfgs(S)):
        im = draw_panel(fig, ax, P, S, norm)

    cb = fig.colorbar(im, cax=cax)
    cb.set_label('A$_V$  [mag]   (column along the projection axis)', labelpad=8, fontsize=10)
    axs[0].legend(handles=_map_legend(n_iso, 'fitted spine, projected'),
                  loc='lower left', bbox_to_anchor=(-0.015, -0.015), fontsize=7.5, framealpha=0.9,
                  facecolor='white', edgecolor='0.45')

    ncl = len(names); ns_ = len(fits_[names[0]]['samples'])
    _inclination_hists(hax, ang_mean, ang_samp, ns_, ncl, ['X$-$Y', 'Y$-$Z', 'X$-$Z'])
    fig.canvas.draw()
    for a, h in zip(axs, hax):
        p = a.get_position(); q = h.get_position()
        h.set_position([p.x0 + 0.03 * p.width, q.y0, p.width * 0.94, q.height])

    fig.suptitle('Edenhofer+2023 3D dust  —  three orthogonal projections, dust column in visual magnitudes,'
                 '  and the inclination of the fitted spines to each plane', fontsize=13, y=0.95)
    out = outdir + 'edenhofer_xyz_topdown_mom0_inclination.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', out)
    print('%-18s %s' % ('cloud', 'out-of-plane angle to X-Y / Y-Z / X-Z (deg), mean spine'))
    for nm, a in zip(names, ang_mean):
        print('%-18s %5.1f %5.1f %5.1f' % (nm, *a))


def projection_panels_figure(csv, fits_dir, outdir, n_iso=N_ISO, sky_cache=None, xyz_cache=None):
    """(from plot_projection_panels.py) projection_panels_xy_lb_bd.png: the
    face-on X-Y, l-b and b-d map panels side by side on one colour scale."""
    outdir = _dir(outdir)
    S = sky_prepare(csv, fits_dir, sky_cache or outdir + '_sky_maps/sky_projection_maps.pkl', n_iso)
    X = xyz_prepare(csv, fits_dir, xyz_cache or outdir + '_xyz_maps/xyz_projection_maps.npz', n_iso)
    maps = (S['Alb'], X['Aa'], S['Abd'])
    norm = LogNorm(min(np.nanpercentile(A, 1) for A in maps),
                   max(np.nanpercentile(A, 99.9) for A in maps))

    fig = plt.figure(figsize=(27, 7.6))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1.55, 0.62, 0.03], wspace=0.22)
    ax_xy, ax_lb, ax_bd, cax = (fig.add_subplot(gs[0, k]) for k in range(4))

    draw_maps(fig, S, norm, ax_lb=ax_lb)
    im = draw_panel(fig, ax_xy, panel_cfgs(X)[0], X, norm)
    draw_maps(fig, S, norm, ax_bd=ax_bd)

    ax_lb.set_title(r'b.  $\ell$–$b$  (plane of the sky),  integrated along $d$,  '
                    r'$d\leq$%d pc' % DMAX, fontsize=11, pad=8)
    ax_xy.set_title('a.  face-on  (X$-$Y, the Galactic plane),  integrated along Z',
                    fontsize=11, pad=8)
    ax_bd.set_title(r'c.  $b$–$d$,  integrated along $\ell$', fontsize=11, pad=8)

    cb = fig.colorbar(im, cax=cax)
    cb.set_label('A$_V$  [mag]   (column along the projection axis)', labelpad=8, fontsize=10)
    ax_lb.legend(handles=_map_legend(n_iso), loc='lower left', fontsize=7.5, framealpha=0.9,
                 facecolor='white', edgecolor='0.45')
    out = outdir + 'projection_panels_xy_lb_bd.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', out)


def inclination_panels_figure(names, fits_dir, out, n_iso=N_ISO):
    """(from plot_inclination_panels.py) 1 x 3 inclination histograms: to the
    plane of the sky (l-b), the Galactic plane (X-Y) and the b-d plane."""
    tag = f'n{n_iso:g}_s40'
    F = {nm: pickle.load(open(f'{fits_dir}/{nm}_spine_fits_{tag}.pkl', 'rb')) for nm in names}
    ang_mean = np.array([inclination_angles(F[nm]['mean']) for nm in names])
    ang_samp = np.array([inclination_angles(s) for nm in names for s in F[nm]['samples']])
    ns_, ncl = len(F[names[0]]['samples']), len(names)

    planes = [r'$\ell$–$b$ plane (plane of the sky)', r'$X$–$Y$ plane (Galactic plane)',
              r'$b$–$d$ plane']
    short = [r'$\ell$–$b$', r'$X$–$Y$', r'$b$–$d$']
    ctr, wid = 0.5 * (BINS[:-1] + BINS[1:]), np.diff(BINS)
    iso = ncl * (np.sin(np.deg2rad(BINS[1:])) - np.sin(np.deg2rad(BINS[:-1])))

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), dpi=150, constrained_layout=True)
    ymax = 0
    for i, ax in enumerate(axs):
        hm, _ = np.histogram(ang_mean[:, i], BINS)
        hs, _ = np.histogram(ang_samp[:, i], BINS)
        ax.bar(ctr, hm, width=wid * 0.96, color='#4fd0e7', edgecolor='#1d6f82', lw=0.9,
               label=f'mean-map spines ({ncl} clouds)', zorder=2)
        ax.stairs(hs / ns_, BINS, color='black', lw=1.4,
                  label=f'per-sample spines (counts / {ns_})', zorder=3)
        ax.stairs(iso, BINS, color='tab:orange', lw=1.2, ls='--',
                  label='isotropic expectation', zorder=3)
        ax.set_xlim(0, 90); ax.set_xticks(BINS)
        ax.set_xlabel(f'angle out of the {short[i]} plane (°; 0 = in plane, 90 = perpendicular)')
        ax.set_ylabel('number of clouds')
        ax.set_title(f'{"abc"[i]}.  inclination to the {planes[i]}   '
                     f'(median {np.median(ang_mean[:, i]):.0f}°)', fontsize=10.5)
        ax.grid(True, axis='y', alpha=0.25)
        ymax = max(ymax, hm.max(), (hs / ns_).max(), iso.max())
    for ax in axs:
        ax.set_ylim(0, ymax * 1.15)
    axs[0].legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig.suptitle(f'Inclination of the fitted spines  ($n_{{iso}}$ = {n_iso:g} cm$^{{-3}}$)',
                 fontsize=12)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)
    for nm, a in zip(names, ang_mean):
        print(f'  {nm:16s} sky {a[0]:5.1f}  galactic {a[1]:5.1f}  b-d {a[2]:5.1f}')


def far_ysos(path=YSO_ALL, rmin=500.0):
    """(from plot_sky_ysos.py) (l, b) of the KYSO sources with rgeo > rmin pc. The
    FITS header has -nan TAMAX/TAMIN cards that astropy refuses until silentfix'ed."""
    with fits.open(path) as h:
        h.verify('silentfix')
        y = Table(h[1].data)
    r = np.asarray(y['rgeo'], float)
    y = y[np.isfinite(r) & (r > rmin)]
    return wrap_l(np.asarray(y['GLON'], float)), np.asarray(y['GLAT'], float)


def closed_niso_maps(S, n_iso, cache, stat='max'):
    """Per cloud, the max n_H over its distance window on the whole sky, kept only
    on the n_H >= n_iso regions (8-connected, periodic in l) that touch the
    catalog box and zeroed elsewhere, cropped to those regions + 1 px. A contour
    at n_iso of this closes however far the cloud's region reaches -- no fixed
    padding. Read from the wrap cube (the released mean map; the stored-sample
    mean the 0.75 deg contours use agrees with it on >= 99% of n_iso pixels).
    stat: 'max' or 'mean' of n_H over the window along each sightline.
    Returns [dict(l, b, mx)] in names order, l unwrapped around the cloud (None
    for a cloud with no pixel >= n_iso in its box)."""
    names, df, cat = S['names'], S['df'], S['cat']
    key = np.concatenate([[n_iso, stat == 'max'], df.win_lo.values, df.win_hi.values,
                          cat[['l_min(deg)', 'l_max(deg)', 'b_min(deg)', 'b_max(deg)']].values.ravel()])
    if os.path.exists(cache):
        z = pickle.load(open(cache, 'rb'))
        if z['key'].shape == key.shape and np.allclose(z['key'], key):
            print('[cache]', cache)
            return z['clouds']
    cube = fits.open(WRAP, memmap=True)[0].data                  # [d, b, l]
    nb = cube.shape[1]
    b_all = -90.0 + CD * np.arange(nb)
    l_all = L0 - CD * (np.arange(NL) + X0)                     # unique cols, l from +180 down
    out = []
    for i, nm in enumerate(names):
        row = cat.iloc[i]
        z0 = int(round(df.win_lo.values[i] - D0)); z1 = int(round(df.win_hi.values[i] - D0)) + 1
        red = np.nanmax if stat == 'max' else np.nanmean
        mx = np.nan_to_num(red(np.asarray(cube[z0:z1, :, X0:X0 + NL], np.float32), axis=0))
        lab, _ = ndi.label(mx >= n_iso, structure=np.ones((3, 3)))
        for y in range(nb):                                          # join labels across l = +-180
            for yy in (y - 1, y, y + 1):
                if 0 <= yy < nb and lab[y, 0] and lab[yy, -1] and lab[y, 0] != lab[yy, -1]:
                    lab[lab == lab[yy, -1]] = lab[y, 0]
        lc, bc, _, (xo, yo) = box_grid(row)
        cols = (np.arange(xo, xo + len(lc)) - X0) % NL
        ids = np.unique(lab[yo:yo + len(bc)][:, cols]); ids = ids[ids > 0]
        keep = np.isin(lab, ids)
        if not keep.any():
            out.append(None); print(f'  {nm:18s} no {stat} n_H >= {n_iso:g} in the box'); continue
        lmid = 0.5 * (row['l_max(deg)'] + row['l_min(deg)'])
        dl = wrap_l(l_all - lmid)                                    # l relative to the box centre
        ys, xs = np.nonzero(keep)
        y0, y1 = max(ys.min() - 1, 0), min(ys.max() + 2, nb)
        order = np.argsort(dl[np.unique(xs)])
        dsel = dl[np.unique(xs)][order]
        cx = np.arange(int(np.floor(dsel.min() / CD)) - 1, int(np.ceil(dsel.max() / CD)) + 2)
        lgrid = lmid + cx * CD                                     # unwrapped l, increasing
        src = np.argmin(np.abs(wrap_l(l_all[None, :] - lgrid[:, None])), axis=1)
        sub = np.where(keep, mx, 0.0)[y0:y1][:, src]
        out.append(dict(l=wrap_l(lmid) + cx * CD, b=b_all[y0:y1], mx=sub.astype(np.float32)))
        print(f'  {nm:18s} closed {stat} n={n_iso:g} region: l {lgrid.min():.1f}..{lgrid.max():.1f}  '
              f'b {b_all[y0]:.1f}..{b_all[y1 - 1]:.1f}', flush=True)
    os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
    pickle.dump({'key': key, 'clouds': out}, open(cache, 'wb'))
    return out


def draw_closed_niso(ax, C, n_iso, color='black', lw=0.8, zorder=5.8):
    """Contour each closed region at n_iso; also at l -+ 360 so a region crossing
    l = +-180 shows on both sides of the map."""
    for c in C:
        if c is None:
            continue
        Lg, Bg = np.meshgrid(c['l'], c['b'])
        for sh in (-360.0, 0.0, 360.0):
            if (c['l'] + sh).max() < -180 or (c['l'] + sh).min() > 180:
                continue
            ax.contour(Lg + sh, Bg, c['mx'], levels=[n_iso], colors=color, linewidths=lw,
                       zorder=zorder)


def full_cube_mean_map(cache):
    """Mean n_H along each sightline over the whole wrap cube (69-1250 pc), on the
    unique-l sky grid [b, l] with l from +180 down. Cached (it reads ~7 GB)."""
    if os.path.exists(cache):
        print('[cache]', cache)
        return np.load(cache)
    cube = fits.open(WRAP, memmap=True)[0].data
    s = np.zeros((cube.shape[1], NL)); c = np.zeros_like(s)
    for k in range(cube.shape[0]):
        a = np.asarray(cube[k, :, X0:X0 + NL], np.float32); ok = np.isfinite(a)
        s += np.where(ok, a, 0.0); c += ok
    m = (s / np.maximum(c, 1)).astype(np.float32); m[c == 0] = np.nan
    os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
    np.save(cache, m)
    return m


def contours3_figure(S, nmap, nnorm, n_iso, C_max, C_mean, full, yl, yb, fl, fb, out,
                     near_col, far_col, ms, n_near, dmax):
    """The YSO sky map with three n_iso contours: max n_H over each cloud's window
    (closed), mean n_H over the window (closed), and mean n_H over the full cube."""
    COLS = {'max': 'limegreen', 'mean': 'darkorange', 'full': 'blue'}
    fig, ax = plt.subplots(figsize=(17, 6.4))
    im = draw_maps(fig, S, nnorm, ax_lb=ax, niso_color='black', labels='edges', cmap='binary',
                   leader_color='0.45', lb_map=nmap, av_contour=False, niso_contour=False,
                   gc_label=False)
    draw_closed_niso(ax, C_max, n_iso, color=COLS['max'], zorder=8.5)
    draw_closed_niso(ax, C_mean, n_iso, color=COLS['mean'], zorder=8.6)
    fmax = np.nanmax(full)
    if fmax >= n_iso:
        lg = wrap_l(L0 - CD * (np.arange(NL) + X0))
        bg = -90.0 + CD * np.arange(full.shape[0])
        ax.contour(lg, bg, np.nan_to_num(full), levels=[n_iso], colors=COLS['full'],
                   linewidths=0.8, zorder=8.7)
    ax.scatter(fl, fb, s=ms, c=far_col, alpha=0.75, lw=0, zorder=5.4, rasterized=True)
    ax.scatter(yl, yb, s=ms, c=near_col, alpha=0.75, lw=0, zorder=5.5, rasterized=True)
    ax.set_title(r'$\ell$–$b$ mean $n_H$ along the sightline,  %d$\,\leq d\leq\,$%d pc,  '
                 r'with KYSO YSOs (red: $r_{\rm geo}\leq500$ pc, magenta: $>500$ pc);  '
                 r'contours at $n_H=%g$ cm$^{-3}$' % (D0, dmax, n_iso), fontsize=11, pad=8)
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
    cb.set_label(r'$\langle n_H \rangle$  [cm$^{-3}$]', labelpad=6)
    empty = lambda C: '' if any(c is not None for c in C) else '  (none)'
    leg = ax.legend(handles=[
        Line2D([], [], color=COLS['max'], lw=0.8, label='max $n_H$ over window' + empty(C_max)),
        Line2D([], [], color=COLS['mean'], lw=0.8, label='mean $n_H$ over window' + empty(C_mean)),
        Line2D([], [], color=COLS['full'], lw=0.8,
               label='mean $n_H$, full cube' + ('' if fmax >= n_iso else f'  (none: max {fmax:.1f})')),
        Line2D([], [], color='#8fecff', lw=2.0, label='fitted spine',
               path_effects=[pe.withStroke(linewidth=3.6, foreground='black', alpha=0.6)]),
        Line2D([], [], color=near_col, marker='o', ls='', ms=4,
               label=f'YSO, $\\leq$500 pc  (N={n_near})'),
        Line2D([], [], color=far_col, marker='o', ls='', ms=4,
               label=f'YSO, $>$500 pc  (N={len(fl)})')],
        loc='upper right', fontsize=8, framealpha=0.9, facecolor='white', edgecolor='0.45')
    leg.set_zorder(10)
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', out)


def sky_ysos_figure(csv, fits_dir, outdir, n_iso=N_ISO, cache=None, ysos=YSO_CAT,
                    ysos_all=YSO_ALL, yso_color='#e41a1c', ms=6):
    """(from plot_sky_ysos.py) sky_lb_ysos_n<N>.png: the l-b map of mean n_H along
    the sightline (D0 <= d <= DMAX) with the clouds' n_iso contours and spines, the YSOs in `ysos` scattered on top, and the
    rgeo > 500 pc sources of `ysos_all` in magenta."""
    FAR_COL = 'magenta'
    NISO_COL = 'limegreen'  # closed n_iso contours, drawn above everything but the labels
    outdir = _dir(outdir)
    S = sky_prepare(csv, fits_dir, cache or outdir + '_sky_maps/sky_projection_maps.pkl', n_iso)
    y = Table.read(ysos)
    yl, yb = wrap_l(np.asarray(y['GLON'], float)), np.asarray(y['GLAT'], float)
    fl, fb = far_ysos(ysos_all)
    C = closed_niso_maps(S, n_iso, outdir + f'_sky_maps/closed_niso_n{n_iso:g}.pkl')

    fig, ax = plt.subplots(figsize=(17, 6.4))
    # mean n_H along each sightline over D0 <= d <= DMAX: the column / path length
    nmap = S['Alb'] * AV / PC / len(S['d_c'])
    nnorm = LogNorm(np.nanpercentile(nmap, 1), np.nanpercentile(nmap, 99.9))
    im = draw_maps(fig, S, nnorm, ax_lb=ax, niso_color='black', labels='edges',
                   cmap='binary', leader_color='0.45', lb_map=nmap, av_contour=False,
                   niso_contour=False, gc_label=False)
    draw_closed_niso(ax, C, n_iso, color=NISO_COL, zorder=8.5)
    ax.scatter(fl, fb, s=ms, c=FAR_COL, alpha=0.75, lw=0, zorder=5.4, rasterized=True)
    ax.scatter(yl, yb, s=ms, c=yso_color, alpha=0.75, lw=0, zorder=5.5, rasterized=True)
    ax.set_title(r'$\ell$–$b$ mean $n_H$ along the sightline,  %d$\,\leq d\leq\,$%d pc,  '
                 r'with KYSO YSOs (red: $r_{\rm geo}\leq500$ pc, magenta: $>500$ pc)' % (D0, DMAX),
                 fontsize=11, pad=8)
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
    cb.set_label(r'$\langle n_H \rangle$  [cm$^{-3}$]', labelpad=6)
    leg = ax.legend(handles=[Line2D([], [], color=NISO_COL, lw=0.8,
                              label=f'$n_H={n_iso:g}$ cm$^{{-3}}$'),
                       Line2D([], [], color='#8fecff', lw=2.0, label='fitted spine',
                              path_effects=[pe.withStroke(linewidth=3.6, foreground='black', alpha=0.6)]),
                       Line2D([], [], color=yso_color, marker='o', ls='', ms=4,
                              label=f'YSO, $\\leq$500 pc  (N={len(y)})'),
                       Line2D([], [], color=FAR_COL, marker='o', ls='', ms=4,
                              label=f'YSO, $>$500 pc  (N={len(fl)})')],
              loc='upper right', fontsize=8, framealpha=0.9, facecolor='white', edgecolor='0.45')
    leg.set_zorder(10)
    out = outdir + f'sky_lb_ysos_n{n_iso:g}.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', out)


def sky_ysos_contours3_figures(csv, fits_dir, outdir, n_values=(5, 10, 15, 20), cache=None,
                               ysos=YSO_CAT, ysos_all=YSO_ALL, ms=6):
    """(from plot_sky_ysos.py --contours3) sky_lb_ysos_contours3_n<N>.png for each N:
    the YSO sky map with three n_H = N contours -- max and mean n_H over each
    cloud's distance window (closed regions touching the box) and mean n_H over
    the full cube (69-1250 pc)."""
    outdir = _dir(outdir)
    S = sky_prepare(csv, fits_dir, cache or outdir + '_sky_maps/sky_projection_maps.pkl')
    y = Table.read(ysos)
    yl, yb = wrap_l(np.asarray(y['GLON'], float)), np.asarray(y['GLAT'], float)
    fl, fb = far_ysos(ysos_all)
    nmap = S['Alb'] * AV / PC / len(S['d_c'])
    nnorm = LogNorm(np.nanpercentile(nmap, 1), np.nanpercentile(nmap, 99.9))
    full = full_cube_mean_map(outdir + '_sky_maps/full_cube_mean_n.npy')
    for n in n_values:
        Cx = closed_niso_maps(S, n, outdir + f'_sky_maps/closed_niso_n{n:g}.pkl')
        Cm = closed_niso_maps(S, n, outdir + f'_sky_maps/closed_niso_mean_n{n:g}.pkl', stat='mean')
        contours3_figure(S, nmap, nnorm, n, Cx, Cm, full, yl, yb, fl, fb,
                         outdir + f'sky_lb_ysos_contours3_n{n:g}.png', '#e41a1c', 'magenta',
                         ms, len(y), DMAX)


def full_cube_max_map(cache):
    """Max n_H along each sightline over the whole wrap cube (69-1250 pc), and the
    distance (pc) of the slice where that max sits, on the unique-l sky grid
    [b, l] with l from +180 down. One streaming pass over the ~7 GB cube; cached.
    Returns (mx, dmx); both NaN where the sightline has no data."""
    if os.path.exists(cache):
        print('[cache]', cache)
        z = np.load(cache)
        return z['mx'], z['dmx']
    cube = fits.open(WRAP, memmap=True)[0].data                  # [d, b, l]
    mx = np.full((cube.shape[1], NL), -np.inf, np.float32)
    kx = np.zeros(mx.shape, np.int16)
    t0 = time.time()
    for k in range(cube.shape[0]):
        a = np.nan_to_num(np.asarray(cube[k, :, X0:X0 + NL], np.float32), nan=-np.inf)
        hi = a > mx
        mx[hi] = a[hi]; kx[hi] = k
        if k % 200 == 0:
            print(f'  d slice {k}/{cube.shape[0]}  {time.time() - t0:.0f}s', flush=True)
    dmx = (D0 + kx).astype(np.float32)
    bad = ~np.isfinite(mx)
    mx[bad] = np.nan; dmx[bad] = np.nan
    os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
    np.savez(cache, mx=mx, dmx=dmx)
    return mx, dmx


def niso_contour_by_distance(mx, dmx, n_iso):
    """Segments of the n_iso contour of the max-n_H map, each tagged with the
    distance of the max at that point. The contour is traced in pixel-index space
    (so it is continuous across l = +-180); a vertex takes the distance of the
    densest of its 4 neighbouring pixels, i.e. the pixel on the n >= n_iso side.
    Returns (segments [N, 2, 2] in (l, b) deg, distance per segment [N])."""
    import contourpy
    z = np.nan_to_num(mx, nan=0.0)
    nb, nl = z.shape
    lines = contourpy.contour_generator(np.arange(nl), np.arange(nb), z).lines(n_iso)
    segs, dist = [], []
    for ln in lines:
        if len(ln) < 2:
            continue
        x, y = ln[:, 0], ln[:, 1]
        x0, y0 = np.clip(np.floor(x).astype(int), 0, nl - 2), np.clip(np.floor(y).astype(int), 0, nb - 2)
        cand = np.stack([(y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)], axis=0)  # [4, 2, n]
        vals = np.stack([z[cy, cx] for cy, cx in cand], axis=0)
        j = np.argmax(vals, axis=0)
        cy = cand[j, 0, np.arange(len(x))]; cx = cand[j, 1, np.arange(len(x))]
        d = dmx[cy, cx]
        l = wrap_l(L0 - CD * (x + X0)); b = -90.0 + CD * y
        pts = np.column_stack([l, b])
        seg = np.stack([pts[:-1], pts[1:]], axis=1)
        keep = np.abs(np.diff(l)) < 180                                # no segment across the seam
        segs.append(seg[keep]); dist.append(0.5 * (d[:-1] + d[1:])[keep])
    return np.concatenate(segs), np.concatenate(dist)


def sky_ysos_dmax_figure(csv, fits_dir, outdir, n_iso=N_ISO, cache=None, ysos=YSO_CAT,
                         ysos_all=YSO_ALL, yso_color='#e41a1c', ms=6, dcmap='viridis_r'):
    """sky_lb_ysos_n<N>_dmax.png: as sky_ysos_figure, but the n_iso contour is of
    the max n_H over the FULL distance axis (69-1250 pc) on the whole sky, not per
    cloud over its window, and each piece of contour is coloured by the distance
    at which that max occurs."""
    from matplotlib.collections import LineCollection
    FAR_COL = 'magenta'
    outdir = _dir(outdir)
    S = sky_prepare(csv, fits_dir, cache or outdir + '_sky_maps/sky_projection_maps.pkl', n_iso)
    y = Table.read(ysos)
    yl, yb = wrap_l(np.asarray(y['GLON'], float)), np.asarray(y['GLAT'], float)
    fl, fb = far_ysos(ysos_all)
    mx, dmx = full_cube_max_map(outdir + '_sky_maps/full_cube_max_n.npz')
    segs, dist = niso_contour_by_distance(mx, dmx, n_iso)
    vis = np.abs(segs[:, :, 1]).max(axis=1) <= BLIM
    print(f'  n={n_iso:g} contour of the full-axis max: {len(segs)} segments, '
          f'{vis.sum()} within |b| <= {BLIM:g}; distance of the max along it: '
          f'{np.nanpercentile(dist[vis], 5):.0f}-{np.nanpercentile(dist[vis], 95):.0f} pc '
          f'(5-95%), median {np.nanmedian(dist[vis]):.0f}')

    fig, ax = plt.subplots(figsize=(17, 6.4))
    nmap = S['Alb'] * AV / PC / len(S['d_c'])
    nnorm = LogNorm(np.nanpercentile(nmap, 1), np.nanpercentile(nmap, 99.9))
    im = draw_maps(fig, S, nnorm, ax_lb=ax, niso_color='black', labels='edges',
                   cmap='binary', leader_color='0.45', lb_map=nmap, av_contour=False,
                   niso_contour=False, gc_label=False)
    dnorm = colors.Normalize(D0, D0 + 1181)
    lc_ = LineCollection(segs, cmap=dcmap, norm=dnorm, linewidths=1.1, zorder=8.5)
    lc_.set_array(dist)
    ax.add_collection(lc_)
    ax.scatter(fl, fb, s=ms, c=FAR_COL, alpha=0.75, lw=0, zorder=5.4, rasterized=True)
    ax.scatter(yl, yb, s=ms, c=yso_color, alpha=0.75, lw=0, zorder=5.5, rasterized=True)
    ax.set_title(r'$\ell$–$b$ mean $n_H$ along the sightline (%d$\,\leq d\leq\,$%d pc),  KYSO YSOs,  '
                 r'and the $n_H=%g$ cm$^{-3}$ contour of the max $n_H$ over the full $d$ axis '
                 r'(%d–%d pc), coloured by the distance of that max'
                 % (D0, DMAX, n_iso, D0, D0 + 1181), fontsize=10.5, pad=8)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div = make_axes_locatable(ax)
    cb2 = fig.colorbar(lc_, cax=div.append_axes('right', size='1.4%', pad=0.12))
    cb2.set_label(r'distance of max $n_H$ on the contour  [pc]', labelpad=6)
    cb = fig.colorbar(im, cax=div.append_axes('right', size='1.4%', pad=1.0))
    cb.set_label(r'$\langle n_H \rangle$  [cm$^{-3}$]', labelpad=6)
    leg = ax.legend(handles=[
        Line2D([], [], color=plt.get_cmap(dcmap)(0.6), lw=1.1,
               label=f'max$_d\\ n_H={n_iso:g}$ cm$^{{-3}}$ (colour: $d$ of max)'),
        Line2D([], [], color='#8fecff', lw=2.0, label='fitted spine',
               path_effects=[pe.withStroke(linewidth=3.6, foreground='black', alpha=0.6)]),
        Line2D([], [], color=yso_color, marker='o', ls='', ms=4,
               label=f'YSO, $\\leq$500 pc  (N={len(y)})'),
        Line2D([], [], color=FAR_COL, marker='o', ls='', ms=4,
               label=f'YSO, $>$500 pc  (N={len(fl)})')],
        loc='upper right', fontsize=8, framealpha=0.9, facecolor='white', edgecolor='0.45')
    leg.set_zorder(10)
    out = outdir + f'sky_lb_ysos_n{n_iso:g}_dmax.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', out)


def full_xyz_max_map(cache):
    """Max n_H along Z through the whole xyz cube, and the Z (pc) of the cell where
    that max sits, on the cube's [Y, X] grid (2 pc). One streaming pass over the
    15.6 GB cube; cached. Returns (mx, zmx, gfull); NaN where a column has no data."""
    if os.path.exists(cache):
        print('[cache]', cache)
        z = np.load(cache)
        return z['mx'], z['zmx'], z['gfull']
    D_ = fits.open(XYZ_CUBE, memmap=True)[1].data                 # axes [Z, Y, X]
    N = D_.shape[0]
    gfull = (np.arange(N) - (N - 1) / 2) * 2.0
    mx = np.full((N, N), -np.inf, np.float32); kx = np.zeros((N, N), np.int16)
    t0 = time.time()
    for k in range(N):
        a = np.nan_to_num(np.asarray(D_[k], np.float32) * XYZ_K, nan=-np.inf)
        hi = a > mx
        mx[hi] = a[hi]; kx[hi] = k
        if k % 150 == 0:
            print(f'  z {k}/{N}  {time.time() - t0:.0f}s', flush=True)
    zmx = gfull[kx].astype(np.float32)
    bad = ~np.isfinite(mx); mx[bad] = np.nan; zmx[bad] = np.nan
    os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
    np.savez(cache, mx=mx, zmx=zmx, gfull=gfull)
    return mx, zmx, gfull


def yso_xy(path, rmin=None, rmax=None):
    """(X, Y) in pc of the KYSO sources with rmin < rgeo <= rmax, in the face-on
    frame (X toward l = 0, Y toward l = 90 deg)."""
    with fits.open(path) as h:
        h.verify('silentfix')
        y = Table(h[1].data)
    r = np.asarray(y['rgeo'], float)
    ok = np.isfinite(r)
    if rmin is not None:
        ok &= r > rmin
    if rmax is not None:
        ok &= r <= rmax
    l = np.deg2rad(np.asarray(y['GLON'], float)[ok]); b = np.deg2rad(np.asarray(y['GLAT'], float)[ok])
    rc = r[ok] * np.cos(b)
    return rc * np.cos(l), rc * np.sin(l)


def xyz_ysos_zmax_figure(csv, fits_dir, outdir, n_iso=N_ISO, cache=None, ysos=YSO_CAT,
                         ysos_all=YSO_ALL, yso_color='#e41a1c', ms=6, zcmap='viridis_r'):
    """xy_ysos_n<N>_zmax.png: the face-on (X-Y) counterpart of sky_lb_ysos_n<N>_dmax.
    Background: mean n_H along Z through the cube. Contour: n_iso of the max n_H
    along Z, coloured by the height Z at which that max sits (viridis_r; 0 = the
    midplane). Spines and labels as in the face-on panel of the xyz figure; YSOs
    at their (X, Y) from rgeo (red <= 500 pc, magenta > 500 pc), inside RMAX."""
    from matplotlib.collections import LineCollection
    import contourpy
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    FAR_COL = 'magenta'
    outdir = _dir(outdir)
    X = xyz_prepare(csv, fits_dir, cache or outdir + '_xyz_maps/xyz_projection_maps.npz', n_iso)
    P = panel_cfgs(X)[0]
    mx, zmx, gfull = full_xyz_max_map(outdir + '_xyz_maps/full_xyz_max_n.npz')
    N = len(gfull)
    # mean n_H along Z: the face-on column over the cube's full Z path (N cells of 2 pc)
    nmap = P['A'] * AV / (N * 2.0 * PC)
    P = dict(P, A=nmap)
    nnorm = LogNorm(np.nanpercentile(nmap, 1), np.nanpercentile(nmap, 99.9))

    # contour of the max along Z, in the panel frame [V = X, H = -Y] (as P['A'])
    mrot = np.nan_to_num(np.rot90(mx, -1), nan=0.0); zrot = np.rot90(zmx, -1)
    segs, zz = [], []
    for ln in contourpy.contour_generator(gfull, gfull, mrot).lines(n_iso):
        if len(ln) < 2:
            continue
        h, v = ln[:, 0], ln[:, 1]
        jx = (h - gfull[0]) / 2.0; iy = (v - gfull[0]) / 2.0
        x0 = np.clip(np.floor(jx).astype(int), 0, N - 2); y0 = np.clip(np.floor(iy).astype(int), 0, N - 2)
        cand = [(y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)]
        vals = np.stack([mrot[a, b] for a, b in cand]); j = np.argmax(vals, axis=0)
        zs = np.stack([zrot[a, b] for a, b in cand])[j, np.arange(len(h))]
        pts = np.column_stack([h, v]); seg = np.stack([pts[:-1], pts[1:]], axis=1)
        inside = np.hypot(h, v) <= RMAX
        keep = inside[:-1] & inside[1:]
        segs.append(seg[keep]); zz.append(0.5 * (zs[:-1] + zs[1:])[keep])
    segs, zz = np.concatenate(segs), np.concatenate(zz)
    zlim = float(np.ceil(np.nanpercentile(np.abs(zz), 95) / 10) * 10)
    print(f'  n={n_iso:g} contour of the max along Z within r <= {RMAX:g} pc: {len(segs)} segments; '
          f'Z of the max: 5-95% {np.nanpercentile(zz, 5):.0f} to {np.nanpercentile(zz, 95):.0f} pc, '
          f'median {np.nanmedian(zz):.0f}; colour range +-{zlim:g} pc')

    fig, ax = plt.subplots(figsize=(11, 10))
    im = draw_panel(fig, ax, P, X, nnorm, cmap='binary', cloud_contours=False)
    lc_ = LineCollection(segs, cmap=zcmap, norm=colors.Normalize(-zlim, zlim), linewidths=1.0,
                         zorder=Z_SPINE - 0.1)
    lc_.set_array(zz)
    ax.add_collection(lc_)
    nx_, ny_ = yso_xy(ysos, rmax=500.0); fx_, fy_ = yso_xy(ysos_all, rmin=500.0)
    rin = lambda h, v: np.hypot(h, v) <= RMAX
    fh, fv = -fy_, fx_; nh, nv = -ny_, nx_
    ax.scatter(fh[rin(fh, fv)], fv[rin(fh, fv)], s=ms, c=FAR_COL, alpha=0.75, lw=0, zorder=5.4,
               rasterized=True)
    ax.scatter(nh[rin(nh, nv)], nv[rin(nh, nv)], s=ms, c=yso_color, alpha=0.75, lw=0, zorder=5.5,
               rasterized=True)
    ax.set_title('face-on (X$-$Y): mean $n_H$ along Z,  KYSO YSOs,  and the '
                 f'$n_H={n_iso:g}$ cm$^{{-3}}$ contour of the max $n_H$ along Z,\n'
                 'coloured by the height Z of that max', fontsize=10.5, pad=8)
    div = make_axes_locatable(ax)
    cb2 = fig.colorbar(lc_, cax=div.append_axes('right', size='2.5%', pad=0.12), extend='both')
    cb2.set_label('Z of max $n_H$ on the contour  [pc]  (0 = midplane)', labelpad=6)
    cb = fig.colorbar(im, cax=div.append_axes('right', size='2.5%', pad=0.95))
    cb.set_label(r'$\langle n_H \rangle$ along Z  [cm$^{-3}$]', labelpad=6)
    leg = ax.legend(handles=[
        Line2D([], [], color=plt.get_cmap(zcmap)(0.8), lw=1.1,
               label=f'max$_Z\\ n_H={n_iso:g}$ cm$^{{-3}}$ (colour: Z of max)'),
        Line2D([], [], color='#8fecff', lw=2.0, label='fitted spine, projected',
               path_effects=[pe.withStroke(linewidth=3.6, foreground='black', alpha=0.6)]),
        Line2D([], [], color=yso_color, marker='o', ls='', ms=4,
               label=f'YSO, $\\leq$500 pc  (N={int(rin(nh, nv).sum())} in view)'),
        Line2D([], [], color=FAR_COL, marker='o', ls='', ms=4,
               label=f'YSO, $>$500 pc  (N={int(rin(fh, fv).sum())} in view)')],
        loc='lower left', fontsize=7.5, framealpha=0.9, facecolor='white', edgecolor='0.45')
    leg.set_zorder(10)
    out = outdir + f'xy_ysos_n{n_iso:g}_zmax.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', out)


# ---- 10-13/14 experiments (not part of the adopted pipeline) ----------------
def saddle_window_figures(table1, win_csv, outdir, floor=0.01, tol=2.0, max_ext=300.0,
                          tag='saddle_balanced'):
    """(from plot_saddle_windows.py) Peak-to-saddle distance windows on the
    mass-per-slice profile, against the adopted n_iso window. Rejected on
    2026-10-14 in favour of the original window."""
    out = _dir(outdir)
    table1 = table1.reset_index(drop=True)
    cubes = build_cubes(table1)
    old = pd.read_csv(win_csv).set_index('cloud')

    profs, recs = [], []
    for i, row in table1.iterrows():
        name, d_mean = row['Name'], float(row['d_mean(pc)'])
        prof = mass_profile(cubes[i])
        k0, kp, (flo, fhi), (klo, khi), note = saddle_window(prof, d_mean, floor, tol, max_ext)
        dend, j, _, spans, _, _ = select_leaf(prof, d_mean)
        leaf = spans[j] if j is not None else (np.nan, np.nan)
        inwin = (ds >= ds[klo]) & (ds <= ds[khi])
        inold = (ds >= old.loc[name, 'win_lo']) & (ds <= old.loc[name, 'win_hi'])
        recs.append(dict(cloud=name, d_mean=d_mean, peak=ds[kp],
                         new_lo=ds[klo], new_hi=ds[khi], first_lo=ds[flo], first_hi=ds[fhi],
                         ratio=float(max(prof[klo], prof[khi]) / min(prof[klo], prof[khi])),
                         note=note,
                         old_lo=float(old.loc[name, 'win_lo']), old_hi=float(old.loc[name, 'win_hi']),
                         leaf_lo=float(leaf[0]), leaf_hi=float(leaf[1]),
                         M_new=float(prof[inwin].sum()), M_old=float(prof[inold].sum())))
        profs.append(prof)
        r = recs[-1]
        print(f"{name:16s} d_mean {d_mean:4.0f}  peak {r['peak']:4.0f}  "
              f"new {r['new_lo']:4.0f}-{r['new_hi']:4.0f}  old {r['old_lo']:4.0f}-{r['old_hi']:4.0f}  "
              f"first {r['first_lo']:4.0f}-{r['first_hi']:4.0f}  ratio {r['ratio']:5.1f}  "
              f"M_new/M_old {r['M_new'] / r['M_old']:.2f}  {note}",
              flush=True)

        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, constrained_layout=True)
        plot_saddle_one(ax, name, prof, d_mean, r)
        ax.legend(handles=SADDLE_LEGEND, fontsize=7.5, loc='upper right', framealpha=0.85)
        fig.savefig(f'{out}{name}_{tag}.png', bbox_inches='tight')
        plt.close(fig)

    n, nc = len(table1), 4
    nr = int(np.ceil((n + 1) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.8 * nc, 3.0 * nr), dpi=130, constrained_layout=True)
    axs = axs.ravel()
    for i, row in table1.iterrows():
        plot_saddle_one(axs[i], row['Name'], profs[i], float(row['d_mean(pc)']), recs[i],
                        with_labels=False)
        axs[i].tick_params(labelsize=7)
    for a in axs[n:]:
        a.axis('off')
    axs[n].legend(handles=SADDLE_LEGEND, loc='center', fontsize=9, frameon=False)
    fig.supxlabel('heliocentric distance (pc)')
    fig.supylabel(r'mass per slice ($M_\odot\,\mathrm{pc}^{-1}$)')
    fig.savefig(f'{out}{tag}_all_clouds.png', bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(recs).to_csv(f'{out}{tag}.csv', index=False)
    print('saved', out)


def av1_aperture_figures(table1, win_csv, outdir, n_iso=N_ISO, av=1.0):
    """(from plot_av1_aperture.py) The A_V >= av aperture (window column) on the
    mass-per-slice profile, and the window the original method gives on it."""
    out = _dir(outdir)
    table1 = table1.reset_index(drop=True)
    cubes = build_cubes(table1)
    old = pd.read_csv(win_csv).set_index('cloud')

    recs, keep = [], []
    for i, row in table1.iterrows():
        name, d_mean = row['Name'], float(row['d_mean(pc)'])
        cube = cubes[i]
        lo, hi = float(old.loc[name, 'win_lo']), float(old.loc[name, 'win_hi'])
        win = (ds >= lo) & (ds <= hi)
        av_map = np.nansum(cube[win], axis=0) * PC_CM / NH_PER_AV
        mask = av_map >= av

        mbox, (xo, yo) = mean_padded_box(table1, name, pad=max(PAD_DEG, 1.0))
        mproxy = CloudBoxProxy(mbox, xo, yo)

        prof_box = mass_profile(cube)
        dend, j, *_ = select_leaf(prof_box, d_mean)
        chk = grow_window(i, table1, dend.leaves[j].get_mask(), mproxy, n_iso)
        ok = np.allclose(chk, (lo, hi))

        prof_ap = masked_profile(cube, mask)
        rec = dict(cloud=name, d_mean=d_mean, old_lo=lo, old_hi=hi, check_ok=ok,
                   ap_frac=float(mask.mean()), new_lo=np.nan, new_hi=np.nan,
                   leaf_lo=np.nan, leaf_hi=np.nan, note='')
        if mask.any():
            dend_a, ja, _, spans, *_ = select_leaf(prof_ap, d_mean)
            if ja is None:
                rec['note'] = 'no leaf in aperture profile'
            else:
                rec['leaf_lo'], rec['leaf_hi'] = spans[ja]
                rec['new_lo'], rec['new_hi'] = grow_window(i, table1, dend_a.leaves[ja].get_mask(),
                                                           mproxy, n_iso)
        else:
            rec['note'] = f'no pixel reaches A_V = {av:g}'
        nw = (ds >= rec['new_lo']) & (ds <= rec['new_hi']) if np.isfinite(rec['new_lo']) else win
        rec['M_box'] = float(prof_box[win].sum())          # current window, box aperture
        rec['M_ap'] = float(prof_ap[win].sum())            # current window, A_V aperture
        rec['M_ap_new'] = float(prof_ap[nw].sum())         # new window, A_V aperture
        recs.append(rec)
        keep.append((row, av_map, mask, prof_box, prof_ap))
        print(f"{name:16s} check {'ok' if ok else 'MISMATCH ' + str(chk)}  aperture {100 * rec['ap_frac']:5.1f}%  "
              f"window {lo:4.0f}-{hi:4.0f} -> {rec['new_lo']:4.0f}-{rec['new_hi']:4.0f}  "
              f"M_ap/M_box {rec['M_ap'] / rec['M_box']:.2f}  {rec['note']}", flush=True)

        fig, (axm, axp) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150, constrained_layout=True,
                                       gridspec_kw={'width_ratios': [1, 1.4]})
        aperture_plot_cloud(axm, axp, row, av_map, mask, prof_box, prof_ap, rec)
        axm.set_xlabel('l (deg)'); axm.set_ylabel('b (deg)')
        axp.set_xlabel('heliocentric distance (pc)')
        axp.set_ylabel(r'mass per slice ($M_\odot\,\mathrm{pc}^{-1}$)')
        axp.legend(handles=APERTURE_LEGEND, fontsize=7, loc='upper right', framealpha=0.85)
        fig.savefig(f'{out}{name}_av1_aperture.png', bbox_inches='tight')
        plt.close(fig)

    n = len(keep)
    nr = int(np.ceil(n / 2))
    fig, axs = plt.subplots(nr, 4, figsize=(22, 3.6 * nr), dpi=110, constrained_layout=True,
                            gridspec_kw={'width_ratios': [1, 1.4, 1, 1.4]})
    for k, (row, av_map, mask, pb, pa) in enumerate(keep):
        r, c_ = divmod(k, 2)
        aperture_plot_cloud(axs[r, 2 * c_], axs[r, 2 * c_ + 1], row, av_map, mask, pb, pa, recs[k])
    if n % 2:
        axs[-1, 2].axis('off'); axs[-1, 3].axis('off')
        axs[-1, 3].legend(handles=APERTURE_LEGEND, loc='center', fontsize=10, frameon=False)
    fig.savefig(f'{out}av1_aperture_all_clouds.png', bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(recs).to_csv(f'{out}av1_aperture_windows.csv', index=False)
    print('saved', out)


def leak_figures(table1, win_csv, outdir, thr=(5.0, 10.0)):
    """(from plot_leak_niso.py) For each threshold, the share of the cloud's
    projected max_d n_H >= thr region that lies outside its catalog box, on the
    stored PAD_STORED padded mean cube."""
    out = _dir(outdir)
    tag = 'leak_n' + '_n'.join(f'{t:g}' for t in thr)
    table1 = table1.reset_index(drop=True)
    win = pd.read_csv(win_csv).set_index('cloud')

    recs, panels = [], []
    for i, row in table1.iterrows():
        name = row['Name']
        lo, hi = float(win.loc[name, 'win_lo']), float(win.loc[name, 'win_hi'])
        mbox, (xo, yo) = mean_padded_box(table1, name, pad=PAD_STORED)
        sel = (ds >= lo) & (ds <= hi)
        P = np.nanmax(mbox[sel], axis=0)                     # (nb, nl) on the padded grid
        ny, nx = P.shape
        lc, bc, _, (bxo, byo) = box_grid(row)
        cat = np.zeros((ny, nx), bool)
        cat[byo - yo:byo - yo + len(bc), bxo - xo:bxo - xo + len(lc)] = True
        lp = L0 - CD * (xo + np.arange(nx))                  # padded-grid pixel centres
        bp = B0 + CD * (yo + np.arange(ny))

        res = {}
        line = f'{name:16s} win {lo:4.0f}-{hi:4.0f}'
        for t in thr:
            r = leak_one(P, cat, t)
            res[t] = r
            rec = dict(cloud=name, thr=t, win_lo=lo, win_hi=hi)
            if r is None:
                rec.update(leak=np.nan, closed=np.nan, box_cover=0.0, area_deg2=0.0)
                line += f' | n>={t:g}: none in box'
            else:
                rec.update({k: v for k, v in r.items() if k != 'region'})
                line += (f" | n>={t:g}: leak {r['leak']:.2f}{'' if r['closed'] else ' (open, >=)'}"
                         f" cover {r['box_cover']:.2f} area {r['area_deg2']:6.1f}")
            recs.append(rec)
        print(line, flush=True)
        panels.append((name, P, lp, bp, lc, bc, res))

    df = pd.DataFrame(recs)
    df.to_csv(f'{out}{tag}.csv', index=False)

    n, nc = len(panels), 4
    nr = int(np.ceil((n + 1) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.8 * nc, 4.3 * nr), dpi=120, constrained_layout=True)
    axs = axs.ravel()
    for ax, (name, P, lp, bp, lc, bc, res) in zip(axs, panels):
        ext = [lp[0] + CD / 2, lp[-1] - CD / 2, bp[0] - CD / 2, bp[-1] + CD / 2]
        pos = P[np.isfinite(P) & (P > 0)]
        ax.imshow(P, origin='lower', extent=ext, cmap='gray_r', aspect='equal',
                  norm=LogNorm(vmin=max(0.3, np.percentile(pos, 1)), vmax=np.percentile(pos, 99.7)))
        for t, col in zip(thr, LEAK_COLORS):
            r = res[t]
            if r is not None:
                ax.contour(lp, bp, r['region'].astype(float), levels=[0.5], colors=col, linewidths=1.3)
        ax.add_patch(Rectangle((lc[-1] - CD / 2, bc[0] - CD / 2), lc[0] - lc[-1] + CD,
                               bc[-1] - bc[0] + CD, fill=False, ec='k', lw=1.2, ls='--'))
        bits = []
        for t in thr:
            r = res[t]
            bits.append(f'{t:g}: ' + ('-' if r is None else
                                      f"{r['leak']:.2f}{'' if r['closed'] else '+'}"))
        ax.set_title(f"{name}   leak  " + '   '.join(bits), fontsize=9)
        ax.tick_params(labelsize=7)
    for a in axs[n:]:
        a.axis('off')
    handles = [Line2D([], [], color=c_, lw=1.6, label=rf'seed region, max$_d\,n_H \geq {t:g}$ cm$^{{-3}}$')
               for t, c_ in zip(thr, LEAK_COLORS)]
    handles.append(Line2D([], [], color='k', ls='--', lw=1.2, label='catalog box'))
    axs[n].legend(handles=handles, loc='center', fontsize=10, frameon=False,
                  title="leak = share of region outside box\n'+' = region reaches the 8 deg pad edge (lower bound)")
    fig.savefig(f'{out}{tag}_all_clouds.png', bbox_inches='tight')
    plt.close(fig)
    print('saved', out)

    piv = df.pivot(index='cloud', columns='thr', values='leak')
    print('\nleak, sorted by n>=%g:' % thr[0])
    print(piv.sort_values(thr[0]).round(2).to_string())
    return df


def av1_contour_figures(table1, outdir, blue='nicest', n_iso=N_ISO, niso_range='window',
                        win_csv=None):
    """(from plot_av1_contours.py) Per cloud: the Edenhofer A_V map summed over
    69-1250 pc with its A_V = 1 contour (orange) and, in blue, either the NICEST
    A_V = 1 contour (blue='nicest') or the projected n_iso contour over the
    cloud's window ('niso', which needs win_csv)."""
    LEGEND = contours_legend(blue, n_iso, niso_range)
    tag = 'av1_contours' if blue == 'nicest' else f'av1_n{n_iso:g}_contours'
    win = pd.read_csv(win_csv).set_index('cloud') if blue == 'niso' else None
    window = lambda nm: ((float(win.loc[nm, 'win_lo']), float(win.loc[nm, 'win_hi']))
                         if win is not None and niso_range == 'window' else None)
    kw = lambda nm: dict(blue=blue, n_iso=n_iso, window=window(nm))
    out = _dir(outdir)

    cubes = build_cubes(table1)
    n = len(table1)
    recs = []
    for i in range(n):
        row = table1.iloc[i]
        fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150, constrained_layout=True)
        im, st = contours_plot_cloud(ax, row, cubes[i], **kw(row['Name']))
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(r'$A_V$ (mag), summed 69-1250 pc')
        ax.legend(handles=LEGEND, loc='upper right', fontsize=7.5, framealpha=0.85)
        fig.savefig(f"{out}{row['Name']}_{tag}.png", bbox_inches='tight')
        plt.close(fig)
        recs.append({'cloud': row['Name'], **st})
        other = ('NICEST>=1', st.get('frac_nicest')) if blue == 'nicest' else \
                (f'max n>={n_iso:g}', st.get('frac_niso'))
        print(f"{row['Name']:18s} Eden>=1: {st['frac_eden']:.2f}  {other[0]}: {other[1]:.2f}",
              flush=True)

    nc = 4
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.6 * nc, 4.0 * nr), dpi=130,
                            constrained_layout=True)
    axs = np.atleast_1d(axs).ravel()
    for i in range(n):
        contours_plot_cloud(axs[i], table1.iloc[i], cubes[i], with_labels=False,
                            **kw(table1.iloc[i]['Name']))
        axs[i].tick_params(labelsize=7)
    for a in axs[n:]:
        a.axis('off')
    if n < len(axs):
        axs[n].legend(handles=LEGEND, loc='center', fontsize=10, frameon=False)
    fig.savefig(f'{out}{tag}_all_clouds.png', bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(recs).to_csv(f'{out}{tag}_area_fractions.csv', index=False)
    print('saved', out)


# ============================================================================
# LEGACY: functions the pipeline does not call
# ============================================================================
# Added 2026-10-28 when the old scripts in code_and_data/ were retired. These
# are the functions from those scripts that never made it into the sections
# above, copied verbatim apart from: the D. (dust_3d_functions) prefix removed,
# names that would clash given a prefix from their script, and hard-coded paths
# pointed at this module's path constants. They are kept for reference and are
# not tested against the current pipeline; same-named helpers above may have
# changed since these were written. plot_n5_vs_n10 and plot_single_with_fit
# (plot_rho0_correlations.py) were deliberately dropped on 2026-10-26.

import argparse
import csv
import glob
import json
import math
import sys
import urllib.parse
import urllib.request
from collections import namedtuple
from pathlib import Path

from numpy.typing import NDArray
from scipy.ndimage import zoom
from astropy.modeling import models, fitting


# ----------------------------------------------------------------------------
# from dust_3d_functions.py: the original notebook helper module (fits, plotting, the skin-mesh code).
# ----------------------------------------------------------------------------
fttr = fitting.LevMarLSQFitter()

gmod = models.Gaussian1D(1., 0., 1.,)

dist       = lambda p1, p2: np.sqrt(np.nansum((p1-p2)**2))


def superGaussian(z, z0, L, eta):
    num = np.abs(z - z0)
    den = L/2

    with np.errstate(over='ignore'):     # (num/den)**eta -> inf for boxy eta, exp(-inf)=0
        return np.exp(-(num/den)**eta)


def superGaussian_scaled(z, rho0, z0, L, eta):
    # superGaussian is normalized to 1 at its crest, so curve_fit needs the
    # amplitude as a free parameter to fit a profile in cm^-3.
    return rho0*superGaussian(z, z0, L, eta)


def mif_cylinder(r, z, params):
    return plummer(r, params['rho0'], params['r0'], params['alpha']) *\
           superGaussian(z, params['z0'], params['L'], params['eta'])


def input_spine(i, source_array, cloud_catalog, sp_type='lbd'):
    # Never returns anything -- as in the source notebook. Depends on global `spines`.
    p1, p2 = spines[i]
    dats = get_99th_percs(i, source_array, cloud_catalog, which='both')

    if sp_type == 'lbd':
        p1_lbd = np.array(p1)
        p2_lbd = np.array(p2)

        p1_xyz = np.array(sph_2_xyz(p1))
        p2_xyz = np.array(sph_2_xyz(p2))
    else:
        p1_lbd = xyz_2_sph(p1)
        p2_lbd = xyz_2_sph(p2)

        p1_xyz = np.array(p1)
        p2_xyz = np.array(p2)

    L1d, alldat = L1ds(p1_xyz, p2_xyz, dats[1].to_numpy())

    all_df = pd.DataFrame(alldat, columns=['x', 'y', 'z', 'd', 'b', 'l', 'A', 'dperp', 'dparl'])
    cube2 = coords_to_cube(all_df[['d', 'b', 'l']].to_numpy(), all_df['dperp'])


def plot_dcube_projctns(dcube, ecube, df, pt, vect, coords, which):
    fig, axs = plt.subplots(1, 3, figsize=(15, 10))

    axs[0].set_ylabel(which)

    lbs = (coords[[1, 0, 2, 3]], coords[[0, 1, 2, 3]])
    lds = (coords[[1, 0, 4, 5]], coords[[0, 1, 4, 5]])
    bds = (coords[[2, 3, 4, 5]], coords[[2, 3, 4, 5]])

    for n, cm, ext, wch in zip([0, 1, 2], ['gnuplot', 'CMRmap', 'cubehelix'],
                                [lbs, lds, bds], ['l,b', 'l,d', 'b,d']):

        im = axs[n].contourf(np.nanmin(dcube.transpose(), axis=n), levels=50, cmap=cm, extent=ext[0])

        axs[n].quiver(pt[0], pt[1], vect[0], vect[1], color='cyan', pivot='mid', scale=4, angles='xy'),
        axs[n].quiver(pt[0], pt[2], vect[0], vect[2], color='cyan', pivot='mid', scale=4, angles='xy')


def plot_99th_perc_pts(axs, df):
    #pass in the cbe instead, and calculate the 99th percentile in here
    axs[0].scatter(df['l'], df['b'], color='red', s=2)
    axs[1].scatter(df['l'], df['d'], color='red', s=2)
    axs[2].scatter(df['b'], df['d'], color='red', s=2)


def axes_mappable(ax):
    """
    The artist on `ax` that carries the color scale, or None.

    Prefers filled artists (imshow, pcolormesh, contourf) over line contour
    sets, which are colour-mapped too but usually drawn in a single colour.
    """
    cands = [a for a in list(ax.images) + list(ax.collections)
             if isinstance(a, cm.ScalarMappable) and a.get_array() is not None]
    filled = [a for a in cands if getattr(a, 'filled', True)]
    return (filled or cands or [None])[-1]


def stack_colorbars(host, sources, labels=None, orientation='horizontal',
                    thickness=0.05, edge=0.12, add_contour_lines=True, **kws):
    """
    Fill one axes with a stack of colorbars belonging to other axes.

    Use this when a subplot grid has a spare cell: the empty panel becomes the
    legend for the panels that hold data, instead of the colorbars stealing
    width from each of them.

    Parameters:
    -----------
    host : matplotlib Axes
        The panel to give up. Its frame and ticks are switched off and the
        colorbars are inset inside it. Because they are insets, they follow
        `host` through tight_layout() and figure resizes.
    sources : sequence of Axes or ScalarMappable
        One entry per colorbar. An Axes is resolved to its colour-mapped
        artist by axes_mappable(), so this works on plots that never handed
        back their mappable.
    labels : sequence of str, optional
        One label per bar.
    orientation : {'horizontal', 'vertical'}
        Bar direction. Horizontal bars stack top to bottom, vertical bars
        stack left to right.
    thickness : float
        Bar thickness as a fraction of the host panel.
    edge : float
        Margin left at the ends of the stack, as a fraction of the host panel.
    add_contour_lines : bool
        When a source Axes also holds line contours, draw those levels onto
        its bar, so the black contours on the map can be read off the scale.
    **kws
        Passed through to fig.colorbar().

    Returns:
    --------
    cbars : list of matplotlib Colorbar
    """
    host.set_axis_off()
    fig = host.get_figure()

    n = len(sources)
    slot = (1.0 - 2 * edge) / n
    cbars = []

    for i, src in enumerate(sources):
        mappable = axes_mappable(src) if hasattr(src, 'collections') else src
        if mappable is None:
            continue

        centre = edge + (i + 0.5) * slot
        if orientation == 'horizontal':
            rect = [edge, (1.0 - centre) - thickness / 2, 1.0 - 2 * edge, thickness]
        else:
            rect = [centre - thickness / 2, edge, thickness, 1.0 - 2 * edge]

        cax = host.inset_axes(rect)
        cb = fig.colorbar(mappable, cax=cax, orientation=orientation, **kws)
        cax.tick_params(labelsize='x-small')

        if labels is not None:
            cb.set_label(labels[i], fontsize='small')

        if add_contour_lines and hasattr(src, 'collections'):
            for cs in src.collections:
                if getattr(cs, 'filled', True) is False:
                    cb.add_lines(cs, erase=False)

        cbars.append(cb)

    return cbars


def healpix_directions(nside):
    """
    Unit vectors toward the center of every HEALPix pixel, shape (12 nside^2, 3).

    HEALPix is used rather than an arbitrary set of directions because its
    pixels are exactly equal-area -- so a solid-angle average over rays is an
    unweighted mean -- and because a direction can be mapped back to its pixel
    index in O(1), which is what makes mask_interior_to_skin() cheap.
    """
    hp = HEALPix(nside=nside, order='ring')
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    lon, lat = lon.to_value(u.rad), lat.to_value(u.rad)
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=1)


def directions_to_healpix(vectors, nside):
    """Pixel index containing each (unnormalized) direction vector, shape (N,)."""
    hp = HEALPix(nside=nside, order='ring')
    lon = np.arctan2(vectors[:, 1], vectors[:, 0])
    lat = np.arcsin(np.clip(vectors[:, 2] / np.linalg.norm(vectors, axis=1), -1, 1))
    return hp.lonlat_to_healpix(lon * u.rad, lat * u.rad)


def _ray_exit_distance(dirs, center, shape, scale):
    """Path length in pc at which each ray leaves the array (slab method)."""
    v = dirs / scale                                  # voxel steps per pc of path
    with np.errstate(divide='ignore', invalid='ignore'):
        t_hi = (np.asarray(shape) - 1 - center) / v
        t_lo = (0 - center) / v
    t = np.where(v > 0, t_hi, t_lo)
    t[v == 0] = np.inf
    return t.min(axis=1)


def _skin_radii_one(dens, threshold_Av, origin, dirs, shape, scale, step,
                    max_radius, chunk_size):
    """Skin radius in pc along every direction from a single origin; NaN if never reached."""
    n_rays = len(dirs)
    radii = np.full(n_rays, np.nan)

    # Each ray only needs sampling until it leaves the box; for an elongated
    # cube most rays exit far sooner than the corner distance. Sorting by exit
    # distance keeps each chunk's sample grid as short as that chunk allows.
    t_exit = _ray_exit_distance(dirs, origin, shape, scale)
    if max_radius is not None:
        t_exit = np.minimum(t_exit, max_radius)
    t_exit = np.maximum(t_exit, 0.0)
    order = np.argsort(t_exit)

    # N(H) per magnitude of A_V, same conversion as plot_moment_maps() in dust_3d_functions.py
    N_H_PER_AV = 2.2e21 * u.cm**-2
    # mag of A_V per unit of (cm^-3 * pc), i.e. what one voxel-pc of density is worth
    AV_PER_CM3_PC = float(((1 * u.cm**-3 * u.pc).cgs / N_H_PER_AV).value)

    for lo in range(0, n_rays, chunk_size):
        sel = order[lo:lo + chunk_size]
        d = dirs[sel]                                    # (D, 3)
        s = np.arange(0.0, t_exit[sel].max() + step, step)   # path length, pc

        # ray samples in voxel coordinates: (3, D, T)
        coords = origin[None, :, None] + d[:, :, None] * s[None, None, :] / scale[None, :, None]
        coords = np.moveaxis(coords, 1, 0)

        n_H = map_coordinates(dens, coords.reshape(3, -1),
                              order=1, mode='constant', cval=0.0)
        n_H = n_H.reshape(len(d), len(s))                # cm^-3

        # cumulative column density along each ray (trapezoid rule), cm^-3 pc -> A_V
        col_Av = np.concatenate(
            [np.zeros((len(d), 1)),
             np.cumsum(0.5 * (n_H[:, :-1] + n_H[:, 1:]) * step, axis=1)],
            axis=1) * AV_PER_CM3_PC

        crossed = col_Av[:, -1] >= threshold_Av
        if not crossed.any():
            continue

        j = np.argmax(col_Av >= threshold_Av, axis=1)[crossed]  # first index at/above
        col_c = col_Av[crossed]
        prev = np.maximum(j - 1, 0)
        rows = np.arange(len(j))
        lo_val, hi_val = col_c[rows, prev], col_c[rows, j]

        # sub-step linear interpolation onto the threshold
        denom = hi_val - lo_val
        frac = np.where(denom > 0, (threshold_Av - lo_val) / np.where(denom > 0, denom, 1), 0.0)
        radii[sel[crossed]] = s[prev] + frac * (s[j] - s[prev])

    return radii


def radial_column_density_threshold(volume_density, threshold_Av=1.0, center=None,
                                    nside=16, pixel_scale_pc=1.0, step=None,
                                    max_radius=None, chunk_size=256,
                                    return_radii=False):
    """
    Casts rays outward from one or more origins in a 3D array and returns the
    points where the column density integrated from the origin first reaches a
    given extinction, i.e. the "skin" of the cloud at that A_V.

    The integral along each ray is

        N(s) = \\int_0^s n(r(s')) ds'        [cm^-3 * pc -> cm^-2]
        A_V  = N(s) / 2.2e21 cm^-2          [mag]

    with n sampled by trilinear interpolation and s in pc. That is the same
    n -> A_V conversion used by plot_moment_maps() in dust_3d_functions.py,
    just along a ray instead of down a cube axis. Density outside the array is
    treated as zero, as are NaNs, so a ray stops accumulating once it leaves
    the box.

    One ray is cast per HEALPix pixel, so the rays are equal-area on the sphere
    and any solid-angle average over them is an unweighted mean.

    With several origins -- points along a spine, say -- the returned skin is
    the union of the crossings from every origin. Note that this is a union of
    point clouds, not the outer boundary of the union of the enclosed regions:
    a crossing found from one origin can sit inside the region belonging to
    another. For the outer surface of the union, build the mask and take
    mask_interior_to_skin(..., region='skin').

    Parameters:
    -----------
    volume_density : np.ndarray
        3D array of volume densities, in cm^-3
    threshold_Av : float
        Extinction threshold in magnitudes of A_V. Default 1 mag.
    center : array_like, optional
        Origin voxel coordinates, either a single (x, y, z) or an (M, 3) array
        of origins to cast from. If None, the center of the box is used.
    nside : int
        HEALPix nside; the number of rays per origin is 12 * nside^2
        (nside=16 -> 3072).
    pixel_scale_pc : float or sequence of 3 floats
        Voxel size in pc along each axis. A sequence handles anisotropic
        voxels: rays are uniform in physical space, not index space. The
        default of 1 pc matches the `np.nansum(cube, axis=i) * u.pc`
        convention used for the moment maps.
    step : float, optional
        Integration step in pc. Defaults to half the smallest voxel dimension.
    max_radius : float, optional
        Maximum distance to search, in pc. If None, each ray runs to the wall
        of the box.
    chunk_size : int
        Number of rays sampled at once; caps peak memory.
    return_radii : bool
        If True, also return the per-pixel skin radii (see Returns).

    Returns:
    --------
    points : np.ndarray, shape (N, 3)
        Voxel coordinates (x, y, z), non-integer, where A_V = threshold_Av is
        reached, pooled over every origin. Rays that never reach it are
        omitted, so N <= M * 12 nside^2.
    radii : np.ndarray, only if return_radii=True
        Skin radius in pc indexed by HEALPix pixel, NaN where the ray never
        reached the threshold. Shape (12 nside^2,) for a single origin, or
        (M, 12 nside^2) for M origins. Pass this and the same `center` to
        mask_interior_to_skin() to avoid recomputing the geometry.
    """
    dens = np.nan_to_num(np.asarray(volume_density, dtype=float))   # once, not per origin
    shape = np.array(dens.shape)
    scale = np.broadcast_to(np.asarray(pixel_scale_pc, dtype=float), (3,))

    if center is None:
        center = shape // 2
    center = np.asarray(center, dtype=float)
    one_origin = center.ndim == 1
    origins = np.atleast_2d(center)                                 # (M, 3)

    if step is None:
        step = 0.5 * scale.min()

    dirs = healpix_directions(nside)

    radii = np.stack([_skin_radii_one(dens, threshold_Av, o, dirs, shape, scale,
                                      step, max_radius, chunk_size)
                      for o in origins])                            # (M, n_rays)

    points = np.concatenate(
        [o[None, :] + dirs[np.isfinite(r)] * r[np.isfinite(r), None] / scale[None, :]
         for o, r in zip(origins, radii)] + [np.empty((0, 3))])

    if one_origin:
        radii = radii[0]
    return (points, radii) if return_radii else points


def mask_interior_to_skin(skin, shape, center=None, pixel_scale_pc=1.0, nside=16,
                          region='interior', unreached='interior',
                          chunk_size=1_000_000):
    """
    Boolean mask of the voxels enclosed by the cloud skin, or of the skin itself.

    A voxel is interior when it is closer to the center than the skin is in
    that same direction, i.e. when less than threshold_Av of extinction sits
    between it and the center.

    Parameters:
    -----------
    skin : np.ndarray
        Either the (12 nside^2,) per-pixel radii from
        radial_column_density_threshold(..., return_radii=True) -- preferred,
        nothing is recomputed -- or the (N, 3) array of skin points, which is
        binned back onto HEALPix pixels. With several origins the radii array
        is required, shape (M, 12 nside^2) -- points alone are ambiguous,
        because there is no way to tell which origin a point belongs to.
    shape : tuple
        Shape of the original 3D array
    center : array_like, optional
        Origin voxel coordinates used for the rays: a single (x, y, z) or an
        (M, 3) array. If None, uses shape // 2. With M origins the mask is the
        union -- a voxel is interior when it is enclosed as seen from any one
        of them.
    pixel_scale_pc : float or sequence of 3 floats
        Voxel size in pc along each axis; must match the value used for the rays.
    nside : int
        HEALPix nside used for the rays.
    region : {'interior', 'skin'}
        'interior' marks every voxel enclosed by the surface. 'skin' marks only
        the surface itself: the outermost layer of the interior, one voxel
        thick. The shell is taken as the boundary of the interior region rather
        than by thresholding |distance - radius|, so it stays watertight where
        the surface runs oblique to the grid.
    unreached : {'interior', 'exterior'}
        What to do with directions where the threshold was never reached (no
        skin exists). 'interior' counts the whole ray as enclosed, which is the
        literal reading of "less than threshold_Av from the center";
        'exterior' excludes it, keeping the mask inside the measured surface.
        Prefer 'exterior' with several origins: an unreached direction runs to
        the wall of the box, and the union over many origins of those wedges
        fills the whole array. On a test filament where 1.6% of rays crossed,
        'interior' returned every one of the 2.6M voxels and 'exterior'
        returned 310k.
    chunk_size : int
        Number of voxels processed at once; caps peak memory.

    Returns:
    --------
    mask : np.ndarray, dtype=bool, shape=shape
        Boolean array where the selected region is True, everything else False
    """
    if region not in ('interior', 'skin'):
        raise ValueError(f"region must be 'interior' or 'skin', got {region!r}")

    shape = tuple(int(s) for s in shape)
    scale = np.broadcast_to(np.asarray(pixel_scale_pc, dtype=float), (3,))

    if center is None:
        center = np.array(shape) // 2
    origins = np.atleast_2d(np.asarray(center, dtype=float))        # (M, 3)

    skin = np.asarray(skin, dtype=float)
    npix = 12 * nside**2
    if skin.ndim == 2 and skin.shape[1] == 3 and skin.shape[1] != npix:
        if len(origins) > 1:
            raise ValueError('with several origins, pass the (M, npix) radii array '
                             'from return_radii=True -- points alone cannot be '
                             'attributed to an origin')
        # (N, 3) points -> per-pixel radii, the same geometry the rays already had
        off = (skin - origins[0]) * scale
        radii = np.full((1, npix), np.nan)
        radii[0, directions_to_healpix(off, nside)] = np.linalg.norm(off, axis=1)
    else:
        radii = skin.reshape(-1, npix)

    if len(radii) != len(origins):
        raise ValueError(f'{len(radii)} radii rows but {len(origins)} origins')

    # unreached directions: +inf reaches the box wall, -inf excludes everything
    radii = np.where(np.isnan(radii),
                     np.inf if unreached == 'interior' else -np.inf,
                     radii)
    reach = radii.max(axis=1)                        # furthest skin per origin

    mask = np.zeros(shape, dtype=bool)
    flat = mask.reshape(-1)
    for lo in range(0, flat.size, chunk_size):
        idx = np.arange(lo, min(lo + chunk_size, flat.size))
        vox = np.stack(np.unravel_index(idx, shape), axis=1) * scale
        hit = np.zeros(len(idx), dtype=bool)

        for origin, rad, far in zip(origins, radii, reach):
            # union: only voxels not already enclosed, and only those close
            # enough to this origin to stand a chance
            off = vox[~hit] - origin * scale
            dist = np.linalg.norm(off, axis=1)
            todo = np.flatnonzero(~hit)[dist <= far]
            if not len(todo):
                continue
            off, dist = vox[todo] - origin * scale, dist[dist <= far]
            at_origin = dist == 0
            off[at_origin] = [0.0, 0.0, 1.0]          # any direction; overridden below
            close = dist < rad[directions_to_healpix(off, nside)]
            close[at_origin] = True
            hit[todo[close]] = True

        flat[idx] = hit

    if region == 'skin':
        # outermost layer of the interior. border_value=1 keeps the array walls
        # from counting as surface where the interior runs off the edge of the
        # box -- there is no measured skin there.
        mask &= ~binary_erosion(mask, border_value=1)

    return mask


def skin_mesh(mask, pixel_scale_pc=1.0, step_size=2):
    """
    Triangulate the surface of a skin mask with marching cubes.

    Shared by the plotly and matplotlib renderers so both draw the identical
    surface.

    Parameters:
    -----------
    mask : np.ndarray, dtype=bool
        3D mask from mask_interior_to_skin().
    pixel_scale_pc : float or sequence of 3 floats
        Voxel size in pc along each axis; the returned vertices are in pc.
    step_size : int
        Marching-cubes stride. This is decimation, not smoothing -- keep it at
        1 to measure off the mesh rather than look at it.

    Returns:
    --------
    verts : np.ndarray, shape (V, 3)   vertex positions in pc
    faces : np.ndarray, shape (F, 3)   vertex indices per triangle
    """
    from skimage.measure import marching_cubes

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError('mask is empty -- nothing to draw')
    if mask.all():
        raise ValueError(
            'mask is the entire array, so it has no surface. This is what '
            "unreached='interior' does when many rays never reach the "
            'threshold: each unreached direction runs to the wall of the box, '
            'and over several origins those wedges fill everything. Rebuild '
            "the mask with mask_interior_to_skin(..., unreached='exterior'), "
            'or check what fraction of rays crossed with np.isfinite(radii).mean().')

    scale = np.broadcast_to(np.asarray(pixel_scale_pc, dtype=float), (3,))

    # Crop to the mask's own bounding box before meshing: on a 1182x55x40 cube
    # the occupied part is usually a small fraction of the array. Pad the crop
    # with one empty voxel so the volume always spans both sides of level=0.5 --
    # otherwise a solidly filled bounding box has range [1, 1] and marching
    # cubes refuses it -- and so a surface running into the wall still closes.
    nz = np.nonzero(mask)
    lo = np.array([i.min() for i in nz])
    sub = mask[tuple(slice(l, i.max() + 1) for l, i in zip(lo, nz))]
    sub = np.pad(sub, 1, constant_values=False)

    verts, faces, _, _ = marching_cubes(sub.astype(float), level=0.5,
                                        step_size=step_size, spacing=tuple(scale))
    return verts + (lo - 1) * scale, faces           # back to whole-cube pc coords


def _mesh_intensity(verts, mask_shape, scale, color_by, center, labels):
    """Per-vertex colour values and a colorbar label, for the mesh renderers."""
    if not isinstance(color_by, str):
        return np.asarray(color_by, dtype=float), ''

    if color_by == 'distance':
        if center is None:
            center = np.array(mask_shape) // 2
        origins = np.atleast_2d(np.asarray(center, dtype=float)) * scale
        return np.min([np.linalg.norm(verts - o, axis=1) for o in origins], axis=0), 'r [pc]'

    i = 'dbl'.index(color_by)
    return verts[:, i], labels[i]


def plot_skin_mesh3d_mpl(mask, ax=None, pixel_scale_pc=1.0, step_size=2, points=None,
                         center=None, color_by='distance', cmap='magma', alpha=1.0,
                         labels=('d [pc]', 'b [pc]', 'l [pc]'), title=None,
                         colorbar=True, edgecolor='none', figsize=(9, 7), dpi=150,
                         **kws):
    """
    The same surface as plot_skin_mesh3d(), drawn into a matplotlib 3D axes.

    Use this when the figure has to live in a paper or alongside the other
    matplotlib panels; use the plotly version to turn the cloud over and look
    at it. Matplotlib's 3D engine sorts whole polygons by depth rather than per
    pixel, so a concave surface can show artefacts where triangles interleave
    -- the closed skin of a single cloud is usually fine, a multi-origin union
    less so.

    Parameters:
    -----------
    mask : np.ndarray, dtype=bool
        3D mask from mask_interior_to_skin().
    ax : matplotlib 3D Axes, optional
        Draw into an existing axes, e.g. one panel of a larger grid. Must have
        been created with projection='3d'. A new figure is made if omitted.
    pixel_scale_pc, step_size, points, center, color_by
        As for plot_skin_mesh3d().
    cmap : str
        Colormap for the surface.
    alpha : float
        Face opacity. Below 1 matplotlib's depth sorting gets noticeably worse.
    labels : sequence of 3 str
        Axis titles, in array-axis order (axis 0 = d, axis 1 = b, axis 2 = l).
    colorbar : bool
        Attach a colorbar for the surface colouring.
    edgecolor : color
        Triangle edges. 'none' is smooth; a faint grey shows the mesh density.
    figsize, dpi
        Used only when `ax` is None.

    Returns:
    --------
    ax : the 3D Axes that was drawn into
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts, faces = skin_mesh(mask, pixel_scale_pc, step_size)
    scale = np.broadcast_to(np.asarray(pixel_scale_pc, dtype=float), (3,))
    intensity, cbar_title = _mesh_intensity(verts, mask.shape, scale, color_by,
                                            center, labels)

    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.get_figure()

    tris = Poly3DCollection(verts[faces], alpha=alpha, edgecolor=edgecolor, **kws)
    tris.set_cmap(cmap)
    tris.set_array(intensity[faces].mean(axis=1))    # per-face, from its vertices
    ax.add_collection3d(tris)

    if points is not None:
        p = np.asarray(points, dtype=float) * scale
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1.5, color='black',
                   depthshade=False, label='ray crossings')

    # add_collection3d does not autoscale, and the default box is a cube, which
    # would squash an elongated cloud into looking round
    span = np.ptp(verts, axis=0)
    for setlim, lo, hi in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim),
                              verts.min(axis=0), verts.max(axis=0)):
        setlim(lo, hi)
    ax.set_box_aspect(span)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(tris, ax=ax, fraction=0.03, pad=0.08).set_label(cbar_title)

    return ax


def plot_skin_mesh3d(mask, pixel_scale_pc=1.0, step_size=2, points=None,
                     center=None, color_by='distance', colorscale='Magma',
                     opacity=0.9, labels=('d [pc]', 'b [pc]', 'l [pc]'),
                     title=None, **kws):
    """
    Interactive 3D surface of a cloud-skin mask, as a plotly Mesh3d.

    Marching cubes turns the boolean mask into a triangulation and plotly
    renders it with real depth sorting, which matplotlib's plot_trisurf does
    not do -- concave parts of the surface come out right.

    Axes are in pc, not voxels, and the scene uses aspectmode='data', so an
    elongated cube keeps its true proportions instead of being squashed into
    plotly's default cube.

    Parameters:
    -----------
    mask : np.ndarray, dtype=bool
        3D mask from mask_interior_to_skin(). Either the filled interior or
        the one-voxel shell works; the interior gives a cleaner surface,
        because marching cubes is finding its boundary anyway.
    pixel_scale_pc : float or sequence of 3 floats
        Voxel size in pc along each axis, as passed to the ray caster.
    step_size : int
        Marching-cubes stride. 1 is the full-resolution surface, 2 roughly
        quarters the face count for a small loss of fidelity. This is
        decimation, not smoothing -- keep it at 1 if you intend to measure
        anything off the mesh rather than look at it.
    points : np.ndarray, shape (N, 3), optional
        Skin points from radial_column_density_threshold(), overlaid as a
        scatter so the reconstructed surface can be checked against the raw
        ray crossings.
    center : array_like, optional
        Origin(s) for `color_by='distance'`, in voxel coordinates. Defaults to
        the box center. An (M, 3) array colours by distance to the nearest.
    color_by : {'distance', 'd', 'l', 'b'} or np.ndarray
        Surface colouring: radial distance from `center`, one of the three
        coordinates, or an explicit per-vertex array.
    opacity : float
        Mesh opacity. Below ~0.5 the interior shows through, which is useful
        when overlaying `points`.
    labels : sequence of 3 str
        Axis titles, in array-axis order.
    title : str, optional
        Figure title.
    **kws
        Passed through to go.Mesh3d().

    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from skimage.measure import marching_cubes

    verts, faces = skin_mesh(mask, pixel_scale_pc, step_size)
    scale = np.broadcast_to(np.asarray(pixel_scale_pc, dtype=float), (3,))
    intensity, cbar_title = _mesh_intensity(verts, mask.shape, scale, color_by,
                                            center, labels)

    fig = go.Figure(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                              i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                              intensity=intensity, colorscale=colorscale,
                              opacity=opacity, name='skin',
                              colorbar=dict(title=cbar_title), **kws))

    if points is not None:
        p = np.asarray(points, dtype=float) * scale
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers',
                                   marker=dict(size=1.5, color='black'),
                                   name='ray crossings'))

    fig.update_scenes(aspectmode='data', xaxis_title=labels[0],
                      yaxis_title=labels[1], zaxis_title=labels[2])
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=30 if title else 0, b=0))
    return fig


def plot_moments_and_mask(cube, mask, coords, which='', how='depth', cmp='magma',
                          threshold_Av=1.0, outline=True, figsize=(15, 8), dpi=150):
    """
    2x3 grid: zeroth moment maps on top, the skin mask below, same projections.

    Column order and orientation follow plot_moment_maps() exactly -- l-b,
    l-d, d-b, with axis 0 of the cube read as distance, axis 1 as latitude and
    axis 2 as longitude -- so the panels line up with the figures already in
    the notebook.

    Each row is on its own shared colour scale, so the three projections in a
    row are comparable with each other. (plot_moment_maps lets contourf pick
    levels per panel, which they are not.)

    Parameters:
    -----------
    cube : np.ndarray
        Volume density in cm^-3.
    mask : np.ndarray, dtype=bool
        Mask from mask_interior_to_skin(), same shape as `cube`.
    coords : sequence of 6
        [l_min, l_max, b_min, b_max, d_min, d_max], as handed to
        plot_moment_maps().
    which : str
        Figure title, e.g. the cloud name.
    how : {'depth', 'silhouette', 'Av'}
        What the bottom row shows. 'depth' is the number of enclosed voxels
        along each sightline, i.e. how deep the mask is; 'silhouette' is its
        binary footprint; 'Av' is the moment map recomputed from the masked
        cube only, in the same units as the top row, so the two rows can be
        differenced by eye to see how much extinction the skin encloses.
    cmp : str
        Colormap for the top row. The bottom row uses it too, except for
        'silhouette', which is drawn binary.
    threshold_Av : float
        Contour level drawn on the top row, matching the value the skin was
        built at.
    outline : bool
        Trace the mask's silhouette over the top row, so the enclosed region
        can be seen against the extinction it came from.
    figsize, dpi : passed to plt.subplots

    Returns:
    --------
    fig, axs : the figure and its (2, 3) array of axes
    """
    cube = np.asarray(cube, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != cube.shape:
        raise ValueError(f'mask shape {mask.shape} != cube shape {cube.shape}')
    coords = np.asarray(coords)

    to_Av = lambda arr, ax: ((np.nansum(arr * u.cm**-3, axis=ax) * u.pc).cgs
                             / (2.2e21 * u.cm**-2)).value

    # (axis, extent, xlabel, ylabel, needs transpose) -- straight from plot_moment_maps
    panels = [(0, coords[:4], 'Galactic Longitude [deg]', 'Galactic Latitude [deg]', False),
              (1, coords[[0, 1, 4, 5]], 'Galactic Longitude [deg]', 'Heliocentric Distance [pc]', False),
              (2, coords[[4, 5, 2, 3]], 'Heliocentric Distance [pc]', 'Galactic Latitude [deg]', True)]

    top = [to_Av(cube, ax) for ax, *_ in panels]
    if how == 'depth':
        bot = [mask.sum(axis=ax) for ax, *_ in panels]
        blabel, bnorm = 'enclosed voxels along sightline', None
    elif how == 'silhouette':
        bot = [mask.any(axis=ax) for ax, *_ in panels]
        blabel, bnorm = 'inside skin', None
    elif how == 'Av':
        bot = [to_Av(np.where(mask, cube, 0.0), ax) for ax, *_ in panels]
        blabel, bnorm = f'$A_V$ enclosed [mag]', 'log'
    else:
        raise ValueError("how must be 'depth', 'silhouette' or 'Av'")

    fig, axs = plt.subplots(2, 3, figsize=figsize, dpi=dpi, layout='constrained')
    col_titles = ('$l$-$b$', '$l$-$d$', '$d$-$b$')

    for row, (imgs, label, norm) in enumerate([(top, '$A_V$ [mag]', 'log'),
                                               (bot, blabel, bnorm)]):
        finite = np.concatenate([im[np.isfinite(im)].ravel() for im in imgs])
        pos = finite[finite > 0]
        vmin = pos.min() if norm == 'log' and len(pos) else (finite.min() if len(finite) else 0)
        vmax = finite.max() if len(finite) else 1

        for col, (im, (ax_i, extent, xlbl, ylbl, tr)) in enumerate(zip(imgs, panels)):
            ax = axs[row, col]
            arr = im.transpose() if tr else im
            m = ax.imshow(arr, extent=extent, origin='lower', aspect='auto',
                          cmap=cmp, norm=norm, vmin=vmin, vmax=vmax)
            # under a log norm the empty (zero) region is "bad" and renders as
            # white, which reads as data next to the dark panels above it
            ax.set_facecolor(plt.get_cmap(cmp)(0.0))
            ax.set_xlabel(xlbl)
            ax.set_ylabel(ylbl)
            if row == 0:
                ax.set_title(col_titles[col])

            if row == 0:
                ax.contour(arr, extent=extent, levels=[threshold_Av],
                           colors='white', linewidths=1.0)
                if outline:
                    sil = mask.any(axis=ax_i)
                    ax.contour(sil.transpose() if tr else sil, extent=extent,
                               levels=[0.5], colors='cyan', linewidths=1.0)

            if col == 2:
                fig.colorbar(m, ax=axs[row, :], fraction=0.02, pad=0.01).set_label(label)

    fig.suptitle(f'{which}    top: zeroth moment, white $A_V$ = {threshold_Av:g} mag'
                 + (', cyan skin outline' if outline else '')
                 + f'    bottom: skin mask ({how})')
    return fig, axs


def plot_contour(axs, cube, coords, val, clr):
    im0 = ((np.nansum(cube*u.cm**-3, axis=0)*u.pc).cgs/(2.2e21*u.cm**-2)).value
    im1 = ((np.nansum(cube*u.cm**-3, axis=1)*u.pc).cgs/(2.2e21*u.cm**-2)).value
    im2 = ((np.nansum(cube*u.cm**-3, axis=2)*u.pc).cgs/(2.2e21*u.cm**-2)).value

    kws2 = {'levels': val, 'colors': clr, 'linewidths': 0.75,
            'linestyles': ['solid', '--', ':', 'solid']}

    axs[0].contour(im0, extent=coords[:4], **kws2)
    axs[1].contour(im1, extent=coords[[0, 1, 4, 5]], **kws2)
    axs[2].contour(im2, extent=coords[2:], **kws2)


def plot_densty_vsr(ax, dperp, densty, rbins, ylbl, ttl, lbl, color='red'):
    # References `name`, which is undefined in this scope -- raises NameError if hit.
    df = bin_by_r(dperp, densty, rbins)
    mins = find_peaks(-df['rho'])

    ax.plot(df['d'], df['rho'], color=color, marker='.')
    ax.scatter(df['d'].iloc[mins[0]], df['rho'].iloc[mins[0]], color='orange', marker='+', label=name)

    ax.loglog(),
    ax.set_xlabel(r'$d_\perp\ [pc]$'),
    ax.set_xlim(1,)
    ax.set_ylabel(lbl)
    ax.set_title(ttl)

    return df['d'].to_numpy(), df['rho'].to_numpy()


def check_thres(i, source_array, cloud_catalog, r, thresh):
    cube, coords = get_cloud_info(i, source_array, cloud_catalog, r)

    full_cube = get_cloud_info(i, source_array, cloud_catalog, 'full')[0]
    fullc_Av = ((np.nansum(full_cube*u.cm**-3, axis=0)*u.pc).cgs/(2.2e21*u.cm**-2)).value

    pta, veca, dperp, dparl, L1da = get_spine(i, source_array, cloud_catalog, r)

    p1a_xyz, p2a_xyz = line2pts(pta, veca)
    p1a_lbd = np.array(xyz_2_sph(p1a_xyz))
    p2a_lbd = np.array(xyz_2_sph(p2a_xyz))

    pta_lbd, veca_lbd = pts2line(p1a_lbd, p2a_lbd)

    df = get_99th_percs(i, source_array, cloud_catalog)

    masked = cube * (cube > thresh)

    axs = plot_moment_maps(masked, coords, '{}'.format(thresh))
    axs[0].contour(fullc_Av, extent=coords[:4], levels=[1], colors='orange')

    plt.show()


def histogram(i, source_array, cloud_catalog, drange='by r', cmapping='sane'):
    cloud_dict = ha(i, source_array, cloud_catalog, drange)

    full_cube = get_cloud_info(i, source_array, cloud_catalog, 'full')[0]
    fullc_Av = ((np.nansum(full_cube*u.cm**-3, axis=0)*u.pc).cgs/(2.2e21*u.cm**-2)).value

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # histogram of distances
    h = axs[0].hist(np.ravel(cloud_dict['dist_cube']), bins=100,
                    alpha=0.5,
                    weights=np.ravel(cloud_dict['dust_cube']))
    fit = fttr(gmod, h[1][:-1], h[0],)

    axs[0].set_ylabel('N (pixels)')
    axs[0].set_title('distances (pc)')

    axs[1].hist(np.ravel(cloud_dict['dust_cube']), bins=10**np.linspace(-2, 3),
                alpha=0.5)

    axs[1].set_title(r'densities ($\mathrm{cm}^{-3}$)')
    axs[1].loglog()

    fig.suptitle(names[i])
    plt.show()


def rho_spinal_profiles(i, source_array, cloud_catalog, drange='by r', plt_='new', color='black', lbl='by r'):
    # BROKEN AS PORTED: curve_fit raises "`ydata` must not be empty!". b_parl is a
    # signed projection in pc, so it spans negative values and the log-spaced rbins
    # below cannot bin it. Left as-is to match 17_.ipynb cell 44.
    if lbl == 'by r':
        lbl = f'r={round(drs[i], 2)}'

    cloud_dict = ha(i, source_array, cloud_catalog, drange)

    rmax = np.nanmax(cloud_dict['r_parl'])
    px_scl = np.log10(2*0.125*u.deg.to(u.rad)*ds[i])
    rbins = 10**np.arange(px_scl, np.log10(rmax)+0.2, 0.1)

    df = bin_by_r(cloud_dict['r_parl'], cloud_dict['dust_cube'], rbins)
    mins = find_peaks(-df['rho'])

    rho0, r0, alpha = np.around((curve_fit(plummer, df['d'], df['rho'], p0=[10, 10, 3], maxfev=5000))[0], 2)

    # -------------------------------------------------------------------

    if plt_ == 'new':
        plt.figure()
        ax = plt.gca()
    else:
        ax = plt_

    s = 100
    ax.scatter(df['d'], df['rho'], color=color, marker='.', label=lbl, s=s)
    ax.scatter(df['d'].iloc[mins[0]], df['rho'].iloc[mins[0]], color=color, marker='+', s=s)
    ax.plot(df['d'], plummer(df['d'], rho0, r0, alpha), color=color)

    err_sum = np.nansum(np.abs(plummer(df['d'], rho0, r0, alpha)-df['rho']))/np.nansum(df['rho'])

    ax.loglog()
    ax.set_xlabel(r'$d_\parallel\ [pc]$')
    ax.set_ylabel(r'$n\ (cm^{-3})$')
    ax.set_xlim(1, rmax+5)
    ax.set_title(names[i])

    print('rho0', rho0, 'r0', r0, 'alpha', alpha)

    return plt.gca(), [rho0, r0, alpha]


def rho_radial_profiles(i, source_array, cloud_catalog, drange='by r', plt_='new', color='black', lbl='by r',
                        tube_r='px', n_jobs=1, parl_origin='edge', cloud_dict=None, perp_slab='L',
                        dmask=None, perp_pad=0.5):
    # Backward-compatible wrapper: compute + plot in one call, same signature and
    # return as before. For parallel runs call rho_radial_profiles_compute across
    # clouds under a Parallel(...), then rho_radial_profiles_plot on each result.
    result = rho_radial_profiles_compute(i, source_array, cloud_catalog, drange, tube_r, n_jobs,
                                         parl_origin, cloud_dict=cloud_dict, perp_slab=perp_slab,
                                         dmask=dmask, perp_pad=perp_pad)
    return rho_radial_profiles_plot(result, plt_, color, lbl)


def fit_spines_over_dranges(i, source_array, cloud_catalog, dranges, n='none', n_jobs=-1,
                            verbose=False):
    """
    Fit the spine of one cloud over several distance ranges.

    The spine is fit to the 99th-percentile pixels of whatever subcube `drange`
    cuts out, so it is not independent of that choice -- widen the range and the
    fit sees more foreground/background structure. Running the same cloud over a
    list of ranges is how that sensitivity gets measured rather than assumed.

    `dranges` is a list of half-widths in pc (or 'by r' / 'full'), e.g.
    r*np.array([0.5, 1, 2, 3, 5]) in the style of 22_finalizing_for_paper cell 28.
    A bare scalar is accepted and comes back as a one-element list.

    Deliberately serial over the ranges, with n_jobs=-1 handed down to each fit:
    the Monte-Carlo spine fit already parallelizes over its seeds, on a small
    array of bright pixels (see pick_best_line). Parallelizing out here instead
    would ship `source_array` -- the whole several-GB dust map -- to every
    worker.

    Returns a list of ha() dicts, one per range, each tagged with its 'drange'.
    Feed it straight to spine_props() or to plot_cloud_summary(cloud_dict=...).
    """
    out = []

    for drange in as_list(dranges):
        cloud_dict = ha(i, source_array, cloud_catalog, drange, n=n, n_jobs=n_jobs)
        out.append(cloud_dict)

        if verbose:
            print('%-12s L1d = %6.3f   d = %.0f-%.0f pc'
                  % (drange_lbl(drange), cloud_dict['L1d'],
                     cloud_dict['coords'][4], cloud_dict['coords'][5]))

    return out


def spine_props(cloud_dicts):
    """
    One row per fitted spine: where it sits, which way it points, how well it fit.

    Built for the output of fit_spines_over_dranges, so a cloud's spine can be
    read against the distance range it was fit over.

    Columns: the range and the distances it cut (d_lo, d_hi); the spine midpoint
    (l, b, d) and direction (dl, db, dd) as get_spine returns them; spine_len,
    the separation of the two endpoints in pc; parl_extent, how far the cloud's
    pixels actually reach along the spine (max - min d_parl), which is the length
    the superGaussian's L is fit to; and L1d, the Av-weighted mean |d_perp| the
    fit minimized -- lower is a tighter spine, but it is not comparable across
    clouds of different size.
    """
    rows = []

    for cd in cloud_dicts:
        (l, b, d), (dl, db, dd) = cd['spine']
        p1, p2 = cd['spine_xyz']

        rows.append({'drange': cd.get('drange', np.nan),
                     'd_lo': cd['coords'][4], 'd_hi': cd['coords'][5],
                     'l': l, 'b': b, 'd': d, 'dl': dl, 'db': db, 'dd': dd,
                     'spine_len': np.linalg.norm(p2-p1),
                     'parl_extent': np.nanmax(cd['r_parl'])-np.nanmin(cd['r_parl']),
                     'L1d': cd['L1d']})

    return pd.DataFrame(rows)


def plot_model_params(param='eta', values=(1, 2, 4, 8, 30),
               rho0=100., r0=2., alpha=2.,
               z0=0., L=10., eta=4.,
               R_T=10.,
               npts=500, cmap='viridis'):
    # Sweep one parameter of the finite-length cylinder model and show what it does:
    #   rho(r,z) = rho0 / [1 + (r/r0)^2]^alpha  *  exp[-(|z - z0|/(L/2))^eta]
    # From 23_incorporating_L_into_the_model.ipynb cell 7.

    #dictionary of default values
    base = dict(rho0=rho0, r0=r0, alpha=alpha, R_T=R_T, L=L, eta=eta, z0=z0)

    #check whether the thing to toggle is in dict_
    if param not in base:
        raise KeyError('param must be one of %s' % list(base))

    #replace the items under that parameter with toggle values
    vars_ = [dict(base, **{param: v}) for v in values]

    pad = 2.0
    rmax = pad*max(p['R_T'] for p in vars_)
    zmax = pad*max(p['L']/2 for p in vars_) + max(abs(p['z0']) for p in vars_)

    rs = np.logspace(-2, np.log10(rmax), npts)
    zs = np.linspace(-zmax, zmax, npts)

    colrs = plt.get_cmap(cmap)(np.linspace(0.1, 0.85, len(values)))

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2),)

    for p, c, v in zip(vars_, colrs, values):
        lbl = '%s = %g' % (param, v)

        # profile along r_perp -- dotted beyond the truncation radius R_T
        inside = rs <= p['R_T']
        axs[0].plot(rs[inside],  plummer(rs[inside],  p['rho0'], p['r0'], p['alpha']), color=c, label=lbl)
        axs[0].plot(rs[~inside], plummer(rs[~inside], p['rho0'], p['r0'], p['alpha']), color=c, ls=':', alpha=.5)

        # profile along r_parl
        axs[1].plot(zs, p['rho0']*superGaussian(zs, p['z0'], p['L'], p['eta']), color=c, label=lbl)

        axs[0].axvline(p['r0'], color='gray', lw=.8)

    axs[0].loglog()
    axs[0].set_xlabel(r'$r\ [pc]$')
    axs[0].set_ylabel(r'$\rho\ (cm^{-3})$')
    axs[0].set_title('radial cut at $z=z_0$')

    axs[1].set_xlabel(r'$z\ [pc]$')
    axs[1].set_title(r'axial cut at $r=0$')

    for a in axs[:2]:
        a.legend(fontsize=10, frameon=False)

    # 2D map for the middle variant of the sweep
    pm = vars_[len(vars_)//2]
    R, Z = np.meshgrid(np.linspace(-pm['R_T'], pm['R_T'], npts), zs)

    rho2d = mif_cylinder(np.abs(R), Z, pm)
    rho2d[np.abs(R) > pm['R_T']] = np.nan

    im = axs[2].pcolormesh(R, Z, rho2d, cmap='magma', shading='auto',
                           norm=colors.LogNorm(vmin=max(rho2d[rho2d > 0].min(), pm['rho0']*1e-4),
                                               vmax=pm['rho0']))

    fig.colorbar(im, ax=axs[2], label=r'$\rho\ (cm^{-3})$')

    axs[2].set_xlabel(r'$r\ [pc]$')
    axs[2].set_ylabel(r'$z\ [pc]$')
    axs[2].set_title('%s = %g' % (param, values[len(vars_)//2]))
    axs[2].set_aspect('equal')

    plt.tight_layout()
    return fig, axs


def Sig_radial_profiles(i, source_array, cloud_catalog, drange='by r', plt_='new', color='black'):
    # Never returns anything -- as in the source notebook. Depends on global `names`.
    cloud_dict = ha(i, source_array, cloud_catalog, drange)

    rbins = 10**np.arange(-0.5, 2, 0.1)

    msk = 1
    dperps = [np.nanmin(cloud_dict['dist_cube'], axis=n) for n in range(3)]  # lb,ld,bd projections
    Sigs = [np.nansum(cloud_dict['dust_cube'], axis=n) for n in range(3)]

    if type(plt_) == str:
        fig, axs = plt.subplots(1, 3, figsize=(15, 7), sharey=True, sharex=True)
        fig.suptitle(names[i])


# ----------------------------------------------------------------------------
# from choosing_distances_from_healpix.py: the pre-HEALPix leaf selection, window widening, spine angles and cloud masses.
# ----------------------------------------------------------------------------
NICEST = NICEST_DIR

zeroth_moment = lambda cube, dx=1, fac=1: np.nansum(cube, axis=0) * dx * fac

peak_x        = lambda x: ds[x.get_peak(subtree=True)[0][0]]


def load_spectral_and_nicest(table1, cubes):
    """CO+HI and NICEST comparison maps, regridded onto each dust cube."""
    hi4pi, dht = fits.open(DATA + 'hi4piDHT_720deg.fits')
    line_ax0s, line_dxs = [360, -50, -100], [-0.125, 0.125, 1.3]

    def sub(big, i):
        co = table1.iloc[i][1:-1].to_numpy().astype('float')
        lo, li, bo, bi, do, di = co
        xo, xi = coord_idx([lo, li], line_dxs[0], line_ax0s[0])
        yo, yi = coord_idx([bo, bi], line_dxs[1], line_ax0s[1])
        zo, zi = coord_idx([do, di], line_dxs[2], line_ax0s[2])
        return big[zo:zi, yo:yi, xo:xi]

    NHs_spectral, NHs_nicest = [], []
    for i in range(len(table1)):
        co_cb, hi_cb = sub(dht.data, i), sub(hi4pi.data, i)
        NHs_spectral.append(2 * zeroth_moment(co_cb, dx=1.3, fac=1.97e21)
                            + zeroth_moment(hi_cb, dx=1.3, fac=1.82e18))
        name = table1['Name'].iloc[i]
        NHs_nicest.append(fits.open(f'{NICEST}/{ALIASES.get(name, name)}_nicest.fits')[0].data)

    tgt = [zeroth_moment(cb).shape for cb in cubes]
    spec_e = [zoom(NHs_spectral[i], (tgt[i][0] / NHs_spectral[i].shape[0],
                                     tgt[i][1] / NHs_spectral[i].shape[1]), order=1)
              for i in range(len(table1))]
    nic_e  = [zoom(NHs_nicest[i] * 8.9 * 2.2e21,
                   (tgt[i][0] / NHs_nicest[i].shape[0],
                    tgt[i][1] / NHs_nicest[i].shape[1]), order=1)
              for i in range(len(table1))]
    return spec_e, nic_e


def sample_spread(sprof, d_mean, min_value=None, min_npix=30, min_delta=20):
    """Rerun the whole selection on each posterior sample.

    Returns (lo, hi, frac, ok): arrays of the selected leaf's near edge, far
    edge and mass fraction, one entry per sample, plus how many samples yielded
    a leaf at all. Each sample gets its OWN robust min_value when min_value is
    None -- the threshold is part of the method, so freezing it at the mean
    map's value would understate the spread.
    """
    los, his, frs = [], [], []
    for pr in sprof:
        _, j, fr, sp, _, _ = select_leaf(pr, d_mean, min_value, min_npix, min_delta)
        if j is None:
            continue
        los.append(sp[j][0]); his.append(sp[j][1]); frs.append(fr[j])
    return np.array(los), np.array(his), np.array(frs), len(los)


def get_best_leaf(i, table1, cubes, masses_per_slice, NHs_ednhfr, baseline,
                  suffix='_choosing_distances_healpix', save=True, show=False,
                  min_npix=None, min_delta=20, min_value=None, sprof=None):
    """Same figure as the notebook's get_best_leaf, on the HEALPix-derived cube.

    Leaf selection uses the dust alone -- no comparison to the CO+HI or NICEST
    baseline. Two criteria, in order:
      1) the leaf whose distance span brackets the catalog d_mean
      2) the leaf holding the largest share of the sightline's total mass
    Fractions are of the mass summed over the WHOLE profile, not just the mass
    the dendrogram assigned to leaves, so they say how much of the aperture a
    leaf actually accounts for. When the two disagree, d_mean wins; if several
    leaves bracket d_mean (nested or blended peaks), the most massive of those
    breaks the tie. `baseline` is now display-only: it draws the comparison
    contours and no longer enters the choice.

    min_value=None picks a per-cloud floor with robust_min_value(); pass a
    number (or -np.inf) to override.
    """
    mps = masses_per_slice[i].value
    d_mean = float(table1.iloc[i]['d_mean(pc)'])
    dend, j, fracs, spans, holds_dmean, mv = select_leaf(
        mps, d_mean, min_value=min_value, min_npix=min_npix, min_delta=min_delta)
    if j is None:
        print(f'  !! {table1["Name"].iloc[i]}: no leaves survive min_value={mv:.1f}')
        return None
    p = dend.plotter()
    p.set_custom_positions(peak_x)

    N = len(dend.leaves)
    NH_vals = np.linspace(20.5, 22.5, 50)
    coords = table1.iloc[i][['l_max(deg)', 'l_min(deg)',
                             'b_min(deg)', 'b_max(deg)']].to_numpy()
    best_leaf = [leaf.idx for leaf in dend.leaves][j]

    fig = plt.figure(figsize=(max(15, N * 3), 7), layout='constrained')
    subfigs = fig.subfigures(2, 1)
    fig.suptitle(table1['Name'].iloc[i] + '  (native HEALPix)')

    row1 = subfigs[0].subplots(1, 4)
    row2 = np.atleast_1d(subfigs[1].subplots(1, N))

    for n, d in enumerate(dend.leaves):
        this_leaf = zeroth_moment(cubes[i] * d.get_mask()[:, np.newaxis, np.newaxis],
                                  1, PC_CM)
        row2[n].contourf(np.log10(this_leaf), levels=NH_vals, extent=coords, cmap='cubehelix')
        row2[n].contour(np.log10(baseline[i]), levels=NH_vals[::10], extent=coords,
                        colors='black', linewidths=0.75, linestyles=['solid', '--', ':'])
        row2[n].set_title(f'{spans[n][0]:.0f}-{spans[n][1]:.0f} pc\n'
                          f'{100 * fracs[n]:.1f}% of mass' + ('  *' if n == j else ''),
                          fontsize=10, color=('red' if n == j else 'black'))

    if sprof is not None:
        for k, pr in enumerate(sprof):
            row1[0].plot(ds, pr, color='tab:orange', lw=0.6, alpha=0.7,
                         label='posterior samples' if k == 0 else None)
    row1[0].plot(ds, mps, color='black', lw=1.4, label='mean map', zorder=5)
    p.plot_tree(row1[0], color='grey')
    row1[0].axvline(d_mean, color='tab:blue', linestyle=':', label='catalog $d_{mean}$')
    row1[0].axhline(mv, color='tab:green', linestyle='--', lw=0.8,
                    label=f'min_value = {mv:.1f}')
    row1[0].legend(fontsize=8)
    for d in dend.leaves:
        col = 'red' if d.idx == best_leaf else 'grey'
        row1[1].plot(ds, d.get_mask() * mps, label=d.idx, color=col)
    row1[1].axvline(d_mean, color='tab:blue', linestyle=':')

    span_txt = ''
    if sprof is not None:
        lo, hi, fr, nok = sample_spread(sprof, d_mean, min_value, min_npix, min_delta)
        if nok:
            # band width is +-1 sigma, so a tight selection draws a band only a
            # couple of pixels across; the solid edge lines are always visible so
            # "too small to see" cannot be mistaken for "not plotted"
            for m, sd in ((lo.mean(), lo.std(ddof=1)), (hi.mean(), hi.std(ddof=1))):
                row1[1].axvspan(m - sd, m + sd, color='tab:orange', alpha=0.3, lw=0)
                row1[1].axvline(m, color='tab:orange', lw=1.0, alpha=0.9)
            span_txt = (f'{lo.mean():.0f}+-{lo.std(ddof=1):.0f} to '
                        f'{hi.mean():.0f}+-{hi.std(ddof=1):.0f} pc,  '
                        f'{100*fr.mean():.1f}+-{100*fr.std(ddof=1):.1f}% of mass '
                        f'[{nok}/{len(sprof)} samples]')
            row1[1].set_title(span_txt, fontsize=9)
            print(f'  {table1["Name"].iloc[i]:18s} {span_txt}')
    for ax in (row1[0], row1[1]):
        ax.set_xlabel('Heliocentric distance (pc)')
        ax.set_ylabel('Mass per slice (Msun/pc)')

    im1 = row1[2].contourf(np.log10(NHs_ednhfr[i]), levels=NH_vals, extent=coords, cmap='cubehelix')
    row1[2].contour(np.log10(baseline[i]), levels=NH_vals[::10], extent=coords,
                    colors='black', linewidths=0.75, linestyles=['solid', '--'])
    row1[2].set_xlim(coords[:2])
    row1[2].set_xlabel('Galactic Longitude')
    row1[2].set_ylabel('Galactic Latitude')

    caxs = row1[3]; caxs.axis('off')
    cax1 = caxs.inset_axes([0.10, 0.06, 0.10, 0.88])
    cb1 = subfigs[0].colorbar(im1, cax=cax1, label=r'$N_H\ [\mathrm{cm^{-2}}]$')
    cb1.ax.tick_params(labelsize=10)
    cb1.locator = matplotlib.ticker.MaxNLocator(4); cb1.update_ticks()

    for ax in row2.flat:
        ax.set_xlim(coords[:2]); ax.set_aspect(1)
        ax.set_yticks([]); ax.set_xticks([])
    subfigs[1].subplots_adjust(wspace=0)

    if save:
        fig.savefig(FIG_LOC + table1['Name'].iloc[i] + suffix, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return [l.get_mask() for l in dend.leaves][j]


def compare_min_value(table1, cubes, min_npix=None, min_delta=20):
    """Leaf choice with no floor vs. the per-cloud robust floor, side by side."""
    hdr = (f'{"cloud":18s} {"thr":>7s} {"thr/pk":>6s} '
           f'{"span (none)":>16s} {"w":>5s} {"frac":>6s}  '
           f'{"span (auto)":>16s} {"w":>5s} {"frac":>6s}  {"rule":>7s}')
    print(hdr); print('-' * len(hdr))
    for i in range(len(table1)):
        prof = mass_profile(cubes[i])
        d_mean = float(table1.iloc[i]['d_mean(pc)'])
        d0, j0, f0, s0, _, _ = select_leaf(prof, d_mean, -np.inf, min_npix, min_delta)
        d1, j1, f1, s1, h1, mv = select_leaf(prof, d_mean, None, min_npix, min_delta)
        name = table1['Name'].iloc[i]
        if j1 is None:
            print(f'{name:18s} {mv:7.1f} {mv / prof.max():6.2f}   (no leaves survive)')
            continue
        rule = 'd_mean' if h1.any() else 'mass'
        # same leaf or a different one? compare where the peak sits, not the
        # span -- trimming the skirt moves the edges without changing the peak
        pk0 = ds[d0.leaves[j0].get_peak(subtree=True)[0][0]]
        pk1 = ds[d1.leaves[j1].get_peak(subtree=True)[0][0]]
        flip = '  <- different peak' if abs(pk0 - pk1) > 1 else ''
        print(f'{name:18s} {mv:7.1f} {mv / prof.max():6.2f} '
              f'{s0[j0][0]:7.0f}-{s0[j0][1]:<8.0f} {s0[j0][1] - s0[j0][0]:5.0f} {100 * f0[j0]:5.1f}%  '
              f'{s1[j1][0]:7.0f}-{s1[j1][1]:<8.0f} {s1[j1][1] - s1[j1][0]:5.0f} {100 * f1[j1]:5.1f}%  '
              f'{rule:>7s}{flip}')


def slice_density(cube, stat='p99'):
    """Per-distance-slice density statistic of a cloud cube, cm^-3.

    'max' is the single densest voxel in the slice and is what the note in the
    notebook literally asks for; 'p99' is the 99th percentile, which tracks the
    same structure but does not hang on one noisy voxel. Default is p99.
    """
    if stat == 'max':
        return np.nanmax(cube, axis=(1, 2))
    return np.nanpercentile(cube, 99, axis=(1, 2))


def widen_window(mask, ndens, n_thresh, stop_at_rise=True):
    """Grow a leaf mask outward until the slice density drops below n_thresh.

    Walks out from each edge of `mask` one slice at a time while
    ndens >= n_thresh. `stop_at_rise` also halts as soon as the density starts
    climbing again -- without it a sightline whose diffuse floor sits above the
    threshold (California and Lupus both do) never terminates, and the window
    walks straight through the valley into the next cloud.

    Returns a new boolean mask; the original is not modified.
    """
    m = np.asarray(mask).copy()
    idx = np.flatnonzero(m)
    lo, hi = idx[0], idx[-1]
    n = len(m)

    k = lo
    while k - 1 >= 0 and ndens[k - 1] >= n_thresh:
        if stop_at_rise and ndens[k - 1] > ndens[k]:
            break
        k -= 1
    new_lo = k

    k = hi
    while k + 1 < n and ndens[k + 1] >= n_thresh:
        if stop_at_rise and ndens[k + 1] > ndens[k]:
            break
        k += 1
    new_hi = k

    m[new_lo:new_hi + 1] = True
    return m


def widened_mask(cube, mask, n_thresh=None, frac_of_peak=None, stat='p99',
                 stop_at_rise=True):
    """widen_window with the threshold given absolutely or as a fraction of peak.

    frac_of_peak is measured against the densest slice INSIDE the leaf, so it
    adapts per cloud; n_thresh is an absolute cm^-3 cut. Give exactly one.
    """
    ndens = slice_density(cube, stat)
    if (n_thresh is None) == (frac_of_peak is None):
        raise ValueError('give exactly one of n_thresh, frac_of_peak')
    if frac_of_peak is not None:
        n_thresh = frac_of_peak * np.nanmax(ndens[mask])
    return widen_window(mask, ndens, n_thresh, stop_at_rise), n_thresh


def spine_angles(table1, cloud, cubes, n_seeds=40, tol=1.01, min_value=None,
                 n_jobs=-1):
    """Refit the spine on every posterior sample; report point + angles.

    The quoted spread pools, across all 6 samples, every candidate line whose
    residual is within `tol` of that sample's best. So it carries both sources
    of uncertainty: how much the density field itself moves between draws, and
    how poorly the L1 fit pins an orientation down on any one of them.
    """
    i = list(table1['Name']).index(cloud)
    prof = mass_profile(cubes[i])
    dend, j, fr, spn, holds, mv = select_leaf(
        prof, float(table1.iloc[i]['d_mean(pc)']), min_value)
    dmask = dend.leaves[j].get_mask()

    mbox, (xo, yo) = mean_padded_box(table1, cloud)
    scubes, indep, (sx, sy) = build_sample_cubes(table1, cloud)

    boxes = [('mean', mbox, xo, yo)] + [(f's{indep[k]}', scubes[k], sx, sy)
                                        for k in range(len(indep))]
    best, pooled = {}, []
    for lbl, box, ox, oy in boxes:
        rows, dats_arr = spine_candidates(i, CloudBoxProxy(box, ox, oy),
                                          table1, dmask, n_seeds, n_jobs)
        L = rows[:, 0]
        keep = rows[L <= L.min() * tol]
        descs = []
        for r in keep:
            p1, p2 = r[1:4], r[4:7]
            a = anchor_point(p1, p2, dats_arr)
            descs.append(spine_descriptors(np.array(xyz_2_sph(a)),
                                           lbd_tangent(p1, p2, a)))
        r0 = rows[L.argmin()]
        p1, p2 = r0[1:4], r0[4:7]
        a0 = anchor_point(p1, p2, dats_arr)
        best[lbl] = spine_descriptors(np.array(xyz_2_sph(a0)),
                                      lbd_tangent(p1, p2, a0))
        best[lbl]['nkeep'] = len(keep)
        if lbl != 'mean':
            pooled.extend(descs)
    return best, pooled


def print_spine_angles(best, pooled, cloud):
    K = ('l', 'b', 'd', 'pa_lb', 'inc_ld', 'inc_bd')
    print(f'\n=== {cloud}: spine refit on each posterior sample')
    print(f'{"":6s} {"n":>3s} {"l":>8s} {"b":>8s} {"d":>7s} '
          f'{"PA_lb":>8s} {"inc_ld":>8s} {"inc_bd":>8s}')
    for lbl, r in best.items():
        print(f'{lbl:6s} {r["nkeep"]:3d} {r["l"]:8.2f} {r["b"]:8.2f} {r["d"]:7.1f} '
              f'{r["pa_lb"]:8.2f} {r["inc_ld"]:8.2f} {r["inc_bd"]:8.2f}')
    P = {k: np.array([d[k] for d in pooled]) for k in K}
    print(f'  pooled over {len(pooled)} candidate lines from 6 samples:')
    unit = dict(l='deg', b='deg', d='pc', pa_lb='deg', inc_ld='deg', inc_bd='deg')
    for k in K:
        print(f'    {k:7s} {P[k].mean():9.2f} +- {P[k].std(ddof=1):6.2f} {unit[k]}'
              f'   [{P[k].min():.2f}, {P[k].max():.2f}]')
    return P


def cloud_masses(table1, cubes, min_value=None, min_npix=None, min_delta=20,
                 sprofs=None):
    """Mass inside each cloud's selected distance window.

    M = sum over the leaf of (mass-per-slice [Msun/pc] * 1 pc). When sample
    profiles are supplied the whole selection is rerun on each draw, so the
    quoted spread carries the leaf choice too, not just the mass at fixed leaf.
    """
    rows = []
    for i in range(len(table1)):
        prof = mass_profile(cubes[i])
        d_mean = float(table1.iloc[i]['d_mean(pc)'])
        dend, j, fr, sp, holds, mv = select_leaf(prof, d_mean, min_value, min_npix, min_delta)
        mask = dend.leaves[j].get_mask()
        M = np.nansum(prof * mask)                      # 1 pc slices
        Ms = None
        if sprofs is not None:
            vals = []
            for pr in sprofs[i]:
                d2, j2, _, _, _, _ = select_leaf(pr, d_mean, min_value, min_npix, min_delta)
                if j2 is not None:
                    vals.append(np.nansum(pr * d2.leaves[j2].get_mask()))
            Ms = np.array(vals)
        rows.append(dict(name=table1['Name'].iloc[i], d_mean=d_mean,
                         lo=sp[j][0], hi=sp[j][1], frac=fr[j], M=M, Ms=Ms,
                         rule='d_mean' if holds.any() else 'mass'))
    return rows


def print_masses(rows):
    hdr = (f'{"cloud":18s} {"d_mean":>7s} {"window (pc)":>14s} {"depth":>6s} '
           f'{"M (Msun)":>12s} {"+-":>10s} {"scatter":>8s} {"frac":>6s} {"rule":>7s}')
    print(hdr); print('-' * len(hdr))
    tot = 0.0
    for r in rows:
        tot += r['M']
        if r['Ms'] is not None and len(r['Ms']) > 1:
            sd = r['Ms'].std(ddof=1)
            err, pct = f'{sd:10.0f}', f'{100 * sd / r["Ms"].mean():7.1f}%'
        else:
            err, pct = ' ' * 10, ' ' * 8
        print(f'{r["name"]:18s} {r["d_mean"]:7.0f} {r["lo"]:6.0f}-{r["hi"]:<7.0f} '
              f'{r["hi"] - r["lo"]:6.0f} {r["M"]:12.0f} {err} {pct} '
              f'{100 * r["frac"]:5.1f}% {r["rule"]:>7s}')
    print('-' * len(hdr))
    print(f'{"total":18s} {"":7s} {"":14s} {"":6s} {tot:12.0f}')


def choosing_distances_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--clouds', nargs='*', default=None)
    ap.add_argument('--baseline', choices=['spectral', 'nicest'], default='spectral')
    ap.add_argument('--min-npix', type=int, default=None,
                    help='fixed slice-width cut; omit to scale with distance '
                         'via npix_for_distance()')
    ap.add_argument('--min-delta', type=float, default=20)
    ap.add_argument('--min-value', default='auto',
                    help="'auto' (per-cloud robust floor), 'none' (-inf), or a number")
    ap.add_argument('--samples', action='store_true',
                    help='overplot the independent posterior samples and quote spreads')
    ap.add_argument('--masses', action='store_true',
                    help='print the cloud mass table (uses --samples for errors)')
    ap.add_argument('--sample-summary', nargs='+', metavar='CLOUD',
                    help='projmaps_profiles with posterior samples, for these clouds')
    ap.add_argument('--n-iso', type=float, default=N_ISO,
                    help=f'volume-density threshold for --sample-summary (default {N_ISO:g})')
    ap.add_argument('--no-spine-slab', action='store_true',
                    help='--sample-summary: fit d_perp on the whole box, not just '
                         'between the walked ends of the spine')
    ap.add_argument('--no-figures', action='store_true',
                    help='--sample-summary: write the CSV only, no projmaps figures')
    ap.add_argument('--outdir', default=None,
                    help='where --sample-summary writes its figures and CSV '
                         '(default FIG_LOC, i.e. ../figures/ itself)')
    ap.add_argument('--spine-angles', nargs='+', metavar='CLOUD',
                    help='refit the spine per sample; report point + projection angles')
    ap.add_argument('--compare', action='store_true',
                    help='print the before/after leaf choice for every cloud, no figures')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--d-max', type=float, default=D_MAX,
                    help=f'catalog cut on d_mean(pc), default {D_MAX:.0f}')
    args = ap.parse_args()

    if args.min_value == 'auto':
        min_value = None
    elif args.min_value == 'none':
        min_value = -np.inf
    else:
        min_value = float(args.min_value)

    table1 = load_catalog(args.d_max)
    print(f'[catalog] {len(table1)} clouds with d_mean < {args.d_max:.0f} pc')
    cubes = build_cubes(table1, cache=None if args.no_cache else CACHE)

    if args.validate:
        validate(table1, cubes)
        return

    if args.compare:
        compare_min_value(table1, cubes, args.min_npix, args.min_delta)
        return

    if args.sample_summary:
        global FIG_LOC
        if args.outdir:
            FIG_LOC = args.outdir.rstrip('/') + '/'
            os.makedirs(FIG_LOC, exist_ok=True)
        allp, indep = build_sample_profiles(table1)
        sprofs = [pr[indep] for pr in allp]
        want = (list(table1['Name']) if args.sample_summary == ['all']
                else args.sample_summary)
        rows = []
        for cl in want:
            print(f'== {cl}')
            _, _, st = sample_summary(table1, cl, cubes, sprofs, min_value=min_value,
                                      n_iso=args.n_iso, save=not args.no_figures,
                                      spine_slab_cut=not args.no_spine_slab,
                                      suffix=f'_projmaps_profiles_samples_n{args.n_iso:g}')
            rows.append(st)
        cols = ['cloud', 'win_lo', 'win_hi', 'M', 'l', 'l_sd', 'b', 'b_sd',
                'd', 'd_sd', 'pa_lb', 'pa_lb_sd', 'inc_ld', 'inc_ld_sd',
                'inc_bd', 'inc_bd_sd', 'rho0', 'rho0_sd', 'r0', 'r0_sd',
                'alpha', 'alpha_sd', 'n_fit']
        df = pd.DataFrame(rows)[cols]
        out = f'{FIG_LOC}spine_angles_and_fits_samples_n{args.n_iso:g}.csv'
        df.to_csv(out, index=False, float_format='%.4f')
        print(f'\n{df.to_string(index=False, float_format=lambda v: f"{v:.2f}")}')
        print(f'\nwrote {out}')
        return

    if args.spine_angles:
        for cl in args.spine_angles:
            b, pooled = spine_angles(table1, cl, cubes, min_value=min_value)
            print_spine_angles(b, pooled, cl)
        return

    if args.masses:
        sp = None
        if args.samples:
            allp, indep = build_sample_profiles(table1)
            sp = [pr[indep] for pr in allp]
        print_masses(cloud_masses(table1, cubes, min_value,
                                  args.min_npix, args.min_delta, sp))
        return

    masses_per_slice = [(np.nansum(cb, axis=(1, 2)) * 1.36 * c.m_p * u.cm**-3
                         * (CD * u.deg.to(u.rad) * ds * u.pc)**2).to(u.Msun / u.pc)
                        for cb in cubes]
    NHs_ednhfr = [zeroth_moment(cb, 1, PC_CM) for cb in cubes]
    sprofs = None
    if args.samples:
        allp, indep = build_sample_profiles(table1)
        sprofs = [pr[indep] for pr in allp]      # one draw per antithetic pair

    spec_e, nic_e = load_spectral_and_nicest(table1, cubes)
    baseline = spec_e if args.baseline == 'spectral' else nic_e
    suffix = ('_choosing_distances_healpix' if args.baseline == 'spectral'
              else '_choosing_distances_healpix_nicest')

    want = range(len(table1))
    if args.clouds:
        names = list(table1['Name'])
        want = [names.index(cl) for cl in args.clouds]

    d_masks = {}
    for i in want:
        d_masks[table1['Name'].iloc[i]] = get_best_leaf(
            i, table1, cubes, masses_per_slice, NHs_ednhfr, baseline, suffix=suffix,
            min_npix=args.min_npix, min_delta=args.min_delta, min_value=min_value,
            sprof=None if sprofs is None else sprofs[i])
        print(f'  saved {FIG_LOC}{table1["Name"].iloc[i]}{suffix}.png')
    return d_masks


# ----------------------------------------------------------------------------
# from make_nicest_contour_figs.py: NICEST and DHT CO contour figures (Aug 2026).
# ----------------------------------------------------------------------------
NICEST_FIG_OUTDIR = RUNS + 'nicest_contours'

LEVEL_PCTS = [70, 80, 90, 95, 98]     # NICEST contour levels, per-cloud percentiles

CO_LEVEL_PCTS = [85, 92, 96, 99]      # fewer CO levels -- two contour sets get busy

CMAP = 'cubehelix_r'                  # near-white at low column, near-black at high
NICEST_COLOR, CO_COLOR = 'green', 'red'
MIN_NPIX, MIN_DELTA = 30, 20

# The DHT cube carries no WCS, so these constants *are* its grid definition.
# b0 is -50: the latitude axis runs -50 -> +49.875 over 800 rows at 0.125 deg.
# (Written +50 in an earlier copy of this file. That still gave correct rows,
#  because coord_idx then returns a negative index that numpy wraps by exactly
#  ny = 800 = 100/0.125 -- an accident that breaks if the array ever changes.)
CO_XYZ0S = [360, -50, -100]

CO_DXYZS = [-0.125, 0.125, 1.3]       # dz = 1.3 km/s, the survey's true resolution


def load_boxes():
    """cloud -> (l_min, l_max, b_min, b_max, source)."""
    out = {}
    for fname, tag in (('zucker2020_clouds_within_1200pc.csv', 'zucker2020'),
                       ('missing_clouds_for_inicest.csv', 'missing_clouds')):
        for _, r in pd.read_csv(f'{INPUTS}{fname}').iterrows():
            out[r['cloud']] = (float(r['l_min(deg)']), float(r['l_max(deg)']),
                               float(r['b_min(deg)']), float(r['b_max(deg)']), tag)
    # Cam was re-fetched on mi_tab1's Camelopardalis box.
    mi = pd.read_csv(f'{RESEARCH}/data/catalogs/mi_tab1.csv')
    r = mi[mi['Name'] == 'Camelopardalis'].iloc[0]
    out['Cam'] = (float(r['l_min(deg)']), float(r['l_max(deg)']),
                  float(r['b_min(deg)']), float(r['b_max(deg)']), 'mi_tab1')
    return out


def nicest_axes(path, l_center):
    """(l, b, data) for a CAR NICEST map; l unwrapped to sit near l_center.

    CRVAL2 == 0 makes the grid plain plate carree, so each axis is just
    CRVAL + CDELT * (pix - CRPIX).  Returned with l ascending so contour()
    gets monotonic coordinates.
    """
    hdu = fits.open(path)[0]
    h, D = hdu.header, hdu.data.astype(float)
    assert h['CTYPE1'] == 'GLON-CAR' and h['CRVAL2'] == 0.0, f"not CAR: {path}"
    ny, nx = shape
    l = h['CRVAL1'] + h['CDELT1'] * (np.arange(nx) + 1 - h['CRPIX1'])
    b = h['CRVAL2'] + h['CDELT2'] * (np.arange(ny) + 1 - h['CRPIX2'])
    # unwrap around the cloud's own longitude so there is no 0/360 or +-180 jump
    l = ((l - l_center + 180.0) % 360.0) - 180.0 + l_center
    if l[0] > l[-1]:
        l, D = l[::-1], D[:, ::-1]
    return l, b, D


def contour_nicest(ax, l, b, D):
    fin = D[np.isfinite(D)]
    if fin.size < 10:
        return
    levels = np.unique(np.nanpercentile(fin, LEVEL_PCTS))
    if levels.size >= 2:
        ax.contour(l, b, D, levels=levels, colors=NICEST_COLOR, linewidths=0.75)


def contour_co(ax, co2d, extent):
    """Red CO contours.  No-op when the cloud falls outside the survey."""
    if co2d is None or co2d.size == 0:
        return
    fin = co2d[np.isfinite(co2d)]
    if fin.size < 10 or np.nanstd(fin) == 0:
        return
    levels = np.unique(np.nanpercentile(fin, CO_LEVEL_PCTS))
    if levels.size >= 2:
        ax.contour(co2d, levels=levels, extent=extent, colors=CO_COLOR,
                   linewidths=0.6, alpha=0.9)


def make_nicest_contour_figs_main():
    ednhfr = fits.open(f'{DATA}edenhofer_lbd_wrap.fits')[0].data
    dht    = fits.open(f'{DATA}hi4piDHT_720deg.fits')[1].data
    boxes = load_boxes()
    maps = {os.path.basename(p).replace('_nicest.fits', ''): p
            for p in glob.glob(f'{NICEST}/*_nicest.fits')}
    todo = sorted(k for k in maps if k in boxes)
    print(f"{len(todo)} clouds; no box for {sorted(set(maps) - set(boxes))}", flush=True)

    summary = []
    for name in todo:
        lmin, lmax, bmin, bmax, src = boxes[name]
        cat = pd.DataFrame([{'l_max(deg)': lmax, 'l_min(deg)': lmin,
                             'b_min(deg)': bmin, 'b_max(deg)': bmax,
                             'd_mean(pc)': 500.0, 'Name': name}])
        try:
            cube, coords = get_cloud_info(0, ednhfr, cat, 'full')
            if min(cube.shape[1:]) < 2:
                print(f"  {name:<18} SKIP degenerate box {cube.shape}", flush=True)
                continue

            mps = (np.nansum(cube, axis=(1, 2)) * 1.36 * c.m_p * u.cm**-3 *
                   (0.229 * u.deg.to(u.rad) * np.arange(68, 1250) * u.pc)**2
                   ).to(u.Msun / u.pc)
            dend = Dendrogram.compute(mps.value, min_npix=MIN_NPIX,
                                      min_delta=MIN_DELTA)
            leaves = dend.leaves
            if not leaves:
                print(f"  {name:<18} SKIP no leaves", flush=True)
                continue

            extent = list(coords[:4])                     # l_max, l_min, b_min, b_max
            nl, nb, nD = nicest_axes(maps[name], 0.5 * (coords[0] + coords[1]))

            # CO on the same catalog box; None when the cloud is off the survey
            try:
                cocube = get_cloud_info(0, dht, cat, 'full',
                                        xyz0s=CO_XYZ0S, dxyzs=CO_DXYZS)[0]
                co2d = zeroth_moment(cocube) if min(cocube.shape[1:]) >= 2 else None
            except Exception:
                co2d = None
            p = dend.plotter()
            p.set_custom_positions(peak_x)

            # ---------------- row 1 ----------------
            fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
            axs[0].plot(ds, mps.value, color='black', lw=1)
            p.plot_tree(axs[0], color='red')
            axs[0].set_xlabel('d (pc)')
            axs[0].set_ylabel(r'dM/dd  [$M_\odot$ pc$^{-1}$]')
            axs[0].set_title('mass per slice + dendrogram', fontsize=10)

            for leaf in leaves:
                axs[1].plot(ds, leaf.get_mask() * mps.value, lw=1)
            axs[1].set_xlabel('d (pc)')
            axs[1].set_title(f'{len(leaves)} leaves', fontsize=10)

            cf = axs[2].contourf(zeroth_moment(cube), levels=40, extent=extent, cmap=CMAP)
            contour_nicest(axs[2], nl, nb, nD)
            contour_co(axs[2], co2d, extent)
            axs[2].set_xlim(extent[0], extent[1]); axs[2].set_ylim(extent[2], extent[3])
            axs[2].set_xlabel('l (deg)'); axs[2].set_ylabel('b (deg)')
            axs[2].set_title('dust zeroth moment + NICEST (green) + CO (red)'
                             if co2d is not None else
                             'dust zeroth moment + NICEST (green); no CO',
                             fontsize=9)

            # inset colourbar, styled light so it reads over the dark end of viridis
            cax = axs[2].inset_axes([0.04, 0.09, 0.36, 0.04])
            cb = fig.colorbar(cf, cax=cax, orientation='horizontal')
            # ticks and label above the bar, or they fall outside the panel and clip
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.xaxis.set_label_position('top')
            cb.set_label(r'$\int n\,\mathrm{d}d$  [cm$^{-3}$ pc]', color='black',
                         fontsize=7, labelpad=1)
            cb.ax.tick_params(colors='black', labelsize=6, length=2, pad=1)
            cb.outline.set_edgecolor('black'); cb.outline.set_linewidth(0.5)
            # the map under the inset can be white or black, so back the text
            for t in [cb.ax.xaxis.label] + cb.ax.get_xticklabels():
                t.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.5))
            cb.locator = matplotlib.ticker.MaxNLocator(3); cb.update_ticks()

            fig.suptitle(f'{name}   ({src})')
            fig.tight_layout()
            fig.savefig(f'{NICEST_FIG_OUTDIR}/{name}_row1.png', dpi=120)
            plt.close(fig)

            # ---------------- row 2 ----------------
            N = len(leaves)
            aspect = abs(bmax - bmin) / max(abs(lmax - lmin), 1e-6)
            h = float(np.clip(3.0 * aspect, 2.2, 6.0))
            # Shared colour scale across the row so panel brightness is comparable
            # between leaves; each panel would otherwise autoscale to its own max.
            leaf_maps = [zeroth_moment(cube * leaf.get_mask()[:, np.newaxis, np.newaxis])
                         for leaf in leaves]
            vmax = float(np.nanmax([np.nanmax(z) for z in leaf_maps]))
            vmin = min(0.0, float(np.nanmin([np.nanmin(z) for z in leaf_maps])))
            shared_levels = np.linspace(vmin, vmax, 41)

            # Three shared-scale variants.  All are comparable across the row; they
            # differ only in how the common range is mapped to colour, which
            # matters when one leaf dominates and the rest would render flat.
            p98 = float(np.nanpercentile(np.concatenate([z.ravel() for z in leaf_maps]), 98))
            variants = [
                ('p98', np.linspace(vmin, p98, 41), None,
                 f'98th pct cap, {p98:.0f} (max {vmax:.0f})'),
            ]

            for tag, lv, norm, blurb in variants:
                fig, axs = plt.subplots(1, N, figsize=(N * 3.0, h + 0.8), squeeze=False)
                axs = axs[0]
                for n, (leaf, zmap) in enumerate(zip(leaves, leaf_maps)):
                    m = leaf.get_mask()
                    axs[n].contourf(zmap, levels=lv, extent=extent, extend='max',
                                    norm=norm, cmap=CMAP)
                    contour_nicest(axs[n], nl, nb, nD)
                    contour_co(axs[n], co2d, extent)
                    axs[n].set_xlim(extent[0], extent[1])
                    axs[n].set_ylim(extent[2], extent[3])
                    axs[n].set_xticks([]); axs[n].set_yticks([])
                    d0, d1 = ds[m][0], ds[m][-1]
                    axs[n].set_title(f'{d0:.0f}-{d1:.0f} pc\npeak {peak_x(leaf):.0f}',
                                     fontsize=9)
                fig.suptitle(f'{name} -- dust zeroth moment per dendrogram leaf'
                             f'  ({blurb})', y=0.99)
                fig.subplots_adjust(wspace=0.08, top=0.74)
                suffix = '' if tag == 'linear' else f'_{tag}'
                fig.savefig(f'{NICEST_FIG_OUTDIR}/{name}_row2{suffix}.png', dpi=120,
                            bbox_inches='tight')
                plt.close(fig)

            summary.append(dict(cloud=name, source=src, n_leaves=N,
                                has_co=co2d is not None,
                                cube_shape=list(cube.shape),
                                leaves=[dict(d_min=float(ds[leaf.get_mask()][0]),
                                             d_max=float(ds[leaf.get_mask()][-1]),
                                             d_peak=float(peak_x(leaf)))
                                        for leaf in leaves]))
            print(f"  {name:<18} {src:<15} N={N:<3} cube={cube.shape}", flush=True)
        except Exception as e:
            print(f"  {name:<18} FAILED: {type(e).__name__}: {e}", flush=True)

    json.dump(summary, open(f'{NICEST_FIG_OUTDIR}/summary.json', 'w'), indent=1)
    print(f"\n{len(summary)}/{len(todo)} clouds plotted -> {NICEST_FIG_OUTDIR}", flush=True)


# ----------------------------------------------------------------------------
# from fetch_inicest_maps.py: downloads NICER/NICEST maps from the iNICEST web service.
# ----------------------------------------------------------------------------
INICEST_BASE = "http://interstellarclouds.fisica.unimi.it"

INICEST_CGI = f"{INICEST_BASE}/cgi-bin"

INICEST_CATALOG = __import__('pathlib').Path(INPUTS) / 'zucker2020_clouds_within_1200pc.csv'

INICEST_OUTDIR = __import__('pathlib').Path(LOCAL) / 'nicest_maps' / 'zucker2020_fwhm0.229deg'

# Target *effective* resolution, set by the smoothing kernel rather than the
# pixel grid: INICEST_SMOOTH_PIX pixels per FWHM.  Sampling the beam with 2 pixels
# keeps the maps Nyquist-sampled at 0.229 deg (HEALPix nside=256) instead of
# letting the kernel wash out a grid of that same size.
INICEST_RESOLUTION_DEG = 0.229

INICEST_SMOOTH_PIX = 2.0

INICEST_PIXEL_DEG = INICEST_RESOLUTION_DEG / INICEST_SMOOTH_PIX      # 0.1145 deg/px

INICEST_SCALE_ARCMIN = INICEST_PIXEL_DEG * 60.0              # 6.87 '/px -- the form wants arcmin

INICEST_PAD_DEG = 5.0
INICEST_EMAIL = "tmekuria@ucsc.edu"

# FITS projection code for the requested maps.  TAN is what the web client
# sends; CAR gives a plain l/b grid (see inicest_build_wcs).
INICEST_PROJECTION = "TAN"

# Products to request.  The full set the form offers is:
#   nicer nicest nicer_ivar nicest_ivar density model_densities model_errors
#   control mask plots catalog
INICEST_PRODUCTS = "nicer,nicest,nicer_ivar,nicest_ivar,density"

# Files downloaded on completion (<name>.fits under /jobs/job<id>/).
INICEST_FITS_PRODUCTS = ["nicer", "nicest", "nicer_ivar", "nicest_ivar", "density"]


# --------------------------------------------------------------------------
# catalog
# --------------------------------------------------------------------------
def inicest_pad_box(lmin, lmax, bmin, bmax, pad=INICEST_PAD_DEG):
    """Grow a galactic box by `pad` degrees on all four sides.

    Latitude is clipped at the poles.  If the padded longitude range would
    wrap all the way around, it is opened up to the full 0-360 circle.
    """
    bmin = max(-90.0, bmin - pad)
    bmax = min(90.0, bmax + pad)

    width = (lmax - lmin) % 360.0  # 0 if the box already spans the circle
    if width + 2 * pad >= 360.0:
        lmin, lmax = 0.0, 360.0
    else:
        lmin = (lmin - pad) % 360.0
        lmax = (lmax + pad) % 360.0
    return lmin, lmax, bmin, bmax


def inicest_read_catalog(path=INICEST_CATALOG, pad=INICEST_PAD_DEG):
    """Zucker+2020 boxes, padded, with longitudes wrapped into [0, 360)."""
    clouds = []
    with open(path) as f:
        for rec in csv.DictReader(f):
            rec = {k.strip(): v.strip() for k, v in rec.items()}
            raw = (float(rec["l_min(deg)"]) % 360.0,
                   float(rec["l_max(deg)"]) % 360.0,
                   float(rec["b_min(deg)"]),
                   float(rec["b_max(deg)"]))
            lmin, lmax, bmin, bmax = inicest_pad_box(*raw, pad=pad)
            clouds.append(dict(
                name=rec["cloud"],
                glon_min=lmin, glon_max=lmax,
                glat_min=bmin, glat_max=bmax,
                raw_glon_min=raw[0], raw_glon_max=raw[1],
                raw_glat_min=raw[2], raw_glat_max=raw[3],
                pad=pad,
                distance=float(rec["d_mean(pc)"]),
            ))
    return clouds


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def inicest_get(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def inicest_count_stars(c):
    q = urllib.parse.urlencode(dict(
        glon_min=c["glon_min"], glon_max=c["glon_max"],
        glat_min=c["glat_min"], glat_max=c["glat_max"]))
    return json.loads(inicest_get(f"{INICEST_CGI}/inicest_count_stars.py?{q}"))["nstars"]


# --------------------------------------------------------------------------
# WCS
# --------------------------------------------------------------------------
def inicest_build_wcs(c, scale=INICEST_SCALE_ARCMIN, projection=INICEST_PROJECTION):
    """Replicate the client's guessWCS/improveWCS, but with the scale pinned.

    Centred on the box and sized so all four corners fit with a 10-pixel
    margin (what improveWCS does with its +20 on each axis).

    `projection` is the three-letter FITS code.  For CAR the reference
    latitude is forced to 0 so the result is a plain plate carree grid --
    linear in both l and b, matching the Edenhofer cube's convention.  A CAR
    map with CRVAL2 != 0 is an oblique variant with curved parallels, which
    is not what we want here.
    """
    from astropy.wcs import WCS

    lmin, lmax = c["glon_min"], c["glon_max"]
    if lmin > lmax:  # box crosses the meridian
        lmin -= 360.0
    crval1 = (0.5 * (lmin + lmax)) % 360.0
    crval2 = 0.0 if projection == "CAR" else 0.5 * (c["glat_min"] + c["glat_max"])

    w = WCS(naxis=2)
    w.wcs.ctype = [f"GLON-{projection}", f"GLAT-{projection}"]
    w.wcs.crval = [crval1, crval2]
    w.wcs.cdelt = [-scale / 60.0, scale / 60.0]
    w.wcs.crpix = [1.0, 1.0]

    corners_l = [lmin, lmax, lmin, lmax]
    corners_b = [c["glat_min"], c["glat_min"], c["glat_max"], c["glat_max"]]
    x, y = w.wcs_world2pix(corners_l, corners_b, 1)

    xmin, xmax = float(min(x)), float(max(x))
    ymin, ymax = float(min(y)), float(max(y))

    naxis1 = int(math.ceil(xmax - xmin)) + 20
    naxis2 = int(math.ceil(ymax - ymin)) + 20
    crpix1 = round(1.0 - xmin) + 10
    crpix2 = round(1.0 - ymin) + 10

    return dict(naxis1=naxis1, naxis2=naxis2, crpix1=crpix1, crpix2=crpix2,
                crval1=round(crval1, 6), crval2=round(crval2, 6))


# --------------------------------------------------------------------------
# job payload
# --------------------------------------------------------------------------
def inicest_box_fields(c):
    """Both representations of the box: corners *and* centre/width.

    The browser always fills in both, and the server appears to parse the
    centre/width pair unconditionally -- leaving them blank makes the job
    abort immediately after starting.
    """
    lmin, lmax = c["glon_min"], c["glon_max"]
    width_l = (lmax - lmin) % 360.0
    if width_l == 0.0 and lmax != lmin:
        width_l = 360.0
    center_l = (lmin + 0.5 * width_l) % 360.0
    center_b = 0.5 * (c["glat_min"] + c["glat_max"])
    width_b = c["glat_max"] - c["glat_min"]
    return {
        "glon_min": f"{lmin:.4f}", "glon_max": f"{lmax:.4f}",
        "glat_min": f"{c['glat_min']:.4f}", "glat_max": f"{c['glat_max']:.4f}",
        "glon_center": f"{center_l:.4f}", "glat_center": f"{center_b:.4f}",
        "glon_width": f"{width_l:.4f}", "glat_width": f"{width_b:.4f}",
    }


def inicest_build_parameters(c, nstars, wcs, products=INICEST_PRODUCTS, scale=INICEST_SCALE_ARCMIN,
                     projection=INICEST_PROJECTION):
    """The nested object the browser keeps in localStorage.inicest."""
    box = inicest_box_fields(c)

    # Pan/zoom state of the all-sky viewer.  Pure UI state -- the widths are
    # identical across steps in real sessions regardless of the selected box --
    # but the key must be present or the server raises before doing any work.
    osd = {"x": 0.0, "y": 0.0, "width": 0.125, "height": 0.07725, "degrees": 0}

    step1 = {"object": "", **box, "OSD_bounds": osd, "nstars": nstars}

    # Control field: the site's default heuristic.  Take the same area as the
    # science field, then let the pipeline keep the 50% of the area with the
    # lowest column density and, within it, the 50% of stars with the lowest
    # individual extinction.  Intrinsic colours and number counts are left
    # blank so the pipeline measures them itself.
    step2 = {
        "useScienceField": True,
        # Mirrored from the science field.  With CF_EQ_SF true the server
        # should ignore these, but they are filled in so no numeric field is
        # ever blank.
        **box,
        "p_area": "50", "p_stars": "50",
        "jerr_max": "0.1", "herr_max": "0.1", "kerr_max": "0.1",
        "mean_j_h": "", "mean_h_k": "",
        "cov11": "", "cov22": "", "cov12": "",
        "max_h": "", "beta": "", "m_compl": "", "s_compl": "",
        "OSD_bounds": osd,
        # Star count of the control field.  Since the control field is the
        # science field here, it is the same number.
        "nstars": nstars,
    }

    step3 = {
        "email": INICEST_EMAIL,
        "products": products,
        "autoWCS": False,
        "naxis1": str(wcs["naxis1"]), "naxis2": str(wcs["naxis2"]),
        "crpix1": str(wcs["crpix1"]), "crpix2": str(wcs["crpix2"]),
        "crval1": str(wcs["crval1"]), "crval2": str(wcs["crval2"]),
        "type1": "galactic", "type2": projection,
        "scale": f"{scale:.4f}",
        "crot": "0", "lonpole": "", "latpole": "",
        "pv2_0": "", "pv2_1": "", "pv2_2": "", "pv2_3": "",
        # Smoothing FWHM is in *pixels*, not arcmin.
        "smooth": f"{INICEST_SMOOTH_PIX}", "gal_contam": False,
        "clip": "3.0", "niters": "3",
        "law_j": "2.50", "law_h": "1.55", "law_k": "1.00",
        "hlf_auto": False, "distance": "",
        "hlf_size": "1.65", "hlf_cntr": "12",
        "ignore_model2": False,
    }

    return {"step1": step1, "step2": step2, "step3": step3}


def inicest_submit(parameters):
    data = urllib.parse.urlencode(
        {"parameters": json.dumps(parameters)}).encode()
    req = urllib.request.Request(f"{INICEST_CGI}/make_maps.py", data=data)
    with urllib.request.urlopen(req, timeout=180) as r:
        body = r.read().decode("utf-8", "replace")
    try:
        return json.loads(body)["jobid"], body
    except Exception:
        return None, body  # server returned an HTML error page


# --------------------------------------------------------------------------
# polling / inicest_download
# --------------------------------------------------------------------------
def inicest_poll(jobid, interval=20, timeout=7200, verbose=True, grace=60):
    """Wait for a job to finish.

    A freshly submitted job can report "error" for a few seconds before it is
    registered, so terminal states are only believed after `grace` seconds.
    """
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        elapsed = time.time() - t0
        try:
            body = inicest_get(f"{INICEST_CGI}/check_job.py?jobid={jobid}").decode()
            status = json.loads(body)
        except Exception:
            status = {}
        state = status.get("status", status)
        if verbose and state != last:
            print(f"      [{int(time.time()-t0):5d}s] {state}")
            last = state
        if isinstance(state, str) and state.lower() == "completed":
            return True
        if (isinstance(state, str) and elapsed >= grace
                and state.lower() in ("error", "failed", "aborted")):
            return False
        # Fall back to the PROGRESS file the results page also watches.
        try:
            prog = inicest_get(f"{INICEST_BASE}/jobs/job{jobid}/PROGRESS", timeout=30).decode().strip()
            if prog and verbose and prog != last:
                print(f"      [{int(time.time()-t0):5d}s] progress {prog}")
                last = prog
            if prog.startswith("100"):
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def inicest_download(jobid, name, outdir=INICEST_OUTDIR, products=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    got = []
    for prod in list(products or INICEST_FITS_PRODUCTS) + ["inicest.log"]:
        fname = prod if prod.endswith(".log") else f"{prod}.fits"
        url = f"{INICEST_BASE}/jobs/job{jobid}/{fname}"
        dest = outdir / f"{name}_{fname}"
        try:
            blob = inicest_get(url, timeout=300)
        except Exception as e:
            print(f"      miss {fname}: {e}")
            continue
        dest.write_bytes(blob)
        got.append(dest.name)
        print(f"      saved {dest.name} ({len(blob):,} bytes)")
    return got


# --------------------------------------------------------------------------
def inicest_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append",
                    help="cloud name (repeatable); default is all")
    ap.add_argument("--list", action="store_true",
                    help="show planned geometry and exit, submitting nothing")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the payload without submitting")
    ap.add_argument("--pad", type=float, default=INICEST_PAD_DEG,
                    help=f"degrees to grow each box on all sides "
                         f"(default {INICEST_PAD_DEG})")
    ap.add_argument("--products", default=INICEST_PRODUCTS,
                    help=f"comma-separated product list (default {INICEST_PRODUCTS})")
    ap.add_argument("--scale", type=float, default=INICEST_SCALE_ARCMIN,
                    help=f"pixel scale in arcmin (default {INICEST_SCALE_ARCMIN})")
    ap.add_argument("--outdir", default=str(INICEST_OUTDIR))
    ap.add_argument("--projection", default=INICEST_PROJECTION,
                    help=f"FITS projection code, e.g. TAN or CAR "
                         f"(default {INICEST_PROJECTION})")
    ap.add_argument("--catalog", default=str(INICEST_CATALOG),
                    help=f"input CSV; needs a `cloud` column plus the four "
                         f"l/b box columns (default {INICEST_CATALOG.name})")
    args = ap.parse_args()

    clouds = inicest_read_catalog(path=args.catalog, pad=args.pad)
    if args.only:
        want = {n.lower() for n in args.only}
        clouds = [c for c in clouds if c["name"].lower() in want]
        if not clouds:
            sys.exit(f"no cloud matched {args.only}")

    manifest = []
    for c in clouds:
        wcs = inicest_build_wcs(c, args.scale, args.projection)
        print(f"{c['name']:18s} l=[{c['glon_min']:6.1f},{c['glon_max']:6.1f}] "
              f"b=[{c['glat_min']:6.1f},{c['glat_max']:6.1f}]  "
              f"{wcs['naxis1']:4d}x{wcs['naxis2']:<4d} px  "
              f"crval=({wcs['crval1']:.3f},{wcs['crval2']:.3f})")
        if args.list:
            continue

        nstars = inicest_count_stars(c)
        print(f"      {nstars:,.0f} stars")
        params = inicest_build_parameters(c, nstars, wcs, args.products, args.scale,
                                  args.projection)
        if args.dry_run:
            print(json.dumps(params, indent=2))
            continue

        jobid, body = inicest_submit(params)
        if jobid is None:
            print(f"      SUBMIT FAILED; server said:\n{body[:1500]}")
            continue
        print(f"      jobid {jobid}")
        ok = inicest_poll(jobid)
        if not ok:
            print("      job did not complete cleanly; trying inicest_download anyway")
        files = inicest_download(jobid, c["name"], args.outdir,
                         products=args.products.split(","))
        manifest.append(dict(cloud=c["name"], jobid=jobid, wcs=wcs,
                             nstars=nstars, pad_deg=args.pad,
                             pixel_deg=args.scale / 60.0,
                             smooth_pix=INICEST_SMOOTH_PIX,
                             resolution_deg=INICEST_SMOOTH_PIX * args.scale / 60.0,
                             box=dict(glon_min=c["glon_min"],
                                      glon_max=c["glon_max"],
                                      glat_min=c["glat_min"],
                                      glat_max=c["glat_max"]),
                             files=files))

    if manifest:
        mpath = Path(args.outdir) / "manifest.json"
        mpath.write_text(json.dumps(manifest, indent=2))
        print(f"\nwrote {mpath}")


# ----------------------------------------------------------------------------
# from interp2box.py: HEALPix -> regular-grid interpolation (its script body, which ran on import, is not copied).
# ----------------------------------------------------------------------------
DustSphere = namedtuple(
    "DustSphere", (
        "data", "nside", "nest", "radii", "coo_bounds", "radii0", "data0",
        "units", "data_uncertainty", "data0_uncertainty"
    )
)

PRESERVED_HEADER_ATTRS = ("AUTHOR", "DATE", "VERSION", "REF", "CONTACT")


def get_sphere(filepath):
    radii0 = None
    rec_dust0 = None
    rec_uncertainty = None
    rec0_uncertainty = None
    with fits.open(filepath, "readonly") as hdul:
        for hdu in hdul:
            nm = hdu.name.lower()
            if isinstance(hdu, fits.PrimaryHDU) and hdu.data is None:
                continue
            elif nm in ("primary", "mean", "samples", "healpix times distance"):
                dust_density = hdu.data
                nside = hdu.header["NSIDE"]
                nest = hdu.header["ORDERING"].lower().startswith("nest")
                units = hdu.header.get("CUNIT")
                if dust_density.shape[-1] != 12 * nside**2:
                    ve = "invalid shape of dust density {!r}"
                    raise ValueError(ve.format(dust_density.shape))
            elif isinstance(hdu, fits.BinTableHDU):
                if hdu.data.names == ['radial pixel centers']:
                    radii = hdu.data["radial pixel centers"]
                elif hdu.data.names == ['radial pixel boundaries']:
                    coo_bounds = hdu.data["radial pixel boundaries"]
                else:
                    ve = "unrecognized entry in BinTableHDU {!r}"
                    raise ValueError(ve.format(hdu.data.names))
            elif isinstance(hdu, fits.ImageHDU) and (
                nm.startswith("mean of integrated inner") or
                nm.startswith("integrated inner")
            ):
                prfx = "inner density integrated within"
                ctp = hdu.header.get("CTYPE")
                ctp = hdu.header.get("CTYPE1") if ctp is None else ctp
                if ctp.lower().startswith(prfx):
                    radii0 = ctp.lower().removeprefix(prfx)
                    if not radii0.endswith("pc"):
                        ve = "unrecognized units {!r}".format(radii0)
                        raise ValueError(ve)
                    radii0 = radii0.removesuffix("pc")
                    radii0 = float(radii0.strip())
                    rec_dust0 = hdu.data
                    if nest != hdu.header["ORDERING"].lower(
                    ).startswith("nest"):
                        raise ValueError("ordering mismatch")
                    if rec_dust0.shape[-1] != 12 * nside**2:
                        ve = "incompatible shape of dust density {!r}"
                        raise ValueError(ve.format(dust_density.shape))
            elif isinstance(hdu, fits.ImageHDU
                           ) and nm.startswith("std. of integrated inner"):
                prfx = "std. of inner density integrated within"
                ctp = hdu.header.get("CTYPE")
                ctp = hdu.header.get("CTYPE1") if ctp is None else ctp
                if hdu.header["CTYPE"].lower().startswith(prfx):
                    radii0_unc = ctp.lower().removeprefix(prfx)
                    if not radii0_unc.endswith("pc"):
                        ve = "unrecognized units {!r}".format(radii0)
                        raise ValueError(ve)
                    radii0_unc = radii0_unc.removesuffix("pc")
                    radii0_unc = float(radii0_unc.strip())
                    rec0_uncertainty = hdu.data
                    if radii0_unc != radii0:
                        raise ValueError("radii mismatch")
                    if nest != hdu.header["ORDERING"].lower(
                    ).startswith("nest"):
                        raise ValueError("ordering mismatch")
                    if rec0_uncertainty.shape[-1] != 12 * nside**2:
                        ve = "incompatible shape of dust density uncertainty {!r}"
                        raise ValueError(ve.format(rec0_uncertainty.shape))
            elif isinstance(hdu, fits.ImageHDU) and nm.lower() == "std.":
                rec_uncertainty = hdu.data
                if nest != hdu.header["ORDERING"].lower().startswith("nest"):
                    raise ValueError("ordering mismatch")
                if rec_uncertainty.shape[-1] != 12 * nside**2:
                    ve = "incompatible shape of dust density uncertainty {!r}"
                    raise ValueError(ve.format(rec_uncertainty.shape))
            else:
                raise ValueError("unrecognized HDU\n{!r}".format(hdu.header))

    return DustSphere(
        dust_density,
        nside=nside,
        nest=nest,
        radii=radii,
        coo_bounds=coo_bounds,
        radii0=radii0,
        data0=rec_dust0,
        units=units,
        data_uncertainty=rec_uncertainty,
        data0_uncertainty=rec0_uncertainty,
    )


def cart2sph(x, y, z):
    """Converts Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x, y, z : jnp.ndarray of float or float
        Cartesian x-, y- and z-position respectively.

    Returns
    -------
    output: tuple of jnp.ndarray of float or float
        Position in radius (r), azimuthal angle (l) and polar angle (b).
    """
    l = np.arctan2(y, x)
    r1 = x**2 + y**2
    b = np.arctan2(z, np.sqrt(r1))
    r = np.sqrt(r1 + z**2)
    return r, l, b


def get_interp_val(m, theta, phi, nest=False, lonlat=False):
    # Taken from the unreleased version of healpy.pixelfunc.
    # Thanks Jayson Vavrek (@jvavrek) for generlizing the function to multiple
    # healpix maps!
    from healpy.pixelfunc import lonlat2thetaphi, npix2nside
    from healpy import _healpy_pixel_lib as pixlib

    m = np.atleast_2d(m)
    _, npix = m.shape
    nside = npix2nside(npix)
    if lonlat:
        theta, phi = lonlat2thetaphi(theta, phi)
    if nest:
        r = pixlib._get_interpol_nest(nside, theta, phi)
    else:
        r = pixlib._get_interpol_ring(nside, theta, phi)
    p = np.array(r[0:4])
    w = np.array(r[4:8])
    del r
    return np.squeeze(np.sum(m[:, p] * w, axis=1))[()]


def interp_hp2rg(
    new_pos,
    r,
    hp_map,
    nest: bool = True,
    fill_value: float = np.nan,
    *,
    verbose=False
) -> NDArray:
    """Interpolates a HEALPIx times radius sphere to a regular grid.

    Parameters
    ----------
    new_pos : array-like
        Array-like of Cartesian (x, y, z) coordinates. The first axis should be
        the dimension axis, i.e. 3.
    r : array-like
        Radii of the HEALPix spheres. Alternatively, can also be a chart in
        which case, the radii are automatically determined from the chart
    hp_map : array-like
        HEALPix times radius sphere. Broadcasts along the first axes.
    fill_value : float
        Value to insert at `new_pos` postions not within the HEALPix times
        radius sphere.
    """
    r = np.asarray(r)
    if r.ndim != 1:
        raise ValueError("`r` must be one dimensional")
    new_pos = np.asarray(new_pos)
    if new_pos.shape[0] != 3:
        raise ValueError("first axis of `new_pos` must be coordinate axis")

    rlb = np.stack(cart2sph(*new_pos)).reshape(3, -1)
    rg_map = np.full(
        hp_map.shape[:-2] + new_pos.shape[1:], fill_value=fill_value
    )
    for i_l in range(r.size - 1):
        if verbose:
            print(f"{i_l + 1:02d}/{r.size - 1:02d}")
        sel = (r[i_l] <= rlb[0]) & (rlb[0] < r[i_l + 1])
        r_slc, (l_slc, b_slc) = rlb[0, sel], np.degrees(rlb[1:, sel])
        v_l = get_interp_val(
            hp_map[..., i_l, :], l_slc, b_slc, nest=nest, lonlat=True
        )
        v_r = get_interp_val(
            hp_map[..., i_l + 1, :], l_slc, b_slc, nest=nest, lonlat=True
        )
        # DIY radial interpolation with broadcasting
        w_r = (r_slc - r[i_l]) / (r[i_l + 1] - r[i_l])
        assert np.all(w_r >= 0.) and np.all(w_r <= 1.)
        rg_map[...,
               sel.reshape(new_pos.shape[1:])] = (1. - w_r) * v_l + w_r * v_r
    return rg_map


def copy_header_attrs(hdu_new, hdu_old):
    hdu_old_header = hdu_old if isinstance(
        hdu_old, fits.Header
    ) else hdu_old.header
    for k in PRESERVED_HEADER_ATTRS:
        if k in hdu_old_header:
            hdu_new.header.set(k, hdu_old_header[k])


# ----------------------------------------------------------------------------
# from format_clouds.py: restores the aligned layout of the Zucker (2020) cloud table.
# ----------------------------------------------------------------------------
# Exact header line from the original file (its d_mean/d_std labels carry a
# leading space, so it is stored verbatim rather than derived).
CATALOG_HEADER = (
    "cloud           ,l_min(deg),l_max(deg),b_min(deg),b_max(deg),"
    " d_mean(pc), d_std(pc),n_sightlines"
)

# Per-column layout for data rows: (width, justify). "<" = left, ">" = right.
CATALOG_COLSPECS = [
    (16, "<"),  # cloud
    (10, ">"),  # l_min(deg)
    (10, ">"),  # l_max(deg)
    (10, ">"),  # b_min(deg)
    (10, ">"),  # b_max(deg)
    (9, ">"),   # d_mean(pc)
    (7, ">"),   # d_std(pc)
    (5, ">"),   # n_sightlines
]


def catalog_format_row(cells):
    out = []
    for cell, (width, justify) in zip(cells, CATALOG_COLSPECS):
        out.append("{0:{1}{2}}".format(cell, justify, width))
    return ",".join(out)


def catalog_align(text):
    lines = text.splitlines()
    rows = [ln.split(",") for ln in lines if ln.strip()]
    # Drop the incoming header (row 0) and re-emit the canonical one.
    body = [catalog_format_row([c.strip() for c in r]) for r in rows[1:]]
    return "\n".join([CATALOG_HEADER, *body]) + "\n"


def format_clouds_main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    with open(src) as f:
        result = catalog_align(f.read())
    with open(dst, "w") as f:
        f.write(result)
    print("Aligned {0} -> {1}".format(src, dst))


# ----------------------------------------------------------------------------
# from strip_clouds.py: strips the padding from the Zucker (2020) cloud table.
# ----------------------------------------------------------------------------
def catalog_strip(text):
    lines = text.splitlines()
    out = [
        ",".join(cell.strip() for cell in ln.split(","))
        for ln in lines
        if ln.strip()
    ]
    return "\n".join(out) + "\n"


def strip_clouds_main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    with open(src) as f:
        result = catalog_strip(f.read())
    with open(dst, "w") as f:
        f.write(result)
    print("Stripped {0} -> {1}".format(src, dst))
