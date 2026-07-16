"""
Functions extracted from 21_serious_time.ipynb.

table1/ds/drs/names are loaded at module level below. The dust cube is passed
in explicitly as `source_array` rather than read from a global.

`spines` is still an unresolved module-level global -- `input_spine` needs it
and nothing here defines it.

`eq2_if` below is a leftover from 17_.ipynb, has no counterpart in
21_serious_time.ipynb, and was syntactically incomplete where it came from,
so it stays commented out.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from astropy.io import fits
import astropy.units as u
import astropy.constants as c

from astropy.modeling import models, fitting
fttr = fitting.LevMarLSQFitter()
gmod = models.Gaussian1D(1., 0., 1.,)

from scipy.optimize import minimize, curve_fit
from scipy.signal import find_peaks

from joblib import Parallel, delayed


# ---------------------------------------------------------------------------

data_loc = '../../../data/'
table1 = pd.read_csv(data_loc+'mif_tab1_final.csv',delimiter=',')

ds    = table1['d(pc)'].to_numpy().astype('float')
names = table1['Name']

drs = np.nanmax((np.nanstd(table1[['l0(°)', 'l1(°)']],axis=1),
                 np.nanstd(table1[['b0(°)', 'b1(°)']],axis=1)),axis=0) *u.deg.to(u.rad)*ds

# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# general helpers / unit conversions
# ---------------------------------------------------------------------------

mag_per_pc = lambda d: (c.m_p*2.2e21*u.cm**-2/(d*u.pc)).to(u.Msun/(u.pc**3))  # integrated over a parsec
n_to_rho   = lambda n: (n.cgs*c.m_p).to(u.Msun/(u.pc**3))
A_to_n     = lambda A: ((A*u.mag/u.pc)*(2.2e21*u.cm**-2/u.mag)).cgs
pts2line   = lambda p1, p2: ((p1+p2)/2, p2-p1)
line2pts   = lambda pt, vec, t=1: (pt + t*vec, pt - t*vec)

dist       = lambda p1, p2: np.sqrt(np.nansum((p1-p2)**2))

coord_idx  = lambda val, dx, x0: ((np.array(val) - x0)/dx).astype('int')


# ---------------------------------------------------------------------------
# equations
# ---------------------------------------------------------------------------

def eq1_if(r, rho_0, ro, alpha):
    num = rho_0
    den = (1 + (r/ro)**2)
    return num/(den**alpha)


# BROKEN IN SOURCE NOTEBOOK: undefined `r`/`rho_of_r(r)`, and the source cell
# was truncated mid-expression ("num = rho_of_r(r)*r*").
# def eq2_if(b_perp, rho_0, ro, alpha, rT):
#     #produce array of rs from bperp
#
#     rho_of_r = eq1_if(b_perp, rho_0, ro, alpha)
#
#     num = rho_of_r(r)*r*
#     den = (1 + (r/ro)**2)
#     return num/(den**alpha)


# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------

def get_subcube(bigcube, coords, xyz0s=[181, 90.6, 68], dxyzs=[-0.23, 0.23, 1]):
    dx, dy, dz = dxyzs
    x0, y0, z0 = xyz0s

    lo, li, bo, bi, do, di = coords

    xo, xi = coord_idx([lo, li], dx, x0)
    yo, yi = coord_idx([bo, bi], dy, y0)
    zo, zi = coord_idx([do, di], dz, z0)

    return bigcube[zo:zi, yo:yi, xo:xi]


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


def get_cloud_info(i, source_array, drange='by r'):
    """Depends on module-level globals: table1, ds, drs."""
    lb = table1.iloc[i][['l0(°)', 'l1(°)', 'b0(°)', 'b1(°)']].to_numpy().astype('float')

    if drange == 'by r':
        coords = np.concatenate((lb, np.array([ds[i]-drs[i], ds[i]+drs[i]])))

    elif drange == 'full':
        coords = np.concatenate((lb, np.array([0, 1182])+68))
    else:
        coords = np.concatenate((lb, np.array([ds[i]-drange, ds[i]+drange])))

    coords[4] = max(coords[4], 68)
    
    cube = get_subcube(source_array, coords)

    return cube, coords


def get_99th_percs(i, source_array, drange='by r', which='99', db=0.23, dl=-0.23):
    cube, coords = get_cloud_info(i, source_array, drange)

    l0, b0, d0 = coords[[0, 2, 4]]
    ds_, bs, ls = np.indices(cube.shape).reshape(3, -1) * np.array([1, db, dl])[:, None] + np.array([d0, b0, l0])[:, None]
    xs, ys, zs = sph_2_xyz([ls, bs, ds_])

    df = pd.DataFrame(np.column_stack([xs, ys, zs, ls, bs, ds_, np.ravel(cube)]), columns=['x', 'y', 'z', 'l', 'b', 'd', 'n'])
    df2 = df.dropna().query('n > 1')  # choose pixels w data above this thresh

    per = np.nanpercentile(df2['n'], 99)  # 99th percentile

    df99 = df2.query('n > {}'.format(per))
    df99_1 = df.query('n > {}'.format(per))

    if which == '99':
        return df99
    elif which == 'both':
        return df99, df
    else:
        return df.to_numpy()


def pick_p1p2(coords99):
    xyz = coords99[:, 0:3]

    N = len(xyz)
    i1, i2 = np.random.randint(N, size=2)
    while i1 == i2:  # picking the same point
        i2 = np.random.randint(N)

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
    L1_d = np.nansum(dperps*Avs)/np.nansum(Avs)

    return L1_d, np.concatenate((pixels, dperps[:, np.newaxis]), axis=1)


def pick_best_line(i, source_array, drange, n_iter):
    dats = get_99th_percs(i, source_array, drange, which='both')

    df = pd.DataFrame(columns=['d', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2'])

    for n in range(n_iter):
        p1, p2 = pick_p1p2(dats[0].to_numpy())

        ps_min = minimize(L1ds_minimizable, np.concatenate([p1, p2]), args=dats[0].to_numpy())

        p1_ = ps_min.x[0:3]
        p2_ = ps_min.x[3:6]

        L1d_min, all_ds = L1ds(p1_, p2_, dats[0].to_numpy())

        df.loc[len(df)] = [L1d_min, p1_[0], p1_[1], p1_[2], p2_[0], p2_[1], p2_[2]]

    df = df.sort_values('d')
    dmin = df.iloc[0]['d']
    p1 = df.iloc[0][['x1', 'y1', 'z1']].to_numpy()
    p2 = df.iloc[0][['x2', 'y2', 'z2']].to_numpy()

    return p1,p2,dmin

def get_spine(i, source_array, drange='by r', n_iter=5):
    """NOTE: source notebook's `return np.around(pt_lbd,2),n` references `n`,
    which is not defined in this function's scope (likely a bug -- `n` looks
    like a leftover from `pick_best_line`'s loop variable)."""
    dats = get_99th_percs(i, source_array, drange, which='both')

    p1_xyz, p2_xyz, dmin = pick_best_line(i, source_array, drange, n_iter)

    L1d, all_dat = L1ds(p1_xyz, p2_xyz, dats[1].to_numpy())

    all_df = pd.DataFrame(all_dat, columns=['x', 'y', 'z', 'd', 'b', 'l', 'n', 'dperp'])
    cube2 = coords_to_cube(all_df[['d', 'b', 'l']].to_numpy(), all_df['dperp'])

    cube3 = np.flip(cube2.transpose(), axis=2)  # reorients to look like cube

    p1_lbd = np.array(xyz_2_sph(p1_xyz))
    p2_lbd = np.array(xyz_2_sph(p2_xyz))

    pt_lbd, vec_lbd = pts2line(p1_lbd, p2_lbd)

    return np.around(pt_lbd, 2), np.around(vec_lbd,2),cube3,L1d 


def input_spine(i, source_array, sp_type='lbd'):
    """NOTE: source notebook never returns from this function; depends on
    module-level global `spines`."""
    p1, p2 = spines[i]
    dats = get_99th_percs(i, source_array, which='both')

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

    all_df = pd.DataFrame(alldat, columns=['x', 'y', 'z', 'd', 'b', 'l', 'A', 'dperp'])
    cube2 = coords_to_cube(all_df[['d', 'b', 'l']].to_numpy(), all_df['dperp'])


def ha(i, source_array, drange, n='none'):
    cube, coords = get_cloud_info(i, source_array, drange)
    if type(n) != str:
        cube = cube*(cube > n)

    pta, veca, cuba, L1da = get_spine(i, source_array, drange)

    return {'coords': coords, 'dust_cube': cube, 'dist_cube': cuba, 'spine': [pta, veca]}


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

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
    axs[0].scatter(df['l'], df['b'], color='red', s=2)
    axs[1].scatter(df['l'], df['d'], color='red', s=2)
    axs[2].scatter(df['b'], df['d'], color='red', s=2)


def plot_spine(axs, spine, scolor='cyan'):
    pt, vect = spine

    kws = {'color': scolor, 'pivot': 'mid', 'scale': 4,
           'angles': 'xy', 'width': 3e-2, 'alpha': 0.7, 'headwidth': 0.}

    axs[0].quiver(pt[0], pt[1], vect[0], vect[1], **kws)
    axs[1].quiver(pt[0], pt[2], vect[0], vect[2], **kws)
    axs[2].quiver(pt[1], pt[2], vect[1], vect[2], **kws)


def plot_moment_maps(cube, coords, which, unit='Av', scolor='cyan'):
    if unit == 'Av':
        im0 = ((np.nansum(cube*u.cm**-3, axis=0)*u.pc).cgs/(2.2e21*u.cm**-2)).value
        im1 = ((np.nansum(cube*u.cm**-3, axis=1)*u.pc).cgs/(2.2e21*u.cm**-2)).value
        im2 = ((np.nansum(cube*u.cm**-3, axis=2)*u.pc).cgs/(2.2e21*u.cm**-2)).value

        lvls = [0.5, 1]
        cmin = 0.25
        cmax = 2.1
    else:
        im0 = ((np.nansum(cube*u.cm**-3, axis=0)*u.pc)*c.m_p).to(u.Msun/u.pc**2).value
        im1 = ((np.nansum(cube*u.cm**-3, axis=1)*u.pc)*c.m_p).to(u.Msun/u.pc**2).value
        im2 = ((np.nansum(cube*u.cm**-3, axis=2)*u.pc)*c.m_p).to(u.Msun/u.pc**2).value

        lvls = [10, 20, 50, 70]
        cmin = 1
        cmax = 20

    fig, axs = plt.subplots(1, 3, dpi=200, figsize=(15, 5))

    cmp = 'binary'

    kws = {'levels': 40, 'cmap': cmp, 'extend': 'max', 'norm': 'log'}
    kws2 = {'levels': lvls, 'colors': 'white', 'linewidths': 0.75,
            'linestyles': ['solid', '--', ':', 'solid']}

    axs[0].contourf(im0, extent=coords[:4], **kws)
    axs[0].contour(im0, extent=coords[:4], **kws2)

    axs[1].contourf(im1, extent=coords[[0, 1, 4, 5]], **kws)
    axs[1].contour(im1, extent=coords[[0, 1, 4, 5]], **kws2)

    axs[2].contourf(im2, extent=coords[2:], **kws)
    axs[2].contour(im2, extent=coords[2:], **kws2)

    for ax in axs[:2]:
        ax.set_xlim(coords[:2])

    axs[0].set_ylim(coords[2:4])
    axs[2].set_xlim(coords[2:4])

    for ax in axs[1:]:
        ax.set_ylim(coords[4:])

    axs[0].set_aspect(1)

    fig.suptitle(which)

    return axs


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
    """NOTE: references `name`, which is not defined in this function's scope."""
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


def check_thres(i, source_array, r, thresh):
    cube, coords = get_cloud_info(i, source_array, r)

    full_cube = get_cloud_info(i, source_array, 'full')[0]
    fullc_Av = ((np.nansum(full_cube*u.cm**-3, axis=0)*u.pc).cgs/(2.2e21*u.cm**-2)).value

    pta, veca, cuba, L1da = get_spine(i, source_array, r)

    p1a_xyz, p2a_xyz = line2pts(pta, veca)
    p1a_lbd = np.array(xyz_2_sph(p1a_xyz))
    p2a_lbd = np.array(xyz_2_sph(p2a_xyz))

    pta_lbd, veca_lbd = pts2line(p1a_lbd, p2a_lbd)

    df = get_99th_percs(i, source_array)

    masked = cube * (cube > thresh)

    axs = plot_moment_maps(masked, coords, '{}'.format(thresh))
    axs[0].contour(fullc_Av, extent=coords[:4], levels=[1], colors='orange')

    plt.show()


def histogram(i, source_array, drange='by r', cmapping='sane'):
    cloud_dict = ha(i, source_array, drange)

    full_cube = get_cloud_info(i, source_array, 'full')[0]
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


def rho_radial_profiles(i, source_array, drange='by r', cmapping='sane', plt_='new', color='black', lbl='by r'):
    """NOTE: source notebook never returns from this function (falls off the
    end after the curve_fit call); depends on module-level globals ds, drs.
    Callers elsewhere in the notebook unpack this as `ax, err_sum = ...`,
    which will fail since nothing is returned."""
    if lbl == 'by r':
        lbl = f'r={round(drs[i], 2)}'

    cloud_dict = ha(i, source_array, drange)

    rmax = np.nanmax(cloud_dict['dist_cube'])
    px_scl = np.log10(2*0.125*u.deg.to(u.rad)*ds[i])
    rbins = 10**np.arange(px_scl, np.log10(rmax)+0.2, 0.1)

    msk = 1
    df = bin_by_r(cloud_dict['dist_cube'], cloud_dict['dust_cube'], rbins)
    mins = find_peaks(-df['rho'])

    rho0, r0, alpha = np.around((curve_fit(eq1_if, df['d'], df['rho'], p0=[10, 10, 3], maxfev=5000))[0], 2)


def Sig_radial_profiles(i, source_array, drange='by r', cmapping='sane', plt_='new', color='black'):
    """NOTE: source notebook never returns from this function; depends on
    module-level global `names`."""
    cloud_dict = ha(i, source_array, drange)

    rbins = 10**np.arange(-0.5, 2, 0.1)

    msk = 1
    dperps = [np.nanmin(cloud_dict['dist_cube'], axis=n) for n in range(3)]  # lb,ld,bd projections
    Sigs = [np.nansum(cloud_dict['dust_cube'], axis=n) for n in range(3)]

    if type(plt_) == str:
        fig, axs = plt.subplots(1, 3, figsize=(15, 7), sharey=True, sharex=True)
        fig.suptitle(names[i])


def make_Av_isosurface(i, source_array, r='by r'):
    dict_ = ha(i, source_array, r)

    Av0 = ((np.nansum(dict_['dust_cube']*u.cm**-3, axis=0)*u.pc).cgs/(2.2e21*u.cm**-2)).value
    Av1 = ((np.nansum(dict_['dust_cube']*u.cm**-3, axis=1)*u.pc).cgs/(2.2e21*u.cm**-2)).value
    Av2 = ((np.nansum(dict_['dust_cube']*u.cm**-3, axis=2)*u.pc).cgs/(2.2e21*u.cm**-2)).value

    m0 = np.broadcast_to(Av0 > 1, dict_['dust_cube'].shape)
    m1 = np.broadcast_to(Av1 > 1, np.array(dict_['dust_cube'].shape)[[1, 0, 2]]).swapaxes(0, 1)
    m2 = np.broadcast_to(Av2 > 1, np.array(dict_['dust_cube'].shape)[[2, 0, 1]]).swapaxes(0, 2).swapaxes(0, 1)

    miso = m0*m1*m2

    capped = miso*dict_['dist_cube']

    return miso
