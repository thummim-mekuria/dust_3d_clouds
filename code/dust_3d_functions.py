

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

plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size']   = 15
plt.rcParams['mathtext.fontset'] = 'stix'


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


def dparl_pt_line(p0s, p1, p2):
    # Signed scalar projection onto the spine: negative behind p1.
    num = np.sum((p1[np.newaxis, :]-p0s)*(p2-p1), axis=-1)
    den = np.linalg.norm(p2-p1)
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


def get_cloud_info(i, source_array, drange='by r'):
    # Depends on module-level globals: table1, ds, drs.
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


def _fit_one_line(dats_arr, seed):
    # One Monte-Carlo spine fit: random point pair -> L1-minimized line.
    # Pure function of (dats_arr, seed) so it can run in a parallel worker.
    rng = np.random.RandomState(seed)
    p1, p2 = pick_p1p2(dats_arr, rng)

    ps_min = minimize(L1ds_minimizable, np.concatenate([p1, p2]), args=dats_arr)

    p1_ = ps_min.x[0:3]
    p2_ = ps_min.x[3:6]

    L1d_min, _ = L1ds(p1_, p2_, dats_arr)

    return [L1d_min, p1_[0], p1_[1], p1_[2], p2_[0], p2_[1], p2_[2]]


def pick_best_line(i, source_array, drange, n_iter, n_jobs=1):
    # Each iteration is an independent random-line fit. n_jobs=1 keeps the original
    # serial behavior; n_jobs=-1 uses all cores, or pass a positive int. Seeds are
    # 0..n_iter-1 so results are reproducible regardless of n_jobs.
    dats = get_99th_percs(i, source_array, drange, which='both')
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

def get_spine(i, source_array, drange='by r', n_iter=5, n_jobs=1):
    # Source notebook returned `np.around(pt_lbd,2),n` -- `n` is undefined here,
    # likely a leftover from pick_best_line's loop variable.
    dats = get_99th_percs(i, source_array, drange, which='both')

    p1_xyz, p2_xyz, dmin = pick_best_line(i, source_array, drange, n_iter, n_jobs=n_jobs)

    # pt_, vec_ = pts2line(p1_xyz,p2_xyz) #increase the distance between the two points 
    # p1_xyz, p2_xyz = line2pts(pt_,vec_,10)

    L1d, all_dat = L1ds(p1_xyz, p2_xyz, dats[1].to_numpy())

    all_df = pd.DataFrame(all_dat, columns=['x', 'y', 'z', 'd', 'b', 'l', 'n', 'dperp', 'dparl'])
    cube2  = coords_to_cube(all_df[['d', 'b', 'l']].to_numpy(), all_df['dperp'])
    cube22 = coords_to_cube(all_df[['d', 'b', 'l']].to_numpy(), all_df['dparl'])

    cube3  = np.flip(cube2.transpose(), axis=2)  # reorients to look like cube
    cube33 = np.flip(cube22.transpose(), axis=2)

    p1_lbd = np.array(xyz_2_sph(p1_xyz))
    p2_lbd = np.array(xyz_2_sph(p2_xyz))

    pt_lbd, vec_lbd = pts2line(p1_lbd, p2_lbd)

    return np.around(pt_lbd, 2), np.around(vec_lbd, 2), cube3, cube33, L1d


def input_spine(i, source_array, sp_type='lbd'):
    # Never returns anything -- as in the source notebook. Depends on global `spines`.
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

    all_df = pd.DataFrame(alldat, columns=['x', 'y', 'z', 'd', 'b', 'l', 'A', 'dperp', 'dparl'])
    cube2 = coords_to_cube(all_df[['d', 'b', 'l']].to_numpy(), all_df['dperp'])


def ha(i, source_array, drange, n='none', n_jobs=1):
    # `b_perp` and `dist_cube` are the same array; `dist_cube` is kept because
    # histogram, Sig_radial_profiles, make_Av_isosurface and
    # rho_radial_profiles still read it under that name.
    # n_jobs is forwarded to the Monte-Carlo spine fit (see pick_best_line).
    cube, coords = get_cloud_info(i, source_array, drange)
    if type(n) != str:
        cube = cube*(cube > n)

    pta, veca, dperp, dparl, L1da = get_spine(i, source_array, drange, n_jobs=n_jobs)

    return {'coords': coords, 'dust_cube': cube,'b_perp': dperp, 'b_parl': dparl,
            'spine': [pta, veca]}


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
    lbax, bdax, ldax, _ = axs.flat 
    pt, vect = spine

    kws = {'color': scolor, 'pivot': 'mid', 'scale': 1,
           'angles': 'xy', 'width': 2e-2, 'alpha': 0.7, 'headwidth': 0.}

    lbax.quiver(pt[0], pt[1], vect[0], vect[1], **kws)
    ldax.quiver(pt[0], pt[2], vect[0], vect[2], **kws)
    bdax.quiver(pt[2], pt[1], vect[2], vect[1], **kws)


def plot_moment_maps(cube, coords, which, unit='Av', scolor='cyan',
                     b_perp=None, b_parl=None):
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

    fig, axs = plt.subplots(2, 2, dpi=200, figsize=(10, 10))#,sharey='row',sharex='col')

    cmp = 'binary'

    kws = {'levels': 40, 'cmap': cmp, 'extend': 'max', 'norm': 'log'}
    kws2 = {'levels': lvls, 'colors': 'white', 'linewidths': 0.75,
            'linestyles': ['solid', '--', ':', 'solid']}
    
    lbax, bdax, ldax, ax0 = axs.flat

    lbax.contourf(im0, extent=coords[:4], **kws)
    lbax.contour(im0, extent=coords[:4], **kws2)

    ldax.contourf(im1, extent=coords[[0, 1, 4, 5]], **kws)
    ldax.contour(im1, extent=coords[[0, 1, 4, 5]], **kws2)

    bdax.contourf(im2.transpose(), extent=coords[[4,5,2,3]], **kws)
    bdax.contour(im2.transpose(), extent=coords[[4,5,2,3]], **kws2)

    for arr, ls in [(b_perp, 'solid'), (b_parl, '--')]:
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

    for ax in [lbax,ldax]:
        ax.set_xlim(coords[:2])

    lbax.set_ylim(coords[2:4])
    ldax.set_ylim(coords[4:])

    bdax.set_xlim(coords[4:])
    bdax.set_ylim(coords[2:4])

    # lbax.set_aspect(1)
    ax0.axis('off') 

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


def check_thres(i, source_array, r, thresh):
    cube, coords = get_cloud_info(i, source_array, r)

    full_cube = get_cloud_info(i, source_array, 'full')[0]
    fullc_Av = ((np.nansum(full_cube*u.cm**-3, axis=0)*u.pc).cgs/(2.2e21*u.cm**-2)).value

    pta, veca, dperp, dparl, L1da = get_spine(i, source_array, r)

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


def rho_spinal_profiles(i, source_array, drange='by r', plt_='new', color='black', lbl='by r'):
    # BROKEN AS PORTED: curve_fit raises "`ydata` must not be empty!". b_parl is a
    # signed projection in pc, so it spans negative values and the log-spaced rbins
    # below cannot bin it. Left as-is to match 17_.ipynb cell 44.
    if lbl == 'by r':
        lbl = f'r={round(drs[i], 2)}'

    cloud_dict = ha(i, source_array, drange)

    rmax = np.nanmax(cloud_dict['b_parl'])
    px_scl = np.log10(2*0.125*u.deg.to(u.rad)*ds[i])
    rbins = 10**np.arange(px_scl, np.log10(rmax)+0.2, 0.1)

    df = bin_by_r(cloud_dict['b_parl'], cloud_dict['dust_cube'], rbins)
    mins = find_peaks(-df['rho'])

    rho0, r0, alpha = np.around((curve_fit(eq1_if, df['d'], df['rho'], p0=[10, 10, 3], maxfev=5000))[0], 2)

    # -------------------------------------------------------------------

    if plt_ == 'new':
        plt.figure()
        ax = plt.gca()
    else:
        ax = plt_

    s = 100
    ax.scatter(df['d'], df['rho'], color=color, marker='.', label=lbl, s=s)
    ax.scatter(df['d'].iloc[mins[0]], df['rho'].iloc[mins[0]], color=color, marker='+', s=s)
    ax.plot(df['d'], eq1_if(df['d'], rho0, r0, alpha), color=color)

    err_sum = np.nansum(np.abs(eq1_if(df['d'], rho0, r0, alpha)-df['rho']))/np.nansum(df['rho'])

    ax.loglog()
    ax.set_xlabel(r'$d_\parallel\ [pc]$')
    ax.set_ylabel(r'$n\ (cm^{-3})$')
    ax.set_xlim(1, rmax+5)
    ax.set_title(names[i])

    print('rho0', rho0, 'r0', r0, 'alpha', alpha)

    return plt.gca(), [rho0, r0, alpha]


def rho_radial_profiles_compute(i, source_array, drange='by r', tube_r='px', n_jobs=1):
    # Pure compute half of rho_radial_profiles: no matplotlib, returns a picklable
    # dict, so it can run inside a Parallel(...) over clouds and be plotted later.
    # n_jobs forwards to the Monte-Carlo spine fit inside ha (see pick_best_line):
    # pass n_jobs=-1 to fit one cloud across all cores. Leave at 1 when this call
    # is itself already inside an outer Parallel over clouds, to avoid nesting.
    # Depends on module-level globals ds, drs.
    # Only d_perp is fit with eq1_if -- the Plummer-like form has no counterpart
    # along the spine. d_parl is signed, so each side of the spine is binned on its
    # own and both are drawn against |d_parl|, solid ahead of p1 and dashed behind.
    # `tube_r` is the radius in pc of the cylinder the d_parl profiles average
    # within -- 'px' for the pixel scale, or np.inf for no cut.
    cloud_dict = ha(i, source_array, drange, n_jobs=n_jobs)

    log_px_scl = np.log10(2*0.125*u.deg.to(u.rad)*ds[i])

    b_perp_bins = 10**np.arange(log_px_scl,
                          np.log10(np.nanmax(cloud_dict['b_perp']))+0.2, 0.1)

    df = bin_by_r(cloud_dict['b_perp'], cloud_dict['dust_cube'], b_perp_bins)
    mins = find_peaks(-df['rho'])

    rho0, r0, alpha = np.around((curve_fit(eq1_if, df['d'], df['rho'], p0=[10, 10, 3], maxfev=5000))[0], 2)

    # d_parl is signed about p1, so bin each side of the spine on its own and
    # plot both against |d_parl| -- one branch per side, per color. Only pixels
    # inside a tube of radius tube_r about the spine are averaged in; d_perp is
    # nan off the cloud, and those compare False, so they drop out here too.
    if tube_r == 'px':
        tube_r = 3*10**log_px_scl

    tube = np.ravel(cloud_dict['b_perp']) <= tube_r
    tube_n = np.ravel(cloud_dict['dust_cube'])[tube]

    parl_branches = []
    for sign, ls in [(1, 'solid'), (-1, '--')]:
        signed = sign*np.ravel(cloud_dict['b_parl'])[tube]
        far = np.nanmax(signed)
        if not np.isfinite(far) or far <= 10**log_px_scl:
            continue
        df_t = bin_by_r(signed, tube_n,
                        10**np.arange(log_px_scl, np.log10(far)+0.2, 0.1))
        parl_branches.append((df_t, find_peaks(-df_t['rho']), ls, sign))

    return {'i': i, 'df': df, 'mins': mins, 'fit': [rho0, r0, alpha],
            'parl_branches': parl_branches}


def rho_radial_profiles_plot(result, plt_='new', color='black', lbl='by r'):
    # Drawing half of rho_radial_profiles -- consumes a dict from _compute. Kept
    # serial (matplotlib is not process-safe). `plt_` is 'new' or an axes array
    # unpacking into (perp_ax, parl_ax); the returned array feeds back in to overplot.
    i = result['i']
    df = result['df']
    mins = result['mins']
    rho0, r0, alpha = result['fit']
    parl_branches = result['parl_branches']

    if lbl == 'by r':
        lbl = f'r={round(drs[i], 2)}'

    if type(plt_) == str:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True,sharex=True,sharey=True)
        fig.suptitle(names[i])
    else:
        axs = plt_

    perp_ax, parl_ax = axs.flat

    s = 100

    perp_ax.scatter(df['d'], df['rho'], color=color, marker='.', label=lbl, s=s)
    perp_ax.scatter(df['d'].iloc[mins[0]], df['rho'].iloc[mins[0]], color=color, marker='+', s=s)
    perp_ax.plot(df['d'], eq1_if(df['d'], rho0, r0, alpha), color=color)

    err_sum = np.nansum(np.abs(eq1_if(df['d'], rho0, r0, alpha)-df['rho']))/np.nansum(df['rho'])

    perp_ax.loglog()
    perp_ax.set_xlabel(r'$d_\perp\ [\mathrm{pc}]$')
    perp_ax.set_ylabel(r'$n\ (cm^{-3})$')
    # perp_ax.set_xlim(1, np.nanmax(cloud_dict['b_perp'])+5)


    for df_t, mins_t, ls, sign in parl_branches:
        parl_ax.plot(df_t['d'], df_t['rho'], color=color, linestyle=ls, label=lbl if sign > 0 else None)
        parl_ax.scatter(df_t['d'].iloc[mins_t[0]], df_t['rho'].iloc[mins_t[0]], color=color, marker='+', s=s)

    parl_ax.loglog()
    parl_ax.set_xlabel(r'$|d_\parallel|\ [\mathrm{pc}]$')
    parl_ax.set_ylabel(r'$n\ (cm^{-3})$')
    # parl_ax.set_xlim(1, np.nanmax(cloud_dict['b_parl'])+5)

    parl_ax.set_xlim(1,)
    parl_ax.set_ylim(1e-1,)


    # print('rho0', rho0, 'r0', r0, 'alpha', alpha)

    return axs, [rho0, r0, alpha]


def rho_radial_profiles(i, source_array, drange='by r', cmapping='sane', plt_='new', color='black', lbl='by r',
                        tube_r='px', n_jobs=1):
    # Backward-compatible wrapper: compute + plot in one call, same signature and
    # return as before. For parallel runs call rho_radial_profiles_compute across
    # clouds under a Parallel(...), then rho_radial_profiles_plot on each result.
    result = rho_radial_profiles_compute(i, source_array, drange, tube_r, n_jobs)
    return rho_radial_profiles_plot(result, plt_, color, lbl)




def Sig_radial_profiles(i, source_array, drange='by r', cmapping='sane', plt_='new', color='black'):
    # Never returns anything -- as in the source notebook. Depends on global `names`.
    cloud_dict = ha(i, source_array, drange)

    rbins = 10**np.arange(-0.5, 2, 0.1)

    msk = 1
    dperps = [np.nanmin(cloud_dict['dist_cube'], axis=n) for n in range(3)]  # lb,ld,bd projections
    Sigs = [np.nansum(cloud_dict['dust_cube'], axis=n) for n in range(3)]

    if type(plt_) == str:
        fig, axs = plt.subplots(1, 3, figsize=(15, 7), sharey=True, sharex=True)
        fig.suptitle(names[i])


# def make_Av_isosurface(i, source_array, r='by r'):
#     dict_ = ha(i, source_array, r)

#     Av0 = ((np.nansum(dict_['dust_cube']*u.cm**-3, axis=0)*u.pc).cgs/(2.2e21*u.cm**-2)).value
#     Av1 = ((np.nansum(dict_['dust_cube']*u.cm**-3, axis=1)*u.pc).cgs/(2.2e21*u.cm**-2)).value
#     Av2 = ((np.nansum(dict_['dust_cube']*u.cm**-3, axis=2)*u.pc).cgs/(2.2e21*u.cm**-2)).value

#     m0 = np.broadcast_to(Av0 > 1, dict_['dust_cube'].shape)
#     m1 = np.broadcast_to(Av1 > 1, np.array(dict_['dust_cube'].shape)[[1, 0, 2]]).swapaxes(0, 1)
#     m2 = np.broadcast_to(Av2 > 1, np.array(dict_['dust_cube'].shape)[[2, 0, 1]]).swapaxes(0, 2).swapaxes(0, 1)

#     miso = m0*m1*m2

#     capped = miso*dict_['dist_cube']

#     return miso
