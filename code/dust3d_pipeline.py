"""
dust3d_pipeline -- run the dust_3d cloud pipeline, or any part of it, with the
functions in dust3d_lib.py.

Steps, in the order they run (each reads what the earlier ones wrote):

  extract      re-extract the padded per-cloud posterior-sample cubes from the
               19.5 GB Edenhofer samples file (slow; opt-in, never in 'all')
  fits         distance windows, spine angles and per-sample Plummer fits
               -> spine_angles_and_fits_samples_n<N>.csv (+ projmaps figures)
  cylinder     40-seed spines (cached in _spine_fits/), cylinder profiles and fits
               -> <cloud>_cylinder_three_row_n<N>.png, radial_profile_fits_n<N>.csv,
                  radial_profiles_n<N>.csv, stacked_profiles_n<N>.png
  windows      -> <cloud>_sample_spine_windows.png
  inputs       rho0/r0/alpha/r_dip + inclination + A_V-contour masses
               -> rho0_inputs_n<N>_linfit.csv
  rho0         -> rho0_correlations_n<N>_linfit_cosi_Mcyl.png (ours and IF23,
                  each with its own log-log fit; mass between the spine caps
                  out to r_dip), rho0_fit_slopes_n<N>.csv
  if23         -> if23_fig7_ours.png, if23_param_comparison.png (ours vs IF23 for the shared clouds)
  inclinations -> spine_inclination_panels_n<N>.png, spine_inclinations[_scatter].png
  projections  -> edenhofer_sky_projections_inclination.png,
                  edenhofer_xyz_topdown_mom0_inclination.png, projection_panels_xy_lb_bd.png,
                  projection_sky_lb.png
  ysos         -> sky_lb_ysos_n<N>.png
  paper        copy the paper's figures into github_repo/figures/ under the names
               main.tex uses, each from the newest run that has it (PAPER_FIGURES
               below); writes figures/fig_sources.csv (which run each came from)

  'all' = fits ... paper. The 10-13/14 experiments run only when named:
  saddle, aperture, leak, contours, ysos3 (sky_lb_ysos_contours3_n{5,10,15,20}.png),
  ysos_dmax (sky_lb_ysos_n<N>_dmax.png: full-axis max-n_H contour coloured by d of the max),
  xy_zmax (xy_ysos_n<N>_zmax.png: the face-on version, max along Z coloured by Z).

Examples (from any directory):
    python dust3d_pipeline.py                                  # everything, into dust_3d/local_only/runs/<today>/
    python dust3d_pipeline.py --steps cylinder windows --clouds Taurus Perseus
    # reuse an existing window table and spine fits instead of refitting:
    python dust3d_pipeline.py --steps cylinder inputs rho0 if23 \
        --win-csv ../../local_only/runs/2026-10-14/n10/spine_angles_and_fits_samples_n10.csv \
        --spine-cache ../../local_only/runs/2026-10-14/n10/_spine_fits
    python dust3d_pipeline.py --steps leak saddle             # the experiments
"""
import argparse
import datetime
import os
import sys

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd                                         # noqa: E402
import dust3d_lib as L                                     # noqa: E402

PIPELINE = ['fits', 'cylinder', 'windows', 'inputs', 'rho0', 'if23', 'inclinations',
            'projections', 'ysos', 'paper']

# The paper's figures: (file in github_repo/figures/, the run output it is copied
# from, the step that writes that output, where it goes in main.tex). {n} is
# the n_iso; single-cloud figures use PAPER_CLOUD.
PAPER_CLOUD = 'Perseus'          # the maps figure
PROFILE_CLOUD = 'Orion A'        # the single-cloud panel of the stacked profiles
PAPER_FIGURES = [
    ('fig1_catalog.png', 'projection_sky_lb.png', 'projections',
     'Cloud Sample: the clouds on the sky (l-b)'),
    ('fig2.png', '{cloud}_cylinder_three_row_n{n}_maps.png', 'cylinder',
     'Methods: l-b / l-d / b-d maps with the fitted spines'),
    ('fig3.png', 'spine_inclination_panels_n{n}.png', 'inclinations',
     'Methods (fig:inclinations): spine inclinations to the l-b, X-Y and b-d planes'),
    ('fig4.png', 'if23_param_comparison.png', 'if23',
     'Results (fig:if23compare): rho0/cos i, r0, alpha vs IF23 for the shared clouds'),
    ('fig5.png', 'stacked_profiles_n{n}.png', 'cylinder',
     'Methods (fig:profiles): one cloud d_perp profile, then every cloud stacked'),
    ('fig6.png', 'rho0_correlations_n{n}_linfit_cosi_Mcyl_r0_alpha.png', 'rho0',
     'Results (fig:params): rho0/cos i vs r0 and alpha, ours and IF23'),
    ('fig7.png', 'rho0_correlations_n{n}_linfit_cosi_Mcyl_M.png', 'rho0',
     'Results (fig:depletion): rho0/cos i vs M_cyl; the t_dep panel is not made yet'),
    ('fig9.png', 'if23_fig7_ours.png', 'if23',
     'Results (fig:scaling): M, projected length and r_dip vs r0'),
]


def newest_run_file(name):
    """Path of `name` in the newest dated run under local_only/runs/ (subfolders
    such as 2026-10-14/n10 count as their date), or None."""
    hits = []
    for d in sorted(os.listdir(L.RUNS)):
        if not d[:4].isdigit():
            continue
        for root, dirs, files in os.walk(os.path.join(L.RUNS, d)):
            dirs[:] = [x for x in dirs if not x.startswith('_')]
            if name in files:
                p = os.path.join(root, name)
                hits.append((d, os.path.getmtime(p), p))
    return max(hits)[2] if hits else None


def paper_figures(n):
    """Copy each PAPER_FIGURES source into github_repo/figures/ and record
    where it came from in figures/fig_sources.csv."""
    import shutil
    rows = []
    for dest, src, step, where in PAPER_FIGURES:
        name = src.format(cloud=PAPER_CLOUD, n=f'{n:g}')
        path = newest_run_file(name)
        if path is None:
            print(f'  {dest:18s} MISSING {name} -- run --steps {step}')
            continue
        shutil.copyfile(path, L.FIGURES + dest)
        rel = os.path.relpath(path, L.LOCAL)
        print(f'  {dest:18s} <- local_only/{rel}')
        rows.append(dict(figure=dest, source=f'local_only/{rel}', step=step, main_tex=where))
    pd.DataFrame(rows).to_csv(L.FIGURES + 'fig_sources.csv', index=False)
    print('wrote', L.FIGURES + 'fig_sources.csv')
EXPERIMENTS = ['saddle', 'aperture', 'leak', 'contours', 'ysos3', 'ysos_dmax', 'xy_zmax']
ALL_STEPS = ['extract'] + PIPELINE + EXPERIMENTS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--steps', nargs='+', default=['all'], choices=ALL_STEPS + ['all'])
    ap.add_argument('--outdir', default=None,
                    help='where everything is written (default dust_3d/local_only/runs/<today>/)')
    ap.add_argument('--n-iso', type=float, default=L.N_ISO,
                    help=f'cloud-defining density threshold, cm^-3 (default {L.N_ISO:g})')
    ap.add_argument('--clouds', nargs='+', default=None,
                    help='fits / cylinder / windows: only these clouds (default: the catalog)')
    ap.add_argument('--win-csv', default=None,
                    help='window + fit table to read (default: the one the fits step writes '
                         'in --outdir)')
    ap.add_argument('--spine-cache', default=None,
                    help='dir of <cloud>_spine_fits_n<N>_s40.pkl (default <outdir>/_spine_fits)')
    ap.add_argument('--refit', action='store_true', help='ignore cached spine fits')
    ap.add_argument('--no-projmaps', action='store_true',
                    help='fits: write the CSV only, not the projmaps_profiles figures')
    ap.add_argument('--no-spine-slab', action='store_true',
                    help='fits: fit d_perp on the whole box, not only between the spine ends')
    ap.add_argument('--extract-pad', type=float, default=L.PAD_STORED)
    ap.add_argument('--extract-outdir', default=L.CUBE_DIR)
    args = ap.parse_args()

    steps = [s for s in ALL_STEPS if s in args.steps or ('all' in args.steps and s in PIPELINE)]
    n = args.n_iso
    tag = f'n{n:g}'
    out = L._dir(args.outdir or os.path.join(L.RUNS, datetime.date.today().isoformat()))
    win_csv = args.win_csv or f'{out}spine_angles_and_fits_samples_{tag}.csv'
    spines = args.spine_cache or out + '_spine_fits'
    radial_csv = f'{out}radial_profile_fits_{tag}.csv'
    inputs_csv = f'{out}rho0_inputs_{tag}_linfit.csv'
    print(f'steps: {" ".join(steps)}\noutdir: {out}\nn_iso: {n:g}\n')

    if 'extract' in steps:
        L.extract_padded_sample_cubes(args.extract_outdir, pad=args.extract_pad)

    table1 = L.load_catalog()
    names = list(table1['Name'])
    print(f'[catalog] {len(table1)} clouds: {", ".join(names)}')

    needs_win = set(steps) - {'extract', 'fits'}
    if needs_win and 'fits' not in steps and not os.path.exists(win_csv):
        raise SystemExit(f'{win_csv} does not exist: run the fits step, or pass --win-csv')

    if 'fits' in steps:
        cubes = L.build_cubes(table1)
        _, _, sprofs = L._sample_profiles(table1)
        L.sample_summary_table(table1, cubes, sprofs, out, clouds=args.clouds, n_iso=n,
                               save_figures=not args.no_projmaps,
                               spine_slab_cut=not args.no_spine_slab)

    if 'cylinder' in steps:
        L.cylinder_figures(table1, win_csv, out, clouds=args.clouds, n_iso=n,
                           refit=args.refit, spine_cache=spines)
        L.stacked_profiles_figure(out, cloud=PROFILE_CLOUD, n_iso=n)

    if 'windows' in steps:
        L.spine_window_figures(table1, win_csv, out, clouds=args.clouds, n_iso=n,
                               refit=args.refit, spine_cache=spines)

    if 'inputs' in steps:
        L.rho0_inputs_table(win_csv, radial_csv, spines, inputs_csv, n_iso=n)
        L.av1_mass_columns(table1, win_csv, inputs_csv)

    if 'rho0' in steps:
        rows = []
        if 'M_cyl' not in pd.read_csv(inputs_csv).columns:
            print(f'  (no M_cyl column in {inputs_csv} -- run the cylinder and inputs steps)')
        else:
            fits = L.rho0_correlation_figures(out, inputs_csv, n_iso=n, mass_col='M_cyl',
                                              tag='_linfit_cosi_Mcyl')
            for col, f in fits.items():
                for who, (m, me, b, r2) in f.items():
                    rows.append(dict(y=col, data=who, slope=m, slope_err=me,
                                     intercept=b, r2=r2))
            pd.DataFrame(rows).round(4).to_csv(out + f'rho0_fit_slopes_n{n:g}.csv', index=False)

    if 'if23' in steps:
        L.if23_fig7_figure(inputs_csv, spines, out + 'if23_fig7_ours.png', tag=f'{tag}_s40')
        L.if23_param_comparison_figure(inputs_csv, out + 'if23_param_comparison.png')

    if 'inclinations' in steps:
        L.inclination_panels_figure(names, spines, f'{out}spine_inclination_panels_{tag}.png', n_iso=n)
        L.spine_inclination_figures(win_csv, out + 'spine_inclinations.png',
                                    scatter_out=out + 'spine_inclinations_scatter.png')

    if 'projections' in steps:
        L.sky_projection_figure(win_csv, spines, out, n_iso=n)
        L.xyz_projection_figure(win_csv, spines, out, n_iso=n)
        L.projection_panels_figure(win_csv, spines, out, n_iso=n)

    if 'ysos' in steps:
        L.sky_ysos_figure(win_csv, spines, out, n_iso=n)

    if 'paper' in steps:
        paper_figures(n)

    # ---- experiments ----
    if 'saddle' in steps:
        L.saddle_window_figures(table1, win_csv, out)
    if 'aperture' in steps:
        L.av1_aperture_figures(table1, win_csv, out, n_iso=n)
    if 'leak' in steps:
        L.leak_figures(table1, win_csv, out)
    if 'contours' in steps:
        L.av1_contour_figures(table1, out, blue='nicest')
        L.av1_contour_figures(table1, out, blue='niso', n_iso=n, win_csv=win_csv)
    if 'ysos3' in steps:
        L.sky_ysos_contours3_figures(win_csv, spines, out)
    if 'ysos_dmax' in steps:
        L.sky_ysos_dmax_figure(win_csv, spines, out, n_iso=n)
    if 'xy_zmax' in steps:
        L.xyz_ysos_zmax_figure(win_csv, spines, out, n_iso=n)


if __name__ == '__main__':
    main()
