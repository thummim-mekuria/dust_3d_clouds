# dust_3d_clouds

Cylinder models of nearby molecular clouds fitted to the Edenhofer et al. (2023)
3D dust map, and the paper that describes them (`main.tex`, synced with Overleaf).

## Layout

| Path | In git? | What it is |
|---|---|---|
| `main.tex`, `references.bib`, `aastex631.cls`, `aasjournal.bst` | yes | the paper (Overleaf syncs these) |
| `figures/fig*` | yes | the paper's figures; nothing else in `figures/` is tracked |
| `code/dust3d_lib.py` | yes | every calculation and plotting function; unused older functions sit in its LEGACY section at the end |
| `code/dust3d_pipeline.py` | yes | the driver: `python code/dust3d_pipeline.py --help` |
| `code/hpc/` | yes | dendrogram + SCIMES segmentation jobs for the Hummingbird cluster (own README) |
| `code/notes/` | yes | design notes the code refers to (e.g. why n_iso = 10 cm^-3) |
| `data/` | yes | small input tables: digitized Imara & Forbes (2023) values, Zucker (2020) cloud boxes, SFR/YSO catalogs |
| `requirements.txt` | yes | Python packages |

On disk this repo is `~/Documents/research/dust_3d/github_repo/`. Everything
machine-specific or large lives next to it in `dust_3d/local_only/`, outside the
repo, so it can never be pushed:

```
dust_3d/
├── research-notebook.md   lab notebook
├── github_repo/           this repository
└── local_only/            never pushed (below)
```

`local_only/`:

| Path | What it is |
|---|---|
| `cache/sample_cubes_pad8/` | per-cloud padded posterior-sample cubes (~9 GB); rebuild with `--steps extract` |
| `cache/healpix_*.npz` | HEALPix cube / profile caches, rebuilt on demand |
| `runs/<date>/` | pipeline output, one folder per run (the pipeline's default `--outdir` is `local_only/runs/<today>/`) |
| `notebooks/` | notebooks 21-25 and the `dust_3d_functions.py` they import |
| `lit_survey/` | the literature survey that built the cloud catalog |
| `archive/` | retired scripts (`legacy_scripts_2026-10-28.tar.gz`), one-off explorations |
| `not_rn/` | early notebooks 01-20 and their data (pre-dust_3d_clouds work) |
| `misc/` | loose figures and the p-v cut notebook from the old `dust_3d/` top level |
| `nicest_maps/` | NICEST maps, if re-fetched (see below) |

## Inputs that live outside this repo

`dust3d_lib.py` reads these by absolute path relative to `~/Documents/research/`:

- `data/maps/` — Edenhofer (2023) cubes (`edenhofer_lbd_wrap.fits`, `edenhofer_xyz.fits`, the HEALPix samples file), `marton_ysos.fits`
- `data/catalogs/mi_tab1.csv` — the cloud catalog; `kyso_rgeo_le500pc.fits` — YSOs
- `dust_3d/local_only/nicest_maps/zucker2020_fwhm0.229deg/nicest/` — NICEST extinction maps, needed
  only by the `contours` experiment step. Deleted 2026-10-28 to save space; regenerate
  them from the iNICEST web service with
  `python -c "import sys; sys.path.insert(0, 'code'); import dust3d_lib as L; L.inicest_main()"`
  (add `--list` to preview, `--only <cloud>` for one cloud).

## Running

```bash
pip install -r requirements.txt
python code/dust3d_pipeline.py                     # all steps, into ../local_only/runs/<today>/
python code/dust3d_pipeline.py --steps cylinder inputs rho0 if23 \
    --win-csv ../local_only/runs/2026-10-14/n10/spine_angles_and_fits_samples_n10.csv \
    --spine-cache ../local_only/runs/2026-10-14/n10/_spine_fits
```

The adopted distance windows and 40-seed spine fits are those in
`local_only/runs/2026-10-14/n10/`; the latest full cylinder / mass / rho0 run is
`local_only/runs/2026-10-26/`.
