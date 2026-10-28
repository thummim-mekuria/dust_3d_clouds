# Dendrogram + SCIMES segmentation of the Edenhofer cube on Hummingbird

Two stages. Stage 1 is the expensive one and is checkpointed to disk so SCIMES
can be re-run with different parameters without recomputing it.

| file | role |
|---|---|
| `cube_io.py` | reads the native two-extension cube, converts to cm-3, wrap-pads MEAN and STD. |
| `dendro_full.py` | stage 1: dendrogram, seam dedup, writes `.hdf5` + `.hdr` sidecar |
| `ppp_catalog.py` | PPP replacement for astrodendro's `ppv_catalog` |
| `run_scimes.py` | stage 2: catalog + SCIMES + assignment cube |
| `dendro.sbatch` | the job |
| `install_scimes.sh` | installs and repairs scimes |

---

## One-time setup on Hummingbird

### 1. Log in and confirm partition access

```bash
ssh <cruzid>@hb.ucsc.edu
sinfo -p 256x44 -o "%P %a %D %c %m %t"      # up? how much memory does it really have?
```

`256x44` is the large-memory node: 256 GB, 2 x 22 cores. It is a **single node** --
if someone else is on it you queue behind them.

### 2. Build the environment (login node, not in a job)

```bash
module avail                       # find the real conda/python module name
module load miniconda3             # <- substitute what you actually found
conda create -n dust3d python=3.11 numpy scipy astropy h5py matplotlib
conda activate dust3d
bash install_scimes.sh
```

python 3.11 rather than 3.12 leaves an easier retreat to `numpy<2` if anything
in the scimes stack misbehaves.

### 3. Stage the data

The native cube is 11.7 GB. `/home` has a 1 TB quota; `/scratch` has no quota but
is **auto-deleted after two weeks without notice**, so inputs and results you care
about belong in `/home`.

**On Hummingbird**, create the destinations (rsync only creates the last
directory, so it fails if the parent is missing):

```bash
mkdir -p ~/data/maps ~/dust3d/slurm
```

**On your local machine** -- these are pushes, so they run where the data is.
Not in a VS Code Remote-SSH terminal, which is already on the cluster:

Single lines on purpose -- backslash continuations get mangled when pasted into
a terminal, which produces confusing "unknown option" errors:

```bash
rsync -avP ~/Documents/coding/research/data/maps/edenhofer_lbd.fits <cruzid>@hb.ucsc.edu:~/data/maps/
```

```bash
rsync -av --delete --exclude=__pycache__ ~/Documents/research/dust_3d/github_repo/code/hpc/ <cruzid>@hb.ucsc.edu:~/dust3d/slurm/
```

The `P` in `-avP` is `--partial --progress`, so an interrupted transfer resumes
instead of restarting -- worth having for 11.7 GB. Verify afterwards by comparing
`md5sum` on both ends.

Note macOS ships rsync **2.6.9** (Apple froze it in 2006 over the GPL3 switch).
It works for this, but `brew install rsync` gets you 3.x with better resume
behaviour. For repeated transfers of this size, Globus over the campus Science
DMZ is faster and restartable by design.

---

## Submitting

### Step 1: smoke test

Do this **before** submitting. An interactive allocation is a much cheaper place
to discover a wrong module name than a queued job is.

Use `128x24`, not `256x44` -- this needs no real memory and there is no reason to
queue behind the single large-memory node just to check that imports work.

```bash
salloc -p 128x24 -c 4 --mem=16G -t 1:00:00
module load miniconda3 && conda activate dust3d
cd ~/dust3d/slurm
python -c "import cube_io; h=cube_io.header_only(); print(h['NAXIS1'], h['CRPIX1'])"
python -c "from scimes import SpectralCloudstering; print('scimes ok')"
exit
```

Expected: `1834 918.0`, then `scimes ok`. If the first line fails it is the cube
path; if the second fails, re-run `bash install_scimes.sh`.

### Step 2: submit

Check the `module load` line and the `CUBE` / `OUT` paths at the top of
`dendro.sbatch` match what worked above, then:

```bash
cd ~/dust3d/slurm
mkdir -p logs
sbatch dendro.sbatch
```

### Monitoring

```bash
squeue -u $USER                     # PD = pending, R = running
tail -f logs/dendro_<jobid>.out     # astrodendro prints a progress bar
scontrol show job <jobid>           # if stuck in PD, the Reason field says why
scancel <jobid>
```

### Afterwards

```bash
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem
```

`MaxRSS` is the number that matters. The expectation is ~25 GiB against a 200 G
request; if that holds, lower `--mem` on later runs so you stop blocking the whole
node. Then copy results off:

```bash
rsync -avP <cruzid>@hb.ucsc.edu:~/dust3d/out/ ./out/
```

---

## Outputs

| file | contents |
|---|---|
| `dendro_p99.hdf5` | the dendrogram |
| `dendro_p99.hdr` | padded WCS + the parameters used + `DENDRMS` |
| `scimes_p99_dendrocat.fits` | per-structure PPP catalog |
| `scimes_p99_cluster_idx.txt` | structure indices SCIMES kept as clouds |
| `scimes_p99_asgn.fits` | assignment cube, integer cluster id per voxel |

Re-running just stage 2 costs seconds:

```bash
python run_scimes.py --dendro out/dendro_p99.hdf5 --out out/scimes_k30 --user-k 30
```

---

## Parameters worth knowing

- `--min-value 2.9` cm-3 is the p99 of the cube. p99.5 = 4.54, p99.9 = 10.79.
- `--min-snr` is **off by default on purpose**. Median per-voxel S/N above
  2.9 cm-3 is only ~2.2, so a cut at 3 discards ~91% of the signal and a cut at
  2 discards ~33%. The STD extension is still read either way so the noise level
  lands in the sidecar as `DENDRMS`, which stage 2 picks up automatically.
- Stage 1 runs the dendrogram **twice** by default: once to find wrap duplicates,
  then again with them masked. `--keep-seam-duplicates` skips the second pass.
  It matters -- the +-30 deg padding duplicates Perseus, among others.
- `--no-std` skips the STD extension and saves ~7 GiB if memory is tight.
