#!/bin/bash
#SBATCH --job-name=dendro
#SBATCH --partition=256x44          # Hummingbird large-memory node: 256 GB, 2 x 22 cores
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4           # astrodendro is single-threaded; memory is the ask
#SBATCH --mem=64G                   # peak ~25 GiB: MEAN + STD padded, plus np.pad temporaries
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tmekuria@ucsc.edu

set -euo pipefail

module load miniconda3              # confirm the name with `module avail`
source activate dust3d

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
# /home has a 1 TB quota, so the 11.7 GB cube lives there fine. Scratch has no
# quota but is auto-deleted after two weeks -- fine for the run, not for inputs
# you do not want to re-upload or results you have not copied off yet.
CUBE=$HOME/data/maps/edenhofer_lbd.fits
OUT=$HOME/dust3d/out
mkdir -p "$OUT" logs

# Padding to +-30 deg past the seam happens inside dendro_full.py, for MEAN and
# STD. together; the pre-wrapped 6.3 GB file is no longer used.
python -u dendro_full.py \
    --cube "$CUBE" \
    --out  "$OUT/dendro_p99.hdf5" \
    --pad-deg 30 \
    --min-value 2.9 \
    --min-delta 1.0 \
    --min-npix 100
    # --min-snr 2   # median per-voxel S/N is only ~2.2; a cut at 3 loses ~91%

python -u run_scimes.py \
    --dendro "$OUT/dendro_p99.hdf5" \
    --out    "$OUT/scimes_p99"

sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,Partition,Elapsed,MaxRSS,State
