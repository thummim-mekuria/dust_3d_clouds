# Why n_H = 10 cm⁻³ defines the clouds

*Adopted 2026-10-14, replacing n_H = 7 cm⁻³ (used through 2026-09-30). In code: `N_ISO` in
`choosing_distances_from_healpix.py`; every plotting script defaults to it.*

## What the threshold does

One volume-density value, n_iso, is used throughout the pipeline:

- **Distance window:** grown slice by slice from the densest slice in the dendrogram leaf, while
  the peak n_H over the catalog box + 1° stays ≥ n_iso, then padded by 10 pc.
- **Spine extent:** the fitted spine is walked outward until the density along it drops below
  n_iso.
- **Profiles and mass:** voxels below n_iso are masked out of the d⊥ profile and the
  density-masked mass.

It is the definition of where a cloud ends, in distance, along its axis and in mass.

## Why 10

**1. It sits well above the diffuse background in every cloud.** Inside each cloud's catalog box
and distance window, the median voxel density is 0.35–4.0 cm⁻³ (research notebook, 2026-09-11).
n = 10 is 2.5× the highest of these medians, so it selects structure rather than the diffuse
medium the clouds sit in. Only 3–25% of the volume is above n = 10, against 6–42% above n = 5.

This margin also matters for the posterior uncertainty. The Edenhofer et al. (2024) per-voxel
scatter is about 45% (STD/MEAN at ~200 pc). A threshold of 5 lies close to the densest
backgrounds, so posterior draws move large numbers of background voxels back and forth across
it. A threshold of 10 does not.

**2. It contains the clouds on the sky better than n = 5, for every cloud.** In projection
(max_d n_H over each cloud's window, on an 8°-padded cube):

- At n = 5, the cloud's connected region reaches the 8° edge for 12 of 14 clouds. It spreads
  into the surrounding gas rather than tracing the cloud.
- At n = 10 the share of the region lying outside the catalog box is lower than at n = 5 for
  every cloud, sharply so for Camelopardalis (0.88 → 0.29) and Perseus (0.79 → 0.45). Four
  clouds close within 8° (California, Camelopardalis, Corona Australis, Perseus) against two at
  n = 5.
- The 3D version gives the same picture. Testing the 3D-connected region within a 12° pad
  (notebook, 2026-09-11), three clouds close at n = 5 (California, Camelopardalis, Corona
  Australis) and five of the current 13 close at n = 10 (those three plus Hercules and
  Perseus).

**3. It removes diffuse mass without eroding the dense clouds.** Going from n = 5 to n = 10
roughly halves the mass of the diffuse clouds (Camelopardalis, Hercules, Polaris) and barely
changes the dense ones (Ophiuchus, Taurus) (notebook, 2026-09-03). The gas removed is the
low-density envelope, which is what a cloud definition should exclude.

**4. It connects to the A_V = 1 mag convention.** n_H = 10 cm⁻³ sustained over 71 pc gives
N_H = 2.2×10²¹ cm⁻², i.e. A_V = 1 mag for N_H/A_V = 2.2×10²¹ cm⁻² mag⁻¹ (Güver & Özel 2009).
The adopted distance windows are 35–210 pc deep (at n = 7), so a cloud defined at n = 10 is
roughly the volume-density counterpart of the A_V > 1 mag column cut used by Imara & Forbes
(2023). That makes our masses and theirs more comparable than at n = 5, where the matching
depth is 140 pc.

## What it is not

- **It is not a closing threshold.** At n = 10 the projected region still runs past the 8° pad
  for 9 of 13 clouds. The thresholds at which individual clouds close range from n = 5 to more
  than 30 cm⁻³ (Ophiuchus; Taurus closes at 18) (notebook, 2026-09-11). No single value closes
  all of them. n = 10 is the lowest round value that is clearly above the diffuse background
  everywhere and that improves containment for every cloud.
- **The fits depend on it.** Raising the floor makes the fitted r₀ and α fall together, the
  signature of truncating the profile (notebook, 2026-09-03). Quoted r₀, α and masses are
  conditional on n = 10, and the n = 5 / 10 comparison should be reported as a systematic.

## Catalog change made with it

λ-Orionis is excluded (`EXCLUDE` in `load_catalog`). It is a ring of dust around the λ Ori
H II region: its NICEST A_V = 1 mag contour and its projected n_H contours trace a shell, not
an elongated structure, so a single straight spine and a cylindrical radial profile do not
describe it. It is excluded in code rather than removed from `mi_tab1.csv`, which is shared
with other projects. The sample is now 13 clouds.
