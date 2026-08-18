# Multi-frequency synthetic data generator

This repository builds synthetic UVFITS observations and matching ground-truth
images from the component model in `components_multifreq.json`.

## Install and run

Create a virtual environment, then install the direct scientific dependencies:

```sh
python -m pip install -r requirements.txt
```

Put `.uvf` or `.uvfits` files in `data/`. Generate synthetic visibility data
with:

```sh
python syngen_pipeline.py
```

Generate ground-truth FITS images and spectral-index FITS/PDF maps with:

```sh
python ground_truth.py
```

Run the dependency-free unit tests with:

```sh
python -m unittest -v
```

## Model configuration

The JSON file contains an optional expected source name, IF indices, and one or
more point components. Each component supplies a name, position in milliarcseconds,
turnover flux in Jy (`sm_jy`), turnover frequency in GHz (`nu_m_ghz`), and
optically thin spectral index (`alpha_thin`). An optional `core_shift` section
makes the named component position frequency-dependent. Set
`absolute_positions` to `false` to express every component relative to the
shifted core; it defaults to `true`.

Observation frequencies always come from `obs.rf` in the input UVFITS files.
The configured spectrum and core shift are evaluated at those frequencies.

## Ground-truth options

`ground_truth.py` deliberately uses short option names:

| Option | Meaning | Default |
|---|---|---|
| `-c PATH` | JSON model configuration | `components_multifreq.json` |
| `-d DIR` | Directory containing `.uvf`/`.uvfits` inputs | `data` |
| `-o DIR` | Requested output directory | `output/ground_truth_images` |
| `-i N...` | IF indices, overriding the JSON list | JSON `if_indices` |
| `-w n\|u\|c` | Per-frequency beam weighting: natural, uniform, or custom | `n` |
| `-b MAJ MIN PA` | Custom per-frequency beam: major/minor FWHM in mas and PA in degrees | required for `-w c` |
| `-m b\|s\|a\|c` | Spectral beam: biggest, smallest, average, or custom | `b` |
| `-g p\|a` | Select the spectral beam per pair or across all frequencies | `p` |
| `-B MAJ MIN PA` | Custom spectral beam: major/minor FWHM in mas and PA in degrees | required for `-m c` |
| `-z FLOAT` | Relative near-zero mask threshold | `1e-6` |
| `-f MAS` | Manual square field of view in mas | automatic |
| `-n NPIX` | Manual square pixel count | automatic |

Custom beam axes must be positive and `MAJ >= MIN`. The program converts mas
and degrees to radians for ehtim. A repeated observing frequency is rejected
because it would make frequency output names and pairs ambiguous.

Examples:

```sh
# Natural image beams; biggest beam chosen independently for every pair
python ground_truth.py

# Uniform image beams; one average beam selected across every frequency
python ground_truth.py -w u -m a -g a

# Independent custom image and spectral beams (major/minor mas, PA degrees)
python ground_truth.py -w c -b 0.8 0.4 25 -m c -B 1.0 0.5 10
```

### Grid, beams, and masking

All frequencies use one square grid. Automatic sizing evaluates every
frequency-dependent component position, takes the greatest radial distance
`rmax`, and pads each side beyond the largest absolute coordinate by `rmax`.
For an entirely centred model the largest beam major axis supplies nonzero
padding. The smallest beam minor axis is sampled by five pixels and the pixel
count is rounded up to a power of two. `-f` and `-n` override the automatic
field of view and pixel count; a field of view that clips a component is
rejected.

The per-frequency image uses the beam fitted from its UVFITS coverage, with
natural weighting by default, or the requested uniform/custom beam. Spectral
maps are always re-convolved from intrinsic images, so an image is never
blurred twice. Biggest and smallest beams are ordered by elliptical area
`major * minor`. An average beam uses arithmetic means for its axes and a
180-degree-periodic circular mean for position angle. Pair scope selects from
the two frequencies; all-frequency scope selects once and reuses that beam.

For each low/high-frequency pair, spectral index is

```text
alpha = log(I_high / I_low) / log(nu_high / nu_low)
```

Pixels are stored as NaN if either input is nonfinite, nonpositive, or no
brighter than `-z` times that image's peak. PDFs render those pixels
transparent and include relative RA/Dec axes and a `Spectral index α`
colorbar.

### Outputs

The requested directory is created without overwriting an existing run. If it
already exists, the lowest free suffix is used, for example
`ground_truth_images_2`.

- `ground_truth_<GHz>GHz.fits` for every unique input frequency
- `spectral_index_<lowGHz>-<highGHz>GHz.fits` for every sorted frequency pair
- `spectral_index_<lowGHz>-<highGHz>GHz.pdf` for every sorted frequency pair

The FITS headers retain image coordinates and record frequency/frequency pair,
beam (`BMAJ`, `BMIN`, `BPA`), weighting or selection mode, mask threshold, and
dimensionless spectral-index units where applicable.

## Synthetic visibility options

The original pipeline also accepts concise flags while retaining its long
options for compatibility:

```text
-d, --uvf-dir DIR
-c, --config PATH
-o, --out-dir DIR
-x, --overwrite
-i, --if-indices N [N ...]
--add-th-noise / --no-add-th-noise
-s, --seed N
-S, --source-check warn|ignore|error
```

Run `python syngen_pipeline.py --help` or `python ground_truth.py -h` for the
authoritative command-line help.
