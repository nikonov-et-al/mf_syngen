"""Ground-truth image and spectral-index product generation."""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .io import (
    discover_uvfits_files,
    load_config,
    load_ehtim_modules,
    normalize_source_name,
    resolve_output_dir,
)
from .model import (
    MAS_TO_RAD,
    PipelineConfig,
    build_model,
    component_position_mas,
    evaluate_component_fluxes,
    materialize_core_subcomponents,
)


DEFAULT_CONFIG_PATH = Path("components_multifreq.json")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUT_DIR = Path("output") / "ground_truth_images"
RAD_TO_MAS = 1.0 / MAS_TO_RAD
POINT_BOUNDARY_TOLERANCE_PIXELS = 1e-10
POINT_BOUNDARY_NUDGE_PIXELS = 1e-12


@dataclass(frozen=True)
class Beam:
    """Elliptical Gaussian beam stored in ehtim's native radian units."""

    major_rad: float
    minor_rad: float
    pa_rad: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.major_rad, self.minor_rad, self.pa_rad)):
            raise ValueError("Beam values must be finite.")
        if self.major_rad <= 0.0 or self.minor_rad <= 0.0:
            raise ValueError("Beam axes must be positive.")
        if self.major_rad < self.minor_rad:
            raise ValueError("Beam major axis must be greater than or equal to its minor axis.")
        object.__setattr__(self, "pa_rad", self.pa_rad % math.pi)

    @classmethod
    def from_mas(cls, major_mas: float, minor_mas: float, pa_deg: float) -> "Beam":
        return cls(major_mas * MAS_TO_RAD, minor_mas * MAS_TO_RAD, math.radians(pa_deg))

    @classmethod
    def from_ehtim(cls, values: Sequence[float]) -> "Beam":
        if len(values) < 3:
            raise ValueError("ehtim beam must contain major axis, minor axis, and PA.")
        return cls(float(values[0]), float(values[1]), float(values[2]))

    @property
    def major_mas(self) -> float:
        return self.major_rad * RAD_TO_MAS

    @property
    def minor_mas(self) -> float:
        return self.minor_rad * RAD_TO_MAS

    @property
    def pa_deg(self) -> float:
        return math.degrees(self.pa_rad)

    @property
    def area(self) -> float:
        return self.major_rad * self.minor_rad

    def as_ehtim(self) -> tuple[float, float, float]:
        return self.major_rad, self.minor_rad, self.pa_rad


@dataclass
class _FrequencyImage:
    path: Path
    frequency_hz: float
    observation: Any
    beam: Beam
    intrinsic: Any | None = None


def select_beam(beams: Sequence[Beam], mode: str) -> Beam:
    """Select or average beams using area and axial PA statistics."""
    if not beams:
        raise ValueError("At least one beam is required.")
    mode = {"b": "biggest", "s": "smallest", "a": "average"}.get(mode, mode)
    if mode == "biggest":
        return max(beams, key=lambda beam: beam.area)
    if mode == "smallest":
        return min(beams, key=lambda beam: beam.area)
    if mode != "average":
        raise ValueError(f"Unknown beam selection mode: {mode!r}")

    count = len(beams)
    pa = 0.5 * math.atan2(
        sum(math.sin(2.0 * beam.pa_rad) for beam in beams) / count,
        sum(math.cos(2.0 * beam.pa_rad) for beam in beams) / count,
    )
    return Beam(
        sum(beam.major_rad for beam in beams) / count,
        sum(beam.minor_rad for beam in beams) / count,
        pa,
    )


def derive_common_grid(
    positions_mas: Iterable[tuple[float, float]],
    beams: Sequence[Beam],
    fov_mas: float | None = None,
    npix: int | None = None,
) -> tuple[float, int]:
    """Return a common square FOV in radians and pixel count."""
    positions = list(positions_mas)
    if not positions:
        raise ValueError("At least one component position is required.")
    if not beams:
        raise ValueError("At least one beam is required.")

    max_axis_mas = max(max(abs(x), abs(y)) for x, y in positions)
    radius_mas = max(math.hypot(x, y) for x, y in positions)
    padding_mas = 2 * radius_mas or max(3 * beam.major_mas for beam in beams)
    automatic_fov_mas = 2.0 * (max_axis_mas + padding_mas)

    selected_fov_mas = automatic_fov_mas if fov_mas is None else float(fov_mas)
    if not math.isfinite(selected_fov_mas) or selected_fov_mas <= 0.0:
        raise ValueError("Field of view must be positive and finite.")
    if selected_fov_mas / 2.0 <= max_axis_mas:
        raise ValueError("Requested field of view would clip a model component.")

    if npix is None:
        pixel_mas = min(beam.minor_mas for beam in beams) / 5.0
        required = max(2, math.ceil(selected_fov_mas / pixel_mas))
        selected_npix = 1 << (required - 1).bit_length()
    else:
        selected_npix = int(npix)
        if selected_npix <= 0:
            raise ValueError("Pixel count must be positive.")

    return selected_fov_mas * MAS_TO_RAD, selected_npix


def calculate_spectral_index(
    low_image: Any,
    high_image: Any,
    low_frequency_hz: float,
    high_frequency_hz: float,
    mask_fraction: float = 1e-6,
):
    """Calculate alpha, storing invalid and near-zero pixels as NaN."""
    np = _load_numpy()
    low = np.asarray(low_image, dtype=float)
    high = np.asarray(high_image, dtype=float)
    if low.shape != high.shape:
        raise ValueError("Spectral-index images must have identical shapes.")
    if low_frequency_hz <= 0.0 or high_frequency_hz <= low_frequency_hz:
        raise ValueError("Spectral-index frequencies must be positive and increasing.")
    if not 0.0 <= mask_fraction < 1.0:
        raise ValueError("Mask fraction must be in [0, 1).")

    low_positive = low[np.isfinite(low) & (low > 0.0)]
    high_positive = high[np.isfinite(high) & (high > 0.0)]
    if not low_positive.size or not high_positive.size:
        raise ValueError("Both images must contain valid positive pixels.")
    low_floor = float(low_positive.max()) * mask_fraction
    high_floor = float(high_positive.max()) * mask_fraction
    valid = (
        np.isfinite(low)
        & np.isfinite(high)
        & (low > low_floor)
        & (high > high_floor)
    )
    if not valid.any():
        raise ValueError("No valid pixels remain after spectral-index masking.")

    alpha = np.full(low.shape, np.nan, dtype=float)
    alpha[valid] = np.log(high[valid] / low[valid]) / math.log(
        high_frequency_hz / low_frequency_hz
    )
    return alpha


def unique_frequency_pairs(frequencies_hz: Iterable[float]) -> list[tuple[float, float]]:
    """Validate unique frequencies and return sorted low/high pairs."""
    frequencies = sorted(float(value) for value in frequencies_hz)
    if any(value <= 0.0 or not math.isfinite(value) for value in frequencies):
        raise ValueError("Observation frequencies must be positive and finite.")
    if len(set(frequencies)) != len(frequencies):
        raise ValueError("Repeated observing frequency is ambiguous.")
    return list(itertools.combinations(frequencies, 2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ground-truth FITS images and pairwise spectral-index FITS/PDF maps.",
        add_help=False,
    )
    parser.add_argument("-h", action="help", help="Show this help message and exit.")
    parser.add_argument("-c", dest="config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("-d", dest="data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("-o", dest="out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("-i", dest="if_indices", type=int, nargs="+", default=None)
    parser.add_argument("-w", dest="weighting", choices=["n", "u", "c"], default="n")
    parser.add_argument("-b", dest="image_beam", type=float, nargs=3, metavar=("MAJ", "MIN", "PA"))
    parser.add_argument("-m", dest="map_beam", choices=["b", "s", "a", "c"], default="b")
    parser.add_argument("-g", dest="beam_scope", choices=["p", "a"], default="p")
    parser.add_argument("-B", dest="spectral_beam", type=float, nargs=3, metavar=("MAJ", "MIN", "PA"))
    parser.add_argument("-z", dest="mask_fraction", type=float, default=1e-6)
    parser.add_argument("-f", dest="fov_mas", type=float, default=None)
    parser.add_argument("-n", dest="npix", type=int, default=None)
    args = parser.parse_args(argv)

    if args.weighting == "c" and args.image_beam is None:
        parser.error("-w c requires -b MAJ MIN PA")
    if args.weighting != "c" and args.image_beam is not None:
        parser.error("-b is valid only with -w c")
    if args.map_beam == "c" and args.spectral_beam is None:
        parser.error("-m c requires -B MAJ MIN PA")
    if args.map_beam != "c" and args.spectral_beam is not None:
        parser.error("-B is valid only with -m c")
    if not 0.0 <= args.mask_fraction < 1.0:
        parser.error("-z must be in [0, 1)")
    if args.fov_mas is not None and args.fov_mas <= 0.0:
        parser.error("-f must be positive")
    if args.npix is not None and args.npix <= 0:
        parser.error("-n must be positive")
    try:
        if args.image_beam is not None:
            Beam.from_mas(*args.image_beam)
        if args.spectral_beam is not None:
            Beam.from_mas(*args.spectral_beam)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def run_ground_truth(
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    if_indices_override: list[int] | None = None,
    weighting: str = "n",
    image_beam: Beam | None = None,
    map_beam: str = "b",
    beam_scope: str = "p",
    spectral_beam: Beam | None = None,
    mask_fraction: float = 1e-6,
    fov_mas: float | None = None,
    npix: int | None = None,
) -> list[Path]:
    """Generate all requested ground-truth products and return their paths."""
    config = load_config(config_path)
    _validate_run_options(weighting, image_beam, map_beam, beam_scope, spectral_beam, mask_fraction)
    if not data_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {data_dir}")
    input_paths = discover_uvfits_files(data_dir)
    if not input_paths:
        raise FileNotFoundError(f"No .uvf or .uvfits files found in {data_dir}")

    eh, _ = load_ehtim_modules()
    if_indices = if_indices_override if if_indices_override is not None else config.if_indices
    records: list[_FrequencyImage] = []
    for input_path in input_paths:
        observation = eh.obsdata.load_uvfits(str(input_path), IF=if_indices)
        frequency_hz = float(observation.rf)
        beam = image_beam or Beam.from_ehtim(
            observation.fit_beam(
                weighting="natural" if weighting == "n" else "uniform",
                units="rad",
            )
        )
        _warn_source_mismatch(input_path, observation, config)
        records.append(_FrequencyImage(input_path, frequency_hz, observation, beam))

    unique_frequency_pairs(record.frequency_hz for record in records)
    records.sort(key=lambda record: record.frequency_hz)
    components = materialize_core_subcomponents(
        config.components,
        config.core_shift,
        (record.frequency_hz for record in records),
    )
    positions = _all_component_positions(
        config, (record.frequency_hz for record in records), components
    )
    grid_beams = [record.beam for record in records]
    if spectral_beam is not None:
        grid_beams.append(spectral_beam)
    fov_rad, selected_npix = derive_common_grid(positions, grid_beams, fov_mas, npix)

    selected_out_dir = resolve_output_dir(out_dir)
    outputs: list[Path] = []
    for record in records:
        fluxes = evaluate_component_fluxes(record.frequency_hz, components)
        coordinate_offsets = _point_pixel_boundary_offsets(
            config,
            record.frequency_hz,
            fov_rad / selected_npix,
            components,
        )
        model = build_model(
            record.observation,
            fluxes,
            config.core_shift,
            config.absolute_positions,
            coordinate_offsets,
        )
        record.intrinsic = model.make_image(fov_rad, selected_npix)
        record.intrinsic.imarr()
        blurred = record.intrinsic.blur_gauss(record.beam.as_ehtim())
        path = selected_out_dir / f"ground_truth_{_frequency_label(record.frequency_hz)}GHz.fits"
        blurred.save_fits(str(path))
        _annotate_fits(
            path,
            record.beam,
            frequency_hz=record.frequency_hz,
            beam_mode={"n": "natural", "u": "uniform", "c": "custom"}[weighting],
        )
        outputs.append(path)

    global_beam = None
    if map_beam != "c" and beam_scope == "a":
        global_beam = select_beam([record.beam for record in records], map_beam)

    for low, high in itertools.combinations(records, 2):
        target_beam = spectral_beam or global_beam or select_beam([low.beam, high.beam], map_beam)
        low_common = low.intrinsic.blur_gauss(target_beam.as_ehtim())
        high_common = high.intrinsic.blur_gauss(target_beam.as_ehtim())
        alpha = calculate_spectral_index(
            low_common.imarr(),
            high_common.imarr(),
            low.frequency_hz,
            high.frequency_hz,
            mask_fraction,
        )
        stem = (
            f"spectral_index_{_frequency_label(low.frequency_hz)}-"
            f"{_frequency_label(high.frequency_hz)}GHz"
        )
        fits_path = selected_out_dir / f"{stem}.fits"
        pdf_path = selected_out_dir / f"{stem}.pdf"
        spectral_image = low_common.copy()
        spectral_image.imvec = alpha.reshape(-1)
        if hasattr(spectral_image, "_imdict"):
            empty = _load_numpy().array([])
            for polarization in spectral_image._imdict:
                if polarization != spectral_image.pol_prim:
                    spectral_image._imdict[polarization] = empty.copy()
        spectral_image.save_fits(str(fits_path))
        _annotate_fits(
            fits_path,
            target_beam,
            low_frequency_hz=low.frequency_hz,
            high_frequency_hz=high.frequency_hz,
            beam_mode={"b": "biggest", "s": "smallest", "a": "average", "c": "custom"}[map_beam],
            beam_scope={"p": "pair", "a": "all"}[beam_scope],
            mask_fraction=mask_fraction,
            spectral=True,
        )
        _plot_spectral_index(
            alpha,
            pdf_path,
            fov_rad,
            low.frequency_hz,
            high.frequency_hz,
            target_beam,
        )
        outputs.extend((fits_path, pdf_path))

    print(f"Output directory: {selected_out_dir}")
    print(f"Generated {len(outputs)} ground-truth product(s).")
    return outputs


def _validate_run_options(
    weighting: str,
    image_beam: Beam | None,
    map_beam: str,
    beam_scope: str,
    spectral_beam: Beam | None,
    mask_fraction: float,
) -> None:
    if weighting not in {"n", "u", "c"}:
        raise ValueError("Weighting must be n, u, or c.")
    if (weighting == "c") != (image_beam is not None):
        raise ValueError("Custom weighting requires exactly one custom image beam.")
    if map_beam not in {"b", "s", "a", "c"}:
        raise ValueError("Map beam must be b, s, a, or c.")
    if (map_beam == "c") != (spectral_beam is not None):
        raise ValueError("Custom map mode requires exactly one custom spectral beam.")
    if beam_scope not in {"p", "a"}:
        raise ValueError("Beam scope must be p or a.")
    if not 0.0 <= mask_fraction < 1.0:
        raise ValueError("Mask fraction must be in [0, 1).")


def _all_component_positions(
    config: PipelineConfig,
    frequencies_hz: Iterable[float],
    components=None,
) -> list[tuple[float, float]]:
    components = config.components if components is None else components
    return [
        component_position_mas(
            component,
            frequency_hz,
            config.core_shift,
            config.absolute_positions,
        )
        for frequency_hz in frequencies_hz
        for component in components
    ]


def _point_pixel_boundary_offsets(
    config: PipelineConfig,
    frequency_hz: float,
    pixel_size_rad: float,
    components=None,
) -> dict[str, tuple[float, float]]:
    """Nudge point sources off strict ehtim pixel-cell boundaries."""
    if pixel_size_rad <= 0.0 or not math.isfinite(pixel_size_rad):
        raise ValueError("Pixel size must be positive and finite.")

    nudge_rad = pixel_size_rad * POINT_BOUNDARY_NUDGE_PIXELS
    offsets: dict[str, tuple[float, float]] = {}
    components = config.components if components is None else components
    for component in components:
        x_mas, y_mas = component_position_mas(
            component,
            frequency_hz,
            config.core_shift,
            config.absolute_positions,
        )
        x_rad = x_mas * MAS_TO_RAD
        y_rad = y_mas * MAS_TO_RAD
        x_offset_rad = _point_pixel_boundary_offset(x_rad, pixel_size_rad, nudge_rad)
        y_offset_rad = _point_pixel_boundary_offset(y_rad, pixel_size_rad, nudge_rad)
        if x_offset_rad == 0.0 and y_offset_rad == 0.0:
            continue

        offsets[component.name] = (x_offset_rad, y_offset_rad)
        print(
            "WARNING: Adjusted point-source coordinate off an ehtim pixel boundary; "
            f"component={component.name!r}, frequency={frequency_hz / 1e9:g} GHz. "
            f"Added (x, y)=({x_offset_rad:.3e}, {y_offset_rad:.3e}) rad "
            f"= ({x_offset_rad * RAD_TO_MAS:.3e}, {y_offset_rad * RAD_TO_MAS:.3e}) mas; "
            f"using (x, y)=({(x_rad + x_offset_rad) * RAD_TO_MAS:.12g}, "
            f"{(y_rad + y_offset_rad) * RAD_TO_MAS:.12g}) mas. "
            "Reason: ehtim point sampling uses strict pixel-boundary comparisons."
        )
    return offsets


def _point_pixel_boundary_offset(
    coordinate_rad: float,
    pixel_size_rad: float,
    nudge_rad: float,
) -> float:
    pixel_coordinate = coordinate_rad / pixel_size_rad
    if math.isclose(
        pixel_coordinate,
        round(pixel_coordinate),
        abs_tol=POINT_BOUNDARY_TOLERANCE_PIXELS,
    ):
        return nudge_rad
    return 0.0


def _warn_source_mismatch(input_path: Path, observation: Any, config: PipelineConfig) -> None:
    if not config.expected_source:
        return
    observed = normalize_source_name(observation.source)
    if observed.casefold() != config.expected_source.casefold():
        print(
            f"WARNING: Source mismatch for {input_path.name}: "
            f"expected {config.expected_source!r}, found {observed!r}."
        )


def _frequency_label(frequency_hz: float) -> str:
    return f"{frequency_hz / 1e9:.9f}".rstrip("0").rstrip(".")


def _annotate_fits(
    path: Path,
    beam: Beam,
    *,
    frequency_hz: float | None = None,
    low_frequency_hz: float | None = None,
    high_frequency_hz: float | None = None,
    beam_mode: str,
    beam_scope: str | None = None,
    mask_fraction: float | None = None,
    spectral: bool = False,
) -> None:
    fits = _load_fits()
    with fits.open(path, mode="update") as hdus:
        if spectral:
            while len(hdus) > 1:
                del hdus[1]
        header = hdus[0].header
        header["BMAJ"] = (math.degrees(beam.major_rad), "Beam major FWHM (deg)")
        header["BMIN"] = (math.degrees(beam.minor_rad), "Beam minor FWHM (deg)")
        header["BPA"] = (beam.pa_deg, "Beam position angle (deg)")
        header["BEAMMODE"] = (beam_mode, "Beam selection or weighting")
        if frequency_hz is not None:
            header["FREQ"] = (frequency_hz, "Observation frequency (Hz)")
        if low_frequency_hz is not None:
            header["FREQLO"] = (low_frequency_hz, "Low frequency (Hz)")
        if high_frequency_hz is not None:
            header["FREQHI"] = (high_frequency_hz, "High frequency (Hz)")
        if beam_scope is not None:
            header["BEAMSCP"] = (beam_scope, "Beam selection scope")
        if mask_fraction is not None:
            header["MASKREL"] = (mask_fraction, "Relative masking threshold")
        if spectral:
            header["BUNIT"] = ("1", "Dimensionless spectral index")
            header["STOKES"] = ("ALPHA", "Spectral-index map")
            header.pop("POLREP", None)
        hdus.flush()


def _plot_spectral_index(
    alpha: Any,
    output_path: Path,
    fov_rad: float,
    low_frequency_hz: float,
    high_frequency_hz: float,
    beam: Beam,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to write spectral-index PDFs.") from exc

    fov_mas = fov_rad * RAD_TO_MAS
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)
    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    rendered = axis.imshow(
        alpha,
        origin="upper",
        extent=(fov_mas / 2.0, -fov_mas / 2.0, -fov_mas / 2.0, fov_mas / 2.0),
        cmap=cmap,
        interpolation="nearest",
    )
    axis.set_xlabel("Relative RA (mas)")
    axis.set_ylabel("Relative Dec (mas)")
    axis.set_title(
        f"{low_frequency_hz / 1e9:g}-{high_frequency_hz / 1e9:g} GHz; "
        f"beam {beam.major_mas:.3g}x{beam.minor_mas:.3g} mas, PA {beam.pa_deg:.3g} deg"
    )
    fig.colorbar(rendered, ax=axis, label="Spectral index α")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def _load_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required to calculate spectral-index maps.") from exc
    return np


def _load_fits():
    try:
        from astropy.io import fits
    except ModuleNotFoundError as exc:
        raise RuntimeError("astropy is required to annotate FITS products.") from exc
    return fits


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_ground_truth(
        config_path=args.config,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        if_indices_override=args.if_indices,
        weighting=args.weighting,
        image_beam=Beam.from_mas(*args.image_beam) if args.image_beam else None,
        map_beam=args.map_beam,
        beam_scope=args.beam_scope,
        spectral_beam=Beam.from_mas(*args.spectral_beam) if args.spectral_beam else None,
        mask_fraction=args.mask_fraction,
        fov_mas=args.fov_mas,
        npix=args.npix,
    )
    return 0
