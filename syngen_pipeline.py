import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MAS_TO_RAD = math.pi / (180.0 * 3600.0 * 1000.0)
OPTICALLY_THICK_INDEX = 2.5
DEFAULT_IF_INDICES = [0, 1, 2, 3]
DEFAULT_CONFIG_PATH = Path("components_multifreq.json")
DEFAULT_UVF_DIR = Path("data")
DEFAULT_OUT_DIR = Path("output") / "synthetic_vis"


@dataclass(frozen=True)
class Component:
    name: str
    x_mas: float
    y_mas: float
    sm_jy: float
    nu_m_ghz: float
    alpha_thin: float


@dataclass(frozen=True)
class CoreShift:
    component: str
    reference_x_mas: float
    reference_y_mas: float
    low_frequency_ghz: float
    reference_frequency_ghz: float
    dx_mas: float
    dy_mas: float
    k_r: float


@dataclass(frozen=True)
class PipelineConfig:
    expected_source: str | None
    absolute_positions: bool
    if_indices: list[int]
    components: list[Component]
    core_shift: CoreShift | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one synthetic uvfits file per input uvf/uvfits file using "
            "the input file's UV coverage and observing frequency."
        )
    )
    parser.add_argument(
        "--uvf-dir",
        type=Path,
        default=DEFAULT_UVF_DIR,
        help=f"Directory with input .uvf/.uvfits files. Default: {DEFAULT_UVF_DIR}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"JSON config describing components and spectral parameters. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for synthetic outputs. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reuse the requested output directory and overwrite matching output files.",
    )
    parser.add_argument(
        "--if-indices",
        type=int,
        nargs="+",
        default=None,
        help="Override IF indices. Example: --if-indices 0 1 2 3",
    )
    parser.add_argument(
        "--add-th-noise",
        dest="add_th_noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ehtim thermal/noise/calibration effects. Default: enabled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed passed to ehtim when thermal noise is enabled. Default: 1",
    )
    parser.add_argument(
        "--source-check",
        choices=["warn", "ignore", "error"],
        default="warn",
        help="How to handle expected source mismatches. Default: warn",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> PipelineConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    expected_source_raw = payload.get("expected_source")
    expected_source = normalize_source_name(expected_source_raw) if expected_source_raw else None
    absolute_positions = payload.get("absolute_positions", True)
    if not isinstance(absolute_positions, bool):
        raise ValueError("Config field 'absolute_positions' must be a boolean.")

    if_indices = payload.get("if_indices", DEFAULT_IF_INDICES)
    if not isinstance(if_indices, list) or not if_indices:
        raise ValueError("Config field 'if_indices' must be a non-empty list of integers.")

    components_payload = payload.get("components")
    if not isinstance(components_payload, list) or not components_payload:
        raise ValueError("Config field 'components' must be a non-empty list.")

    components: list[Component] = []
    for idx, item in enumerate(components_payload, start=1):
        try:
            component = Component(
                name=str(item["name"]),
                x_mas=float(item["x_mas"]),
                y_mas=float(item["y_mas"]),
                sm_jy=float(item["sm_jy"]),
                nu_m_ghz=float(item["nu_m_ghz"]),
                alpha_thin=float(item["alpha_thin"]),
            )
        except KeyError as exc:
            raise ValueError(f"Component #{idx} is missing required field {exc.args[0]!r}.") from exc

        if component.sm_jy <= 0.0:
            raise ValueError(f"Component {component.name!r} has non-positive sm_jy.")
        if component.nu_m_ghz <= 0.0:
            raise ValueError(f"Component {component.name!r} has non-positive nu_m_ghz.")

        components.append(component)

    core_shift_payload = payload.get("core_shift")
    core_shift = None
    if core_shift_payload is not None:
        if not isinstance(core_shift_payload, dict):
            raise ValueError("Config field 'core_shift' must be an object.")
        try:
            component_name = str(core_shift_payload["component"])
            low_frequency_ghz = float(core_shift_payload["low_frequency_ghz"])
            reference_frequency_ghz = float(core_shift_payload["reference_frequency_ghz"])
            dx_mas = float(core_shift_payload["dx_mas"])
            dy_mas = float(core_shift_payload["dy_mas"])
            k_r = float(core_shift_payload.get("k_r", 1.0))
        except KeyError as exc:
            raise ValueError(f"Config field 'core_shift' is missing required field {exc.args[0]!r}.") from exc

        if low_frequency_ghz <= 0.0 or reference_frequency_ghz <= 0.0:
            raise ValueError("Core-shift frequencies must be positive.")
        if math.isclose(low_frequency_ghz, reference_frequency_ghz):
            raise ValueError("Core-shift low and reference frequencies must be different.")
        if k_r <= 0.0:
            raise ValueError("Core-shift k_r must be positive.")

        core_components = [component for component in components if component.name == component_name]
        if len(core_components) != 1:
            raise ValueError(
                f"Core-shift component {component_name!r} must match exactly one component."
            )
        core_component = core_components[0]
        core_shift = CoreShift(
            component=component_name,
            reference_x_mas=core_component.x_mas,
            reference_y_mas=core_component.y_mas,
            low_frequency_ghz=low_frequency_ghz,
            reference_frequency_ghz=reference_frequency_ghz,
            dx_mas=dx_mas,
            dy_mas=dy_mas,
            k_r=k_r,
        )

    if not absolute_positions and core_shift is None:
        raise ValueError("Config field 'absolute_positions' can be false only with 'core_shift'.")

    return PipelineConfig(
        expected_source=expected_source,
        absolute_positions=absolute_positions,
        if_indices=[int(value) for value in if_indices],
        components=components,
        core_shift=core_shift,
    )


def normalize_source_name(value: object) -> str:
    return str(value).strip()


def discover_uvfits_files(uvf_dir: Path) -> list[Path]:
    candidates = [
        path
        for path in uvf_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".uvf", ".uvfits"}
    ]
    return sorted(candidates, key=lambda path: path.name.lower())


def resolve_output_dir(out_dir: Path, overwrite: bool = False) -> Path:
    if out_dir.exists() and not out_dir.is_dir():
        raise NotADirectoryError(f"Output path exists and is not a directory: {out_dir}")

    if overwrite:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    candidate = out_dir
    suffix = 2
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = out_dir.with_name(f"{out_dir.name}_{suffix}")
            suffix += 1


def tau_m_from_alpha(alpha_thin: float, alpha_thick: float = OPTICALLY_THICK_INDEX) -> float:
    radicand = 1.0 - (8.0 * alpha_thin) / (3.0 * alpha_thick)
    if radicand <= 0.0:
        raise ValueError(
            f"Invalid alpha_thin={alpha_thin}; homogeneous-sphere turnover would be undefined."
        )
    return 1.5 * (math.sqrt(radicand) - 1.0)


def homogeneous_sphere_flux_jy(
    nu_hz: float,
    sm_jy: float,
    nu_m_hz: float,
    alpha_thin: float,
    alpha_thick: float = OPTICALLY_THICK_INDEX,
) -> float:
    """Evaluate a homogeneous synchrotron sphere spectrum normalized at turnover."""
    if nu_hz <= 0.0:
        raise ValueError("Observation frequency must be positive.")
    if sm_jy <= 0.0:
        raise ValueError("Turnover flux sm_jy must be positive.")
    if nu_m_hz <= 0.0:
        raise ValueError("Turnover frequency nu_m_hz must be positive.")

    x = nu_hz / nu_m_hz
    tau_m = tau_m_from_alpha(alpha_thin, alpha_thick)
    attenuation = 1.0 - math.exp(-tau_m * (x ** (alpha_thin - alpha_thick)))
    normalization = 1.0 - math.exp(-tau_m)
    return sm_jy * (x ** alpha_thick) * attenuation / normalization


def load_ehtim_modules():
    try:
        import ehtim as eh
        from ehtim.model import Model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ehtim is not installed in the current Python environment. "
            "Install ehtim before running the synthetic uvfits pipeline."
        ) from exc

    return eh, Model


def core_shift_offset_mas(frequency_hz: float, core_shift: CoreShift | None) -> tuple[float, float]:
    """Return physical core displacement relative to the reference frequency."""
    if core_shift is None:
        return 0.0, 0.0
    if frequency_hz <= 0.0:
        raise ValueError("Observation frequency must be positive.")

    frequency_ghz = frequency_hz / 1e9
    if math.isclose(frequency_ghz, core_shift.reference_frequency_ghz):
        return 0.0, 0.0
    if math.isclose(frequency_ghz, core_shift.low_frequency_ghz):
        return core_shift.dx_mas, core_shift.dy_mas

    exponent = -1.0 / core_shift.k_r
    numerator = (
        frequency_ghz**exponent - core_shift.reference_frequency_ghz**exponent
    )
    denominator = (
        core_shift.low_frequency_ghz**exponent
        - core_shift.reference_frequency_ghz**exponent
    )
    factor = numerator / denominator
    return core_shift.dx_mas * factor, core_shift.dy_mas * factor


def component_position_mas(
    component: Component,
    frequency_hz: float,
    core_shift: CoreShift | None,
    absolute_positions: bool = True,
) -> tuple[float, float]:
    """Return component position in absolute or core-centered coordinates."""
    if core_shift is None:
        return component.x_mas, component.y_mas

    core_dx_mas, core_dy_mas = core_shift_offset_mas(frequency_hz, core_shift)
    core_x_mas = core_shift.reference_x_mas + core_dx_mas
    core_y_mas = core_shift.reference_y_mas + core_dy_mas
    if component.name == core_shift.component:
        x_mas, y_mas = core_x_mas, core_y_mas
    else:
        x_mas, y_mas = component.x_mas, component.y_mas
    if absolute_positions:
        return x_mas, y_mas
    return x_mas - core_x_mas, y_mas - core_y_mas


def build_model(
    obs_real,
    component_fluxes: Iterable[tuple[Component, float]],
    core_shift: CoreShift | None,
    absolute_positions: bool,
):
    _, Model = load_ehtim_modules()

    mod = Model(
        ra=obs_real.ra,
        dec=obs_real.dec,
        rf=obs_real.rf,
        source=obs_real.source,
        mjd=obs_real.mjd,
    )

    for component, flux_jy in component_fluxes:
        x_mas, y_mas = component_position_mas(
            component, float(obs_real.rf), core_shift, absolute_positions
        )
        mod = mod.add_point(
            F0=flux_jy,
            x0=x_mas * MAS_TO_RAD,
            y0=y_mas * MAS_TO_RAD,
        )

    return mod


def evaluate_component_fluxes(
    frequency_hz: float,
    components: Iterable[Component],
) -> list[tuple[Component, float]]:
    fluxes: list[tuple[Component, float]] = []
    for component in components:
        flux_jy = homogeneous_sphere_flux_jy(
            nu_hz=frequency_hz,
            sm_jy=component.sm_jy,
            nu_m_hz=component.nu_m_ghz * 1e9,
            alpha_thin=component.alpha_thin,
        )
        fluxes.append((component, flux_jy))
    return fluxes


def enforce_source_policy(
    input_path: Path,
    observed_source: str,
    expected_source: str | None,
    source_check: str,
) -> None:
    if not expected_source or source_check == "ignore":
        return

    if observed_source.casefold() == expected_source.casefold():
        return

    message = (
        f"Source mismatch for {input_path.name}: expected {expected_source!r}, "
        f"found {observed_source!r}. Continuing because only UV coverage is required."
    )

    if source_check == "warn":
        print(f"WARNING: {message}")
        return

    raise ValueError(message)


def process_file(
    input_path: Path,
    out_dir: Path,
    if_indices: list[int],
    components: list[Component],
    core_shift: CoreShift | None,
    absolute_positions: bool,
    add_th_noise: bool,
    seed: int,
    expected_source: str | None,
    source_check: str,
) -> Path:
    eh, _ = load_ehtim_modules()

    obs_real = eh.obsdata.load_uvfits(str(input_path), IF=if_indices)
    observed_source = normalize_source_name(obs_real.source)
    observed_freq_ghz = float(obs_real.rf) / 1e9

    print(f"Input: {input_path.name}")
    print(f"  source: {observed_source}")
    print(f"  frequency: {observed_freq_ghz:.6f} GHz")

    enforce_source_policy(
        input_path=input_path,
        observed_source=observed_source,
        expected_source=expected_source,
        source_check=source_check,
    )

    component_fluxes = evaluate_component_fluxes(float(obs_real.rf), components)
    if core_shift is not None:
        core_dx_mas, core_dy_mas = core_shift_offset_mas(float(obs_real.rf), core_shift)
        print(f"  injected physical core shift: ({core_dx_mas:.10f}, {core_dy_mas:.10f}) mas")
    for component, flux_jy in component_fluxes:
        x_mas, y_mas = component_position_mas(
            component, float(obs_real.rf), core_shift, absolute_positions
        )
        print(
            f"  component {component.name}: {flux_jy:.6f} Jy "
            f"at ({x_mas:.10f}, {y_mas:.10f}) mas"
        )

    mod = build_model(obs_real, component_fluxes, core_shift, absolute_positions)
    if add_th_noise:
        obs_syn = mod.observe_same(obs_real, add_th_noise=True, seed=seed)
    else:
        obs_syn = mod.observe_same_nonoise(obs_real)

    output_path = out_dir / f"{input_path.stem}_synthetic.uvfits"
    obs_syn.save_uvfits(str(output_path))
    print(f"  output: {output_path}")
    return output_path


def run_pipeline(
    uvf_dir: Path,
    config: PipelineConfig,
    out_dir: Path,
    if_indices_override: list[int] | None,
    add_th_noise: bool,
    seed: int,
    source_check: str,
    overwrite: bool = False,
) -> list[Path]:
    if not uvf_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {uvf_dir}")
    if not uvf_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {uvf_dir}")

    input_files = discover_uvfits_files(uvf_dir)
    if not input_files:
        raise FileNotFoundError(f"No .uvf or .uvfits files found in {uvf_dir}")

    out_dir = resolve_output_dir(out_dir, overwrite)
    if_indices = if_indices_override if if_indices_override is not None else config.if_indices

    print(f"Discovered {len(input_files)} input file(s) in {uvf_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Using IF indices: {if_indices}")
    print(f"Position frame: {'absolute' if config.absolute_positions else 'core-centered'}")
    if config.expected_source:
        print(f"Expected source: {config.expected_source}")

    outputs: list[Path] = []
    for input_path in input_files:
        output_path = process_file(
            input_path=input_path,
            out_dir=out_dir,
            if_indices=if_indices,
            components=config.components,
            core_shift=config.core_shift,
            absolute_positions=config.absolute_positions,
            add_th_noise=add_th_noise,
            seed=seed,
            expected_source=config.expected_source,
            source_check=source_check,
        )
        outputs.append(output_path)

    print(f"Generated {len(outputs)} synthetic file(s).")
    return outputs


def main() -> int:
    parsed_args = parse_args()
    config = load_config(parsed_args.config)
    run_pipeline(
        uvf_dir=parsed_args.uvf_dir,
        config=config,
        out_dir=parsed_args.out_dir,
        if_indices_override=parsed_args.if_indices,
        add_th_noise=parsed_args.add_th_noise,
        seed=parsed_args.seed,
        source_check=parsed_args.source_check,
        overwrite=parsed_args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
