"""Synthetic visibility pipeline."""

from pathlib import Path

from .io import discover_uvfits_files, load_ehtim_modules, normalize_source_name, resolve_output_dir
from .model import (
    Component,
    CoreShift,
    PipelineConfig,
    build_model,
    component_position_mas,
    core_shift_offset_mas,
    evaluate_component_fluxes,
    materialize_core_subcomponents,
)


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
    observation=None,
) -> Path:
    if observation is None:
        eh, _ = load_ehtim_modules()
        obs_real = eh.obsdata.load_uvfits(str(input_path), IF=if_indices)
    else:
        obs_real = observation
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
    obs_syn = (
        mod.observe_same(obs_real, add_th_noise=True, seed=seed)
        if add_th_noise
        else mod.observe_same_nonoise(obs_real)
    )
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
    eh, _ = load_ehtim_modules()
    observations = [
        eh.obsdata.load_uvfits(str(input_path), IF=if_indices)
        for input_path in input_files
    ]
    components = materialize_core_subcomponents(
        config.components,
        config.core_shift,
        [float(observation.rf) for observation in observations],
    )
    print(f"Discovered {len(input_files)} input file(s) in {uvf_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Using IF indices: {if_indices}")
    print(f"Position frame: {'absolute' if config.absolute_positions else 'core-centered'}")
    if config.expected_source:
        print(f"Expected source: {config.expected_source}")

    outputs = [
        process_file(
            input_path=input_path,
            out_dir=out_dir,
            if_indices=if_indices,
            components=components,
            core_shift=config.core_shift,
            absolute_positions=config.absolute_positions,
            add_th_noise=add_th_noise,
            seed=seed,
            expected_source=config.expected_source,
            source_check=source_check,
            observation=observation,
        )
        for input_path, observation in zip(input_files, observations)
    ]
    print(f"Generated {len(outputs)} synthetic file(s).")
    return outputs
