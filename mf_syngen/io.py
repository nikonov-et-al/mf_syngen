"""Configuration, filesystem, and optional ehtim integration helpers."""

import json
import math
from pathlib import Path

from .model import DEFAULT_IF_INDICES, Component, CoreShift, PipelineConfig


def normalize_source_name(value: object) -> str:
    return str(value).strip()


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
            raise ValueError(
                f"Config field 'core_shift' is missing required field {exc.args[0]!r}."
            ) from exc
        if low_frequency_ghz <= 0.0 or reference_frequency_ghz <= 0.0:
            raise ValueError("Core-shift frequencies must be positive.")
        if math.isclose(low_frequency_ghz, reference_frequency_ghz):
            raise ValueError("Core-shift low and reference frequencies must be different.")
        if k_r <= 0.0:
            raise ValueError("Core-shift k_r must be positive.")

        matches = [component for component in components if component.name == component_name]
        if len(matches) != 1:
            raise ValueError(
                f"Core-shift component {component_name!r} must match exactly one component."
            )
        core_component = matches[0]
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


def discover_uvfits_files(uvf_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in uvf_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".uvf", ".uvfits"}
        ),
        key=lambda path: path.name.lower(),
    )


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
