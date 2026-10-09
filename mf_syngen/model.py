"""Model configuration and frequency-dependent source calculations."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

MAS_TO_RAD = math.pi / (180.0 * 3600.0 * 1000.0)
OPTICALLY_THICK_INDEX = 2.5
DEFAULT_IF_INDICES = [0, 1, 2, 3]


@dataclass(frozen=True)
class Component:
    name: str
    x_mas: float
    y_mas: float
    sm_jy: float
    nu_m_ghz: float
    alpha_thin: float
    # Set only for generated fixed core subcomponents.
    core_frequency_ghz: float | None = None


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
    subcomponents: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    expected_source: str | None
    absolute_positions: bool
    if_indices: list[int]
    components: list[Component]
    core_shift: CoreShift | None


def load_config(config_path: Path) -> PipelineConfig:
    """Compatibility import; configuration I/O lives in :mod:`mf_syngen.io`."""
    from .io import load_config as load

    return load(config_path)


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
    numerator = frequency_ghz**exponent - core_shift.reference_frequency_ghz**exponent
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


def evaluate_component_fluxes(
    frequency_hz: float,
    components: Iterable[Component],
) -> list[tuple[Component, float]]:
    return [
        (
            component,
            homogeneous_sphere_flux_jy(
                nu_hz=frequency_hz,
                sm_jy=component.sm_jy,
                nu_m_hz=component.nu_m_ghz * 1e9,
                alpha_thin=component.alpha_thin,
            ),
        )
        for component in components
    ]


def materialize_core_subcomponents(
    components: Iterable[Component],
    core_shift: CoreShift | None,
    frequencies_hz: Iterable[float],
) -> list[Component]:
    """Expand one logical core into fixed, one-per-band core components."""
    source = list(components)
    if core_shift is None or not core_shift.subcomponents:
        return source
    core_matches = [c for c in source if c.name == core_shift.component]
    if len(core_matches) != 1:
        raise ValueError(f"Core-shift component {core_shift.component!r} must match exactly one component.")
    core = core_matches[0]
    freqs = sorted({float(freq) for freq in frequencies_hz})
    if not freqs or any(not math.isfinite(freq) or freq <= 0.0 for freq in freqs):
        raise ValueError("Subcomponent frequencies must be positive.")
    targets = [homogeneous_sphere_flux_jy(freq, core.sm_jy, core.nu_m_ghz * 1e9, core.alpha_thin)
               for freq in freqs]
    templates = []
    for turnover_hz in freqs:
        templates.append([
            homogeneous_sphere_flux_jy(freq, 1.0, turnover_hz, core.alpha_thin)
            for freq in freqs
        ])
    sums = [sum(row[idx] for row in templates) for idx in range(len(freqs))]
    ratios = [total / target for total, target in zip(sums, targets)]
    scale = sum(ratios) / sum(value * value for value in ratios)
    generated = [
        Component(
            name=f"{core.name}_{freq / 1e9:.9f}GHz",
            x_mas=core_shift.reference_x_mas + core_shift_offset_mas(freq, core_shift)[0],
            y_mas=core_shift.reference_y_mas + core_shift_offset_mas(freq, core_shift)[1],
            sm_jy=scale,
            nu_m_ghz=freq / 1e9,
            alpha_thin=core.alpha_thin,
            core_frequency_ghz=freq / 1e9,
        )
        for freq in freqs
    ]
    return [component for component in source if component.name != core_shift.component] + generated


def build_model(
    obs_real,
    component_fluxes: Iterable[tuple[Component, float]],
    core_shift: CoreShift | None,
    absolute_positions: bool,
    coordinate_offsets_rad: Mapping[str, tuple[float, float]] | None = None,
):
    from .io import load_ehtim_modules

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
        x_offset_rad, y_offset_rad = (0.0, 0.0)
        if coordinate_offsets_rad is not None:
            x_offset_rad, y_offset_rad = coordinate_offsets_rad.get(component.name, (0.0, 0.0))
        mod = mod.add_point(
            F0=flux_jy,
            x0=x_mas * MAS_TO_RAD + x_offset_rad,
            y0=y_mas * MAS_TO_RAD + y_offset_rad,
        )
    return mod
