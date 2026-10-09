import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from syngen_pipeline import (
    component_position_mas,
    core_shift_offset_mas,
    evaluate_component_fluxes,
    homogeneous_sphere_flux_jy,
    load_config,
    materialize_core_subcomponents,
    OPTICALLY_THICK_INDEX,
    parse_args,
    resolve_output_dir,
)


CONFIG = {
    "if_indices": [0, 1, 2, 3],
    "components": [
        {
            "name": "core",
            "x_mas": 0.2,
            "y_mas": 0.1,
            "sm_jy": 1.5,
            "nu_m_ghz": 40.0,
            "alpha_thin": -0.5,
        },
        {
            "name": "jet1",
            "x_mas": 0.8,
            "y_mas": -0.4,
            "sm_jy": 0.44,
            "nu_m_ghz": 15.0,
            "alpha_thin": -1.4,
        },
    ],
    "core_shift": {
        "component": "core",
        "low_frequency_ghz": 22.1279375,
        "reference_frequency_ghz": 42.8279375,
        "dx_mas": 0.0894427191,
        "dy_mas": -0.0447213595,
        "k_r": 1.0,
    },
}


class CoreShiftTests(unittest.TestCase):
    def load(self, payload=CONFIG):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_config(path)

    def test_reference_frequency_has_zero_shift(self):
        config = self.load()
        self.assertEqual(
            core_shift_offset_mas(42.8279375e9, config.core_shift),
            (0.0, 0.0),
        )

    def test_low_frequency_has_configured_shift(self):
        config = self.load()
        self.assertEqual(
            core_shift_offset_mas(22.1279375e9, config.core_shift),
            (0.0894427191, -0.0447213595),
        )

    def test_absolute_component_positions_are_default(self):
        config = self.load()
        core, jet = config.components

        self.assertTrue(config.absolute_positions)
        self.assertAlmostEqual(
            component_position_mas(core, 22.1279375e9, config.core_shift)[0],
            0.2894427191,
        )
        self.assertAlmostEqual(
            component_position_mas(core, 22.1279375e9, config.core_shift)[1],
            0.0552786405,
        )
        self.assertEqual(component_position_mas(jet, 22.1279375e9, config.core_shift), (0.8, -0.4))
        self.assertEqual(
            component_position_mas(core, 42.8279375e9, config.core_shift), (0.2, 0.1)
        )

    def test_core_centered_component_positions(self):
        payload = json.loads(json.dumps(CONFIG))
        payload["absolute_positions"] = False
        config = self.load(payload)
        core, jet = config.components

        self.assertEqual(
            component_position_mas(core, 22.1279375e9, config.core_shift, config.absolute_positions),
            (0.0, 0.0),
        )
        self.assertAlmostEqual(
            component_position_mas(jet, 22.1279375e9, config.core_shift, config.absolute_positions)[0],
            0.5105572809,
        )
        self.assertAlmostEqual(
            component_position_mas(jet, 22.1279375e9, config.core_shift, config.absolute_positions)[1],
            -0.4552786405,
        )
        x_mas, y_mas = component_position_mas(
            jet, 42.8279375e9, config.core_shift, config.absolute_positions
        )
        self.assertAlmostEqual(x_mas, 0.6)
        self.assertAlmostEqual(y_mas, -0.5)

    def test_config_without_core_shift_keeps_absolute_positions(self):
        payload = dict(CONFIG)
        payload.pop("core_shift")
        config = self.load(payload)
        self.assertIsNone(config.core_shift)
        self.assertEqual(component_position_mas(config.components[1], 22.1279375e9, None), (0.8, -0.4))

    def test_invalid_absolute_positions_is_rejected(self):
        payload = json.loads(json.dumps(CONFIG))
        payload["absolute_positions"] = "false"
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            self.load(payload)

        payload = json.loads(json.dumps(CONFIG))
        payload.pop("core_shift")
        payload["absolute_positions"] = False
        with self.assertRaisesRegex(ValueError, "only with 'core_shift'"):
            self.load(payload)

    def test_invalid_core_shift_is_rejected(self):
        payload = json.loads(json.dumps(CONFIG))
        payload["core_shift"]["component"] = "missing"
        with self.assertRaisesRegex(ValueError, "must match exactly one component"):
            self.load(payload)

        payload = json.loads(json.dumps(CONFIG))
        payload["core_shift"]["k_r"] = 0.0
        with self.assertRaisesRegex(ValueError, "k_r must be positive"):
            self.load(payload)

        payload = json.loads(json.dumps(CONFIG))
        payload["core_shift"]["low_frequency_ghz"] = 42.8279375
        with self.assertRaisesRegex(ValueError, "must be different"):
            self.load(payload)

    def test_subcomponents_default_false_and_must_be_boolean(self):
        self.assertFalse(self.load().core_shift.subcomponents)
        payload = json.loads(json.dumps(CONFIG))
        payload["core_shift"]["subcomponents"] = "true"
        with self.assertRaisesRegex(ValueError, "subcomponents.*boolean"):
            self.load(payload)

    def test_subcomponents_are_fixed_one_per_frequency(self):
        payload = json.loads(json.dumps(CONFIG))
        payload["core_shift"]["subcomponents"] = True
        config = self.load(payload)
        frequencies = [22.1279375e9, 42.8279375e9]
        components = materialize_core_subcomponents(config.components, config.core_shift, frequencies)
        cores = [component for component in components if component.core_frequency_ghz is not None]
        self.assertEqual([component.name for component in cores], [
            "core_22.127937500GHz", "core_42.827937500GHz"
        ])
        self.assertAlmostEqual(
            component_position_mas(cores[0], frequencies[0], config.core_shift)[0],
            0.2894427191,
        )
        self.assertAlmostEqual(cores[0].x_mas, 0.2894427191)
        self.assertAlmostEqual(cores[0].y_mas, 0.0552786405)
        self.assertEqual(
            component_position_mas(cores[1], frequencies[1], config.core_shift), (0.2, 0.1)
        )
        self.assertNotEqual(
            component_position_mas(cores[0], frequencies[1], config.core_shift),
            component_position_mas(cores[1], frequencies[1], config.core_shift),
        )

    def test_subcomponent_normalization_matches_legacy_core_best_fit(self):
        payload = json.loads(json.dumps(CONFIG))
        payload["core_shift"]["subcomponents"] = True
        config = self.load(payload)
        frequencies = [22.1279375e9, 42.8279375e9]
        core = config.components[0]
        generated = [component for component in materialize_core_subcomponents(
            config.components, config.core_shift, frequencies
        ) if component.core_frequency_ghz is not None]
        targets = [evaluate_component_fluxes(freq, [core])[0][1] for freq in frequencies]
        modeled = [sum(flux for _, flux in evaluate_component_fluxes(freq, generated))
                   for freq in frequencies]
        ratios = [modeled[idx] / generated[0].sm_jy / targets[idx] for idx in range(2)]
        expected_scale = sum(ratios) / sum(value * value for value in ratios)
        self.assertAlmostEqual(generated[0].sm_jy, expected_scale, places=12)
        # Every generated sphere has the requested optically-thick asymptote.
        self.assertAlmostEqual(
            homogeneous_sphere_flux_jy(1e6, 1.0, generated[0].nu_m_ghz * 1e9, generated[0].alpha_thin)
            / homogeneous_sphere_flux_jy(2e6, 1.0, generated[0].nu_m_ghz * 1e9, generated[0].alpha_thin),
            2 ** -OPTICALLY_THICK_INDEX,
            places=3,
        )

    def test_subcomponents_emit_at_all_bands_and_swap_peak(self):
        payload = json.loads(json.dumps(CONFIG))
        payload["core_shift"]["subcomponents"] = True
        config = self.load(payload)
        frequencies = [15.0e9, 22.1279375e9, 42.8279375e9]
        generated = [component for component in materialize_core_subcomponents(
            config.components, config.core_shift, frequencies
        ) if component.core_frequency_ghz is not None]
        self.assertEqual(len(generated), 3)
        for frequency, associated in zip(frequencies, generated):
            fluxes = dict(evaluate_component_fluxes(frequency, generated))
            self.assertTrue(all(flux > 0.0 for flux in fluxes.values()))
            self.assertEqual(max(fluxes.values()), fluxes[associated])

    def test_subcomponent_rejects_nonfinite_frequency(self):
        payload = json.loads(json.dumps(CONFIG))
        payload["core_shift"]["subcomponents"] = True
        config = self.load(payload)
        for frequency in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, "frequencies must be positive"):
                materialize_core_subcomponents(config.components, config.core_shift, [frequency])


class OutputDirectoryTests(unittest.TestCase):
    def test_overwrite_cli_defaults_false_and_flag_enables_it(self):
        with patch("sys.argv", ["syngen_pipeline.py"]):
            self.assertFalse(parse_args().overwrite)
        with patch("sys.argv", ["syngen_pipeline.py", "--overwrite"]):
            self.assertTrue(parse_args().overwrite)

    def test_missing_directory_uses_requested_name(self):
        with tempfile.TemporaryDirectory() as directory:
            requested = Path(directory) / "synthetic_vis"
            self.assertEqual(resolve_output_dir(requested), requested)
            self.assertTrue(requested.is_dir())

    def test_existing_directories_use_lowest_free_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            requested = Path(directory) / "synthetic_vis"
            requested.mkdir()
            (requested.parent / "synthetic_vis_2").mkdir()

            selected = resolve_output_dir(requested)

            self.assertEqual(selected, requested.parent / "synthetic_vis_3")
            self.assertTrue(selected.is_dir())

    def test_existing_files_survive_suffix_and_overwrite_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            requested = Path(directory) / "synthetic_vis"
            requested.mkdir()
            marker = requested / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            self.assertEqual(resolve_output_dir(requested), requested.parent / "synthetic_vis_2")
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertEqual(resolve_output_dir(requested, overwrite=True), requested)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_file_at_requested_output_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            requested = Path(directory) / "synthetic_vis"
            requested.write_text("occupied", encoding="utf-8")

            with self.assertRaisesRegex(NotADirectoryError, "not a directory"):
                resolve_output_dir(requested)


if __name__ == "__main__":
    unittest.main()
