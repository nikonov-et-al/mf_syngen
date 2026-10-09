import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mf_syngen.model import Component, CoreShift, PipelineConfig, materialize_core_subcomponents


def _config(subcomponents):
    return PipelineConfig(
        expected_source=None,
        absolute_positions=True,
        if_indices=[0],
        components=[
            Component("core", 0.2, 0.1, 1.0, 30.0, -0.7),
            Component("jet", 0.8, -0.4, 0.3, 15.0, -1.2),
        ],
        core_shift=CoreShift(
            "core", 0.2, 0.1, 20.0, 40.0, 0.1, -0.05, 1.0, subcomponents
        ),
    )


class SyntheticIntegrationTests(unittest.TestCase):
    def test_run_materializes_complete_shared_list_for_all_bands(self):
        from mf_syngen import synthetic

        frequencies = {"low.uvf": 20e9, "high.uvf": 40e9, "mid.uvf": 30e9}
        observations = {
            name: SimpleNamespace(rf=rf, source="test") for name, rf in frequencies.items()
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for name in frequencies:
                (data / name).write_bytes(b"")
            fake_eh = SimpleNamespace(
                obsdata=SimpleNamespace(
                    load_uvfits=lambda path, IF: observations[Path(path).name]
                )
            )
            calls = []

            def fake_process(**kwargs):
                calls.append(kwargs)
                return root / (Path(kwargs["input_path"]).stem + ".out")

            with (
                patch.object(synthetic, "load_ehtim_modules", return_value=(fake_eh, object)),
                patch.object(synthetic, "process_file", side_effect=fake_process),
            ):
                synthetic.run_pipeline(data, _config(True), root / "out", None, False, 1, "ignore")

            self.assertEqual(len(calls), 3)
            self.assertEqual(
                sorted(c.core_frequency_ghz for c in calls[0]["components"] if c.core_frequency_ghz),
                [20.0, 30.0, 40.0],
            )
            self.assertTrue(all(call["components"] is calls[0]["components"] for call in calls))

    def test_disabled_mode_passes_original_component_set_unchanged(self):
        from mf_syngen import synthetic

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (data / "one.uvf").write_bytes(b"")
            original = _config(False).components
            fake_eh = SimpleNamespace(
                obsdata=SimpleNamespace(load_uvfits=lambda path, IF: SimpleNamespace(rf=20e9, source="test"))
            )
            captured = []
            with (
                patch.object(synthetic, "load_ehtim_modules", return_value=(fake_eh, object)),
                patch.object(synthetic, "process_file", side_effect=lambda **kw: captured.append(kw) or root / "x"),
            ):
                synthetic.run_pipeline(data, _config(False), root / "out", None, False, 1, "ignore")
            self.assertEqual(
                [component.name for component in captured[0]["components"]],
                [component.name for component in original],
            )
            self.assertTrue(all(component.core_frequency_ghz is None for component in captured[0]["components"]))


class GroundTruthIntegrationTests(unittest.TestCase):
    def test_ground_truth_uses_same_materialized_components_each_band(self):
        from mf_syngen import ground_truth

        class FakeImage:
            imvec = None

            def imarr(self):
                return [[1.0]]

            def blur_gauss(self, _beam):
                return self

            def copy(self):
                return self.__class__()

            def save_fits(self, path):
                Path(path).write_bytes(b"FITS")

        class FakeModel:
            def make_image(self, _fov, _npix):
                return FakeImage()

        frequencies = {"low.uvf": 20e9, "high.uvf": 40e9}
        observations = {
            name: SimpleNamespace(
                rf=rf,
                source="test",
                fit_beam=lambda **_: (2 * ground_truth.MAS_TO_RAD, ground_truth.MAS_TO_RAD, 0.0),
            )
            for name, rf in frequencies.items()
        }
        config = _config(True)
        builds = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for name in frequencies:
                (data / name).write_bytes(b"")
            fake_eh = SimpleNamespace(
                obsdata=SimpleNamespace(load_uvfits=lambda path, IF: observations[Path(path).name])
            )
            with (
                patch.object(ground_truth, "load_config", return_value=config),
                patch.object(ground_truth, "load_ehtim_modules", return_value=(fake_eh, object)),
                patch.object(ground_truth, "build_model", side_effect=lambda obs, fluxes, *args: builds.append(fluxes) or FakeModel()),
                patch.object(ground_truth, "_annotate_fits"),
                patch.object(ground_truth, "calculate_spectral_index", return_value=SimpleNamespace(reshape=lambda _: [0.0])),
                patch.object(ground_truth, "_plot_spectral_index", side_effect=lambda *args: Path(args[1]).write_bytes(b"PDF")),
            ):
                ground_truth.run_ground_truth(data_dir=data, out_dir=root / "out")

        self.assertEqual(len(builds), 2)
        expected = ["jet", "core_20.000000000GHz", "core_40.000000000GHz"]
        self.assertTrue(all([component.name for component, _ in fluxes] == expected for fluxes in builds))


if __name__ == "__main__":
    unittest.main()
