import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mf_syngen.ground_truth import (
    Beam,
    calculate_spectral_index,
    derive_common_grid,
    parse_args,
    _point_pixel_boundary_offsets,
    select_beam,
    unique_frequency_pairs,
)
from mf_syngen.model import Component, MAS_TO_RAD, PipelineConfig


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class BeamTests(unittest.TestCase):
    def test_custom_beam_validation_and_unit_conversion(self):
        beam = Beam.from_mas(2.0, 1.0, 25.0)
        self.assertAlmostEqual(beam.major_rad, 2.0 * MAS_TO_RAD)
        self.assertAlmostEqual(beam.minor_rad, MAS_TO_RAD)
        self.assertAlmostEqual(beam.pa_rad, math.radians(25.0))
        self.assertAlmostEqual(beam.major_mas, 2.0)
        self.assertAlmostEqual(beam.minor_mas, 1.0)
        self.assertAlmostEqual(beam.pa_deg, 25.0)

        for values in [(0, 1, 0), (1, 0, 0), (1, 2, 0)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Beam.from_mas(*values)

    def test_biggest_and_smallest_use_elliptical_area(self):
        wide = Beam.from_mas(4.0, 1.0, 10.0)
        large_area = Beam.from_mas(3.0, 2.0, 20.0)
        self.assertIs(select_beam([wide, large_area], "biggest"), large_area)
        self.assertIs(select_beam([wide, large_area], "smallest"), wide)
        self.assertIs(select_beam([wide, large_area], "b"), large_area)
        self.assertIs(select_beam([wide, large_area], "s"), wide)

    def test_average_beam_uses_periodic_position_angle(self):
        average = select_beam(
            [Beam.from_mas(4.0, 2.0, 179.0), Beam.from_mas(2.0, 1.0, 1.0)],
            "average",
        )
        self.assertAlmostEqual(average.major_mas, 3.0)
        self.assertAlmostEqual(average.minor_mas, 1.5)
        distance_from_axis = min(abs(average.pa_deg), abs(average.pa_deg - 180.0))
        self.assertLess(distance_from_axis, 1e-8)


class GridTests(unittest.TestCase):
    def test_extent_sampling_and_power_of_two_pixel_count(self):
        fov_rad, npix = derive_common_grid(
            [(1.0, 0.0)],
            [Beam.from_mas(0.5, 0.2, 0.0)],
        )
        self.assertAlmostEqual(fov_rad, 4.0 * MAS_TO_RAD)
        self.assertEqual(npix, 256)

    def test_centred_model_uses_largest_major_axis_as_padding(self):
        fov_rad, npix = derive_common_grid(
            [(0.0, 0.0)],
            [Beam.from_mas(2.0, 1.0, 0.0), Beam.from_mas(1.0, 0.5, 0.0)],
        )
        self.assertAlmostEqual(fov_rad, 4.0 * MAS_TO_RAD)
        self.assertEqual(npix, 128)

    def test_manual_overrides_are_exact_and_clipping_is_rejected(self):
        beam = Beam.from_mas(0.5, 0.2, 0.0)
        fov_rad, npix = derive_common_grid(
            [(1.0, 0.0)], [beam], fov_mas=3.0, npix=300
        )
        self.assertAlmostEqual(fov_rad, 3.0 * MAS_TO_RAD)
        self.assertEqual(npix, 300)

        with self.assertRaisesRegex(ValueError, "clip"):
            derive_common_grid([(1.0, 0.0)], [beam], fov_mas=1.5)

    def test_point_on_pixel_boundary_is_nudged(self):
        config = PipelineConfig(
            expected_source=None,
            absolute_positions=True,
            if_indices=[0],
            components=[Component("core", 0.0, 0.3, 1.0, 20.0, -0.7)],
            core_shift=None,
        )
        pixel_size_rad = 0.25 * MAS_TO_RAD

        with patch("builtins.print") as printed:
            offsets = _point_pixel_boundary_offsets(config, 20e9, pixel_size_rad)

        self.assertEqual(offsets["core"][1], 0.0)
        self.assertGreater(offsets["core"][0], 0.0)
        self.assertIn("using (x, y)", printed.call_args.args[0])


class CliTests(unittest.TestCase):
    def test_defaults(self):
        args = parse_args([])
        self.assertEqual(args.config, Path("components_multifreq.json"))
        self.assertEqual(args.data_dir, Path("data"))
        self.assertEqual(args.out_dir, Path("output") / "ground_truth_images")
        self.assertEqual(args.weighting, "n")
        self.assertEqual(args.map_beam, "b")
        self.assertEqual(args.beam_scope, "p")
        self.assertEqual(args.mask_fraction, 1e-6)

    def test_short_options_and_independent_custom_beams(self):
        args = parse_args(
            [
                "-c", "model.json", "-d", "uv", "-o", "maps", "-i", "0", "2",
                "-w", "c", "-b", "2", "1", "25", "-m", "c",
                "-g", "a", "-B", "3", "1.5", "10", "-z", "0.001",
                "-f", "8", "-n", "512",
            ]
        )
        self.assertEqual(args.if_indices, [0, 2])
        self.assertEqual(args.image_beam, [2.0, 1.0, 25.0])
        self.assertEqual(args.spectral_beam, [3.0, 1.5, 10.0])
        self.assertEqual((args.fov_mas, args.npix), (8.0, 512))

    def test_custom_modes_require_their_own_beam(self):
        for argv in (["-w", "c"], ["-m", "c"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parse_args(argv)


class FrequencyTests(unittest.TestCase):
    def test_pairs_are_sorted_and_duplicates_are_rejected(self):
        self.assertEqual(
            unique_frequency_pairs([30.0, 10.0, 20.0]),
            [(10.0, 20.0), (10.0, 30.0), (20.0, 30.0)],
        )
        with self.assertRaisesRegex(ValueError, "Repeated"):
            unique_frequency_pairs([10.0, 10.0])


class MockedIntegrationTests(unittest.TestCase):
    class FakeImage:
        def __init__(self):
            self.imvec = None

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
            return MockedIntegrationTests.FakeImage()

    class FakeAlpha:
        def reshape(self, _shape):
            return [0.0]

    def test_three_frequencies_create_three_images_and_three_pairs(self):
        from mf_syngen import ground_truth as module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            frequencies = {"a.uvf": 10e9, "b.uvf": 20e9, "c.uvf": 30e9}
            for name in frequencies:
                (data_dir / name).write_bytes(b"")
            config_path = root / "config.json"
            config_path.write_text(
                '{"components":[{"name":"core","x_mas":0,"y_mas":0,'
                '"sm_jy":1,"nu_m_ghz":20,"alpha_thin":-0.7}]}',
                encoding="utf-8",
            )

            def load_uvfits(path, IF):
                del IF
                return SimpleNamespace(
                    rf=frequencies[Path(path).name],
                    source="test",
                    fit_beam=lambda **_: (2 * MAS_TO_RAD, MAS_TO_RAD, 0.0),
                )

            fake_eh = SimpleNamespace(obsdata=SimpleNamespace(load_uvfits=load_uvfits))

            def write_pdf(_alpha, output_path, *_args):
                output_path.write_bytes(b"PDF")

            with (
                patch.object(module, "load_ehtim_modules", return_value=(fake_eh, object)),
                patch.object(module, "build_model", return_value=self.FakeModel()),
                patch.object(module, "calculate_spectral_index", return_value=self.FakeAlpha()),
                patch.object(module, "_annotate_fits") as annotate,
                patch.object(module, "_plot_spectral_index", side_effect=write_pdf) as plot,
            ):
                outputs = module.run_ground_truth(
                    config_path=config_path,
                    data_dir=data_dir,
                    out_dir=root / "output" / "ground_truth_images",
                )

            self.assertEqual(len(outputs), 9)
            self.assertEqual(len(list((root / "output" / "ground_truth_images").glob("*.fits"))), 6)
            self.assertEqual(len(list((root / "output" / "ground_truth_images").glob("*.pdf"))), 3)
            self.assertEqual(annotate.call_count, 6)
            self.assertEqual(plot.call_count, 3)


@unittest.skipUnless(HAS_NUMPY, "NumPy is not installed")
class SpectralIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np

        cls.np = np

    def test_recovers_constant_spectral_index(self):
        alpha = -0.7
        low = self.np.array([[1.0, 2.0], [3.0, 4.0]])
        high = low * 2.0**alpha
        result = calculate_spectral_index(low, high, 10e9, 20e9)
        self.np.testing.assert_allclose(result, alpha)

    def test_masks_relative_zero_invalid_and_nonpositive_pixels(self):
        low = self.np.array([[10.0, 1e-8], [-1.0, self.np.nan]])
        high = self.np.array([[20.0, 1e-8], [1.0, 1.0]])
        result = calculate_spectral_index(low, high, 10e9, 20e9, 1e-6)
        self.assertAlmostEqual(result[0, 0], 1.0)
        self.assertTrue(self.np.isnan(result[0, 1:]).all())
        self.assertTrue(self.np.isnan(result[1, :]).all())

    def test_fully_invalid_images_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid"):
            calculate_spectral_index(
                self.np.zeros((2, 2)), self.np.ones((2, 2)), 10e9, 20e9
            )


if __name__ == "__main__":
    unittest.main()
