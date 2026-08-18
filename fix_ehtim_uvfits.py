#!/usr/bin/env python3
"""Convert an LL-only random-groups UVFITS file for eht-imaging.

The conversion adds a flagged RR slot.  It does not relabel the original LL
visibility as RR, and it never overwrites the input file.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any


HISTORY = (
    "Converted LL-only STOKES axis (CRVAL3=-2) to RR+LL layout for "
    "eht-imaging compatibility."
)
HISTORY_2 = "Original LL retained as LL; dummy RR added with zero weight."


def _require_dependencies():
    try:
        import numpy as np
        from astropy.io import fits
    except ModuleNotFoundError as exc:
        raise RuntimeError("converter requires astropy and numpy") from exc
    return fits, np


def _same_number(value: Any, expected: float) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _arrays_equal(left, right, np) -> bool:
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except (TypeError, ValueError):
        return bool(np.array_equal(left, right))


def _check_input(header) -> None:
    found = {
        "CTYPE3": header.get("CTYPE3"),
        "NAXIS3": header.get("NAXIS3"),
        "CRVAL3": header.get("CRVAL3"),
        "CRPIX3": header.get("CRPIX3"),
        "CDELT3": header.get("CDELT3"),
    }
    print("Input polarization axis:")
    for key, value in found.items():
        print(f"  {key} = {value!r}")

    errors = []
    if str(found["CTYPE3"]).strip().upper() != "STOKES":
        errors.append(f"CTYPE3={found['CTYPE3']!r}, expected 'STOKES'")
    try:
        naxis3_ok = int(found["NAXIS3"]) == 1
    except (TypeError, ValueError):
        naxis3_ok = False
    if not naxis3_ok:
        errors.append(f"NAXIS3={found['NAXIS3']!r}, expected 1")
    if not _same_number(found["CRVAL3"], -2.0):
        errors.append(f"CRVAL3={found['CRVAL3']!r}, expected -2")
    if errors:
        raise ValueError("input is not LL-only UVFITS: " + "; ".join(errors))


def _data_axis(header, data_shape: tuple[int, ...], fits_axis: int) -> int:
    """Return NumPy axis for a FITS axis after checking random-groups layout."""
    naxis = int(header["NAXIS"])
    fits_shape = tuple(int(header[f"NAXIS{i}"]) for i in range(2, naxis + 1))
    expected = (int(header["GCOUNT"]), *reversed(fits_shape))
    if data_shape != expected:
        raise ValueError(
            "unsupported random-groups DATA layout: "
            f"header implies {expected}, DATA has {data_shape}"
        )
    if not 2 <= fits_axis <= naxis:
        raise ValueError(f"FITS axis {fits_axis} is not a random-groups DATA axis")
    axis = 1 + naxis - fits_axis
    if data_shape[axis] != int(header[f"NAXIS{fits_axis}"]):
        raise ValueError(f"FITS axis {fits_axis} does not match DATA shape")
    return axis


def _component_axis(header, data_shape: tuple[int, ...]) -> int:
    axes = [
        axis
        for axis in range(2, int(header["NAXIS"]) + 1)
        if str(header.get(f"CTYPE{axis}", "")).strip().upper() == "COMPLEX"
    ]
    if axes != [2] or int(header["NAXIS2"]) != 3:
        raise ValueError(
            "expected UVFITS visibility component axis CTYPE2='COMPLEX', NAXIS2=3"
        )
    return _data_axis(header, data_shape, 2)


def _new_group_data(primary, new_data, fits, np):
    old_data = primary.data
    parnames = list(old_data.parnames or [])
    pardata = [np.array(old_data.field(index), copy=True) for index in range(len(parnames))]
    header = primary.header
    return fits.GroupData(
        new_data,
        bitpix=int(header["BITPIX"]),
        pardata=pardata,
        parnames=parnames,
        bscale=header.get("BSCALE"),
        bzero=header.get("BZERO"),
        parbscales=[header.get(f"PSCAL{index + 1}") for index in range(len(parnames))],
        parbzeros=[header.get(f"PZERO{index + 1}") for index in range(len(parnames))],
    )


def _extension_snapshot(hdul, np):
    snapshot = []
    for hdu in hdul[1:]:
        data = None if hdu.data is None else np.array(hdu.data, copy=True)
        snapshot.append((type(hdu).__name__, hdu.header.tostring(), data))
    return snapshot


def _verify(input_hdul, output_path: Path, fits, np, extension_snapshot) -> None:
    with fits.open(output_path, memmap=False) as output_hdul:
        source = input_hdul[0]
        result = output_hdul[0]
        expected = {"CTYPE3": "STOKES", "NAXIS3": 2, "CRVAL3": -1, "CRPIX3": 1, "CDELT3": -1}
        for key, value in expected.items():
            actual = result.header.get(key)
            if isinstance(value, str):
                ok = str(actual).strip().upper() == value
            else:
                ok = _same_number(actual, value)
            if not ok:
                raise ValueError(f"output verification failed: {key}={actual!r}, expected {value!r}")

        source_array = np.array(source.data.data, copy=False)
        result_array = np.array(result.data.data, copy=False)
        source_pol_axis = _data_axis(source.header, source_array.shape, 3)
        result_pol_axis = _data_axis(result.header, result_array.shape, 3)
        source_component_axis = _component_axis(source.header, source_array.shape)
        result_component_axis = _component_axis(result.header, result_array.shape)
        if result_array.shape[result_pol_axis] != 2:
            raise ValueError("output verification failed: STOKES DATA axis is not length 2")

        source_ll = np.take(source_array, 0, axis=source_pol_axis)
        result_ll = np.take(result_array, 1, axis=result_pol_axis)
        result_rr = np.take(result_array, 0, axis=result_pol_axis)
        if not _arrays_equal(source_ll, result_ll, np):
            raise ValueError("output verification failed: LL data changed")
        source_component_axis -= source_component_axis > source_pol_axis
        result_component_axis -= result_component_axis > result_pol_axis
        source_weight = np.take(source_ll, 2, axis=source_component_axis)
        result_weight = np.take(result_ll, 2, axis=result_component_axis)
        if not _arrays_equal(source_weight, result_weight, np):
            raise ValueError("output verification failed: LL weights changed")
        for component in range(3):
            rr_component = np.take(result_rr, component, axis=result_component_axis)
            if not _arrays_equal(rr_component, np.zeros_like(rr_component), np):
                raise ValueError("output verification failed: dummy RR is not all zero")

        if len(source.data) != len(result.data):
            raise ValueError("output verification failed: visibility group count changed")
        if int(source.header["GCOUNT"]) != int(result.header["GCOUNT"]):
            raise ValueError("output verification failed: GCOUNT changed")
        for axis in range(1, int(source.header["NAXIS"]) + 1):
            if axis != 3 and source.header.get(f"NAXIS{axis}") != result.header.get(f"NAXIS{axis}"):
                raise ValueError(f"output verification failed: NAXIS{axis} changed")

        if len(output_hdul[1:]) != len(extension_snapshot):
            raise ValueError("output verification failed: extension HDU count changed")
        for hdu, (class_name, header_text, data) in zip(output_hdul[1:], extension_snapshot):
            if type(hdu).__name__ != class_name or hdu.header.tostring() != header_text:
                raise ValueError("output verification failed: extension HDU metadata changed")
            if data is None:
                preserved = hdu.data is None
            else:
                preserved = hdu.data is not None and _arrays_equal(
                    data, np.array(hdu.data, copy=False), np
                )
            if not preserved:
                raise ValueError("output verification failed: extension HDU data changed")


def _check_ehtim(output_path: Path) -> bool:
    try:
        import ehtim as eh
    except ModuleNotFoundError:
        return False
    eh.obsdata.load_uvfits(str(output_path))
    return True


def convert(input_path: Path, output_path: Path | None = None, overwrite: bool = False, check_ehtim: bool = False) -> Path:
    fits, np = _require_dependencies()
    input_path = input_path.expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}-fix{input_path.suffix}")
    output_path = output_path.expanduser()
    if input_path.resolve() == output_path.resolve():
        raise ValueError("refusing to overwrite the source file")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists; use --overwrite to replace it: {output_path}")

    with fits.open(input_path, memmap=False) as input_hdul:
        print(f"Input: {input_path}")
        primary = input_hdul[0]
        if not isinstance(primary, fits.GroupsHDU) or primary.data is None:
            raise ValueError("input primary HDU is not a random-groups UVFITS HDU")
        _check_input(primary.header)
        input_array = np.array(primary.data.data, copy=True)
        print("Detected polarization: LL only")
        pol_axis = _data_axis(primary.header, input_array.shape, 3)
        _component_axis(primary.header, input_array.shape)
        new_shape = list(input_array.shape)
        new_shape[pol_axis] = 2
        converted = np.zeros(new_shape, dtype=input_array.dtype)
        target = [slice(None)] * converted.ndim
        target[pol_axis] = 1
        source = [slice(None)] * input_array.ndim
        source[pol_axis] = 0
        converted[tuple(target)] = input_array[tuple(source)]

        header = primary.header.copy()
        header["NAXIS3"] = 2
        header["CRPIX3"] = 1
        header["CRVAL3"] = -1
        header["CDELT3"] = -1
        header.add_history(HISTORY)
        header.add_history(HISTORY_2)
        new_primary = fits.GroupsHDU(
            data=_new_group_data(primary, converted, fits, np),
            header=header,
        )
        extension_snapshot = _extension_snapshot(input_hdul, np)
        output_hdul = fits.HDUList([new_primary, *input_hdul[1:]])
        try:
            # AIPS UVFITS commonly uses legacy XTENSION='A3DTABLE'. Astropy
            # accepts it when reading but rejects it during its write-time
            # BinTableHDU verification; preserve the original extension cards.
            output_hdul.writeto(output_path, overwrite=overwrite, output_verify="ignore")
            _verify(input_hdul, output_path, fits, np, extension_snapshot)
        finally:
            output_hdul.close()

    if check_ehtim:
        if _check_ehtim(output_path):
            print("ehtim compatibility check passed.")
        else:
            print("ehtim not installed; compatibility check skipped.")
    print(f"\nOutput: {output_path}")
    print("Verification passed.")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert LL-only random-groups UVFITS to eht-imaging RR+LL layout."
    )
    parser.add_argument("input", type=Path, help="LL-only .uvf/.uvfits input file")
    parser.add_argument("--output", type=Path, help="output path; default inserts -fix before extension")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing output")
    parser.add_argument("--check-ehtim", action="store_true", help="load output with ehtim when installed")
    args = parser.parse_args()
    try:
        convert(args.input, args.output, args.overwrite, args.check_ehtim)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
