#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "zarr",
#     "imageio",
# ]
# ///
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
import shutil
import logging
from argparse import ArgumentParser

import imageio.v3 as iio
import numpy as np
import zarr
import zarr.codecs

logger = logging.getLogger(__name__)

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"

DEFAULT_OME_ZARR_VERSION = "0.5+rfc3"


def eprint(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


class DataWriter:
    def __init__(
        self, force=False, ome_zarr_version=DEFAULT_OME_ZARR_VERSION, data_dir=data_dir
    ) -> None:
        self.force = force
        self.ome_zarr_version = ome_zarr_version
        self.data_dir = data_dir

    def path_exists(self, path: Path) -> bool:
        if path.exists():
            if self.force:
                shutil.rmtree(path)
            else:
                eprint(f"{path} exists, skipping download. Use --force to overwrite.")
                return True
        return False

    def get_ecg(self):
        """1D electrocardiogram signal.

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.datasets.electrocardiogram.html#scipy.datasets.electrocardiogram
        """
        zarr_root = self.data_dir / "ecg_1d.ome.zarr"
        if self.path_exists(zarr_root):
            return None

        with TemporaryDirectory() as path:
            ecg_path = Path(path) / "ecg.dat"
            logger.info(
                "Downloading https://raw.githubusercontent.com/scipy/dataset-ecg/main/ecg.dat"
            )
            urlretrieve(
                "https://raw.githubusercontent.com/scipy/dataset-ecg/main/ecg.dat",
                ecg_path,
            )
            with open(ecg_path, "rb") as f:
                multi = np.load(f)
                ecg = multi["ecg"].astype("int64")
        ecg = np.astype((ecg - 1024) / 200.0, "float32")

        attributes = {
            "provenance": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.datasets.electrocardiogram.html#scipy.datasets.electrocardiogram",
            "description": "1D electrocardiogram signal from scipy.datasets.electrocardiogram",
            "ome": {
                "version": self.ome_zarr_version,
                "multiscales": [
                    {
                        "name": "ecg",
                        "axes": [{"name": "t", "type": "time", "unit": "seconds"}],
                        "datasets": [
                            {
                                "path": "s0",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [1 / 360],  # 360 Hz sampling rate
                                    },
                                    {
                                        "type": "translation",
                                        "translation": [
                                            19 * 60 + 35
                                        ],  # 19:35 minutes offset
                                    },
                                ],
                            }
                        ],
                    }
                ],
            },
        }
        logger.info(f"Writing OME-Zarr to {zarr_root}")
        grp = zarr.create_group(
            zarr_root,
            attributes=attributes,
        )
        grp.create_array(
            "s0", data=ecg, dimension_names=["t"], compressors=[zarr.codecs.GzipCodec()]
        )

        return ecg

    def get_astronaut_xcy(self):
        zarr_root = self.data_dir / "astronaut_xcy.ome.zarr"
        if self.path_exists(zarr_root):
            return None

        arr = iio.imread("imageio:astronaut.png")
        transposed = arr.transpose((0, 2, 1))

        attributes = {
            "description": "3D dataset with non-standard XCY axis order",
            "ome": {
                "version": self.ome_zarr_version,
                "multiscales": [
                    {
                        "name": "astronaut_xcy",
                        "axes": [
                            {"name": "x", "type": "space"},
                            {"name": "c", "type": "channel"},
                            {"name": "y", "type": "space"},
                        ],
                        "datasets": [
                            {
                                "path": "s0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                                ],
                            }
                        ],
                    }
                ],
            },
        }
        logger.info(f"Writing OME-Zarr to {zarr_root}")
        grp = zarr.create_group(
            zarr_root,
            attributes=attributes,
        )
        grp.create_array(
            "s0",
            data=transposed,
            dimension_names=["x", "c", "y"],
            compressors=[zarr.codecs.GzipCodec()],
        )

        return arr

    def calc_ramp_6d(self):
        zarr_root = self.data_dir / "ramp_6d.ome.zarr"
        if self.path_exists(zarr_root):
            return None

        attributes = {
            "description": "6D dataset with zero border and linear ramp in center",
            "ome": {
                "version": self.ome_zarr_version,
                "multiscales": [
                    {
                        "name": "ramp_6d",
                        "axes": [
                            {"name": "a"},
                            {"name": "b"},
                            {"name": "c"},
                            {"name": "d"},
                            {"name": "e"},
                            {"name": "f"},
                        ],
                        "datasets": [
                            {
                                "path": "s0",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
        }
        outer = np.zeros((16,) * 6, dtype="float32")
        inner = np.linspace(0, 1, 8**6, dtype="float32").reshape((8,) * 6)
        outer[(slice(4, 12),) * 6] = inner

        logger.info(f"Writing OME-Zarr to {zarr_root}")
        grp = zarr.create_group(
            zarr_root,
            attributes=attributes,
        )
        grp.create_array(
            "s0",
            data=outer,
            dimension_names=list("abcdef"),
            compressors=[zarr.codecs.GzipCodec()],
        )

    def calc_trig_6d(self):
        zarr_root = data_dir / "trig_6d.ome.zarr"
        if self.path_exists(zarr_root):
            return None

        attributes = {
            "description": "6D dataset with trig function values",
            "ome": {
                "version": "0.5+rfc3",
                "multiscales": [
                    {
                        "name": "trig_6d",
                        "axes": [
                            {"name": "t", "type": "time"},
                            {"name": "c", "type": "channel"},
                            {"name": "z", "type": "space"},
                            {"name": "y", "type": "space"},
                            {"name": "x", "type": "space"},
                            {"name": "w"},
                        ],
                        "datasets": [
                            {
                                "path": "s0",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
        }
        shape = (32, 3, 32, 40, 48, 56)
        scale = np.linspace(1.0, 0.2, 16)
        t_scales = np.array([*scale, *scale[::-1]])

        def fn(t, c, z, y, x, w):
            pd = np.pi * 2 / 32
            zyxw = (
                np.sin(z * pd) + np.cos(y * pd) + np.sin(x * pd) + np.cos(w * pd)
            ) / 4
            t_scaled = zyxw * t_scales[np.astype(t, int)]
            c_stepped = t_scaled / (c + 1)
            return np.astype(c_stepped, "float32")

        arr = np.fromfunction(fn, shape, dtype="float32")
        arr /= arr.max()
        arr *= 255
        arr = np.astype(arr, "uint8")

        logger.info(f"Writing OME-Zarr to {zarr_root}")
        grp = zarr.create_group(
            zarr_root,
            attributes=attributes,
        )
        grp.create_array(
            "s0",
            data=arr,
            dimension_names=["t", "c", "z", "y", "x", "w"],
            compressors=[zarr.codecs.GzipCodec()],
        )


def main():
    parser = ArgumentParser(description="Download ECG dataset and store as OME-Zarr.")
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="force re-download and overwrite existing data",
    )
    parser.add_argument(
        "--ome-zarr-version",
        "-o",
        help=f"OME-Zarr version string to use, default '{DEFAULT_OME_ZARR_VERSION}'",
        default=DEFAULT_OME_ZARR_VERSION,
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        help="data directory to store OME-Zarr datasets",
        default=data_dir,
    )
    args = parser.parse_args()

    w = DataWriter(args.force, args.ome_zarr_version, args.data_dir)
    w.get_astronaut_xcy()
    w.get_ecg()
    # w.calc_trig_6d(force=args.force)
    w.calc_ramp_6d()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
