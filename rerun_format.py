#!/usr/bin/env python3
"""
Load an Open Photogrammetry Format (OPF) project and display the cameras and point cloud.

Requires Python 3.10 or higher because of [pyopf](https://pypi.org/project/pyopf/).
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import requests
import rerun as rr
import tqdm
from PIL import Image
from pyopf.io import load
from pyopf.resolve import resolve

import os
import open3d as o3d

DESCRIPTION = """
# Open Photogrammetry Format

Visualizes an Open Photogrammetry Format (OPF) project, displaying the cameras and point cloud.

The full source code for this example is available
[on GitHub](https://github.com/rerun-io/rerun/blob/latest/examples/python/open_photogrammetry_format).

### Links
* [OPF specification](https://pix4d.github.io/opf-spec/index.html)
* [Dataset source](https://support.pix4d.com/en-us/articles/360000235126#OPF)
* [pyopf](https://github.com/Pix4D/pyopf)
"""


@dataclass
class DatasetSpec:
    dir_name: str
    url: str


DATASETS = {
    "hd": DatasetSpec(
        "hd", "..."
    ),


    "olympic": DatasetSpec(
        "olympic_flame", "https://s3.amazonaws.com/mics.pix4d.com/example_datasets/olympic_flame.zip"
    ),
    "rainwater": DatasetSpec(
        "catch_rainwater_demo", "https://s3.amazonaws.com/mics.pix4d.com/example_datasets/catch_rainwater_demo.zip"
    ),
    "rivaz": DatasetSpec("rivaz_demo", "https://s3.amazonaws.com/mics.pix4d.com/example_datasets/rivaz_demo.zip"),
}
DATASET_DIR: Final = Path(__file__).parent / "dataset"

def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %s to %s", url, path)
    response = requests.get(url, stream=True)
    with tqdm.tqdm.wrapattr(
        open(path, "wb"),
        "write",
        miniters=1,
        total=int(response.headers.get("content-length", 0)),
        desc=f"Downloading {path.name}",
    ) as f:
        for chunk in response.iter_content(chunk_size=4096):
            f.write(chunk)

def unzip_dir(archive: Path, destination: Path) -> None:
    """Unzip the archive to the destination, using tqdm to display progress."""
    logging.info("Extracting %s to %s", archive, destination)
    with zipfile.ZipFile(archive, "r") as zip_ref:
        zip_ref.extractall(destination)

class OPFProject:
    def __init__(self, path: Path, log_as_frames: bool = True) -> None:
        """
        Create a new OPFProject from the given path.

        Parameters
        ----------
        path : Path
            Path to the project file.
        log_as_frames : bool, optional
            Whether to log the cameras as individual frames, by default True

        """
        import os

        self.path = path
        # TODO(Pix4D/pyopf#6): https://github.com/Pix4D/pyopf/issues/6
        # pyopf doesn't seem to work with regular windows paths, but a "UNC dos path" works
        path_as_str = "\\\\.\\" + str(path.absolute()) if os.name == "nt" else str(path)
        self.project = resolve(load(path_as_str))
        self.log_as_frames = log_as_frames

    @classmethod
    def from_dataset(cls, dataset: str, log_as_frames: bool = True) -> OPFProject:
        """
        Download the dataset if necessary and return the project file.

        Parameters
        ----------
        dataset : str
            Name of the dataset to download.
        log_as_frames : bool, optional
            Whether to log the cameras as individual frames, by default True

        """
        spec = DATASETS[dataset]
        if not (DATASET_DIR / spec.dir_name).exists():
            zip_file = DATASET_DIR / f"{dataset}.zip"
            if not zip_file.exists():
                download_file(DATASETS[dataset].url, zip_file)
            unzip_dir(DATASET_DIR / f"{dataset}.zip", DATASET_DIR)

        return cls(DATASET_DIR / spec.dir_name / "project.opf", log_as_frames=log_as_frames)

    def _merge_ply_files(self, root_path):
        merged_pcd = o3d.geometry.PointCloud()

        i = 0
        for folder in os.listdir(root_path + '/result'):
            folder_path = os.path.join(root_path + '/result', folder)
            if not os.path.isdir(folder_path):
                continue
            print(i)
            i = i+1
            for file in os.listdir(folder_path):
                if file.startswith("optCUDA_after_") and file.endswith(".ply"):
                    file_path = os.path.join(folder_path, file)
                    pcd = o3d.io.read_point_cloud(file_path)

                    colors = np.asarray(pcd.colors)
                    valid_indices = np.any(colors != 0, axis=1)

                    pcd = pcd.select_by_index(np.where(valid_indices)[0])

                    merged_pcd += pcd
                    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.01)


        #colors = np.asarray(merged_pcd.colors)
        #valid_indices = np.any(colors != 0, axis=1)
        #merged_pcd = merged_pcd.select_by_index(np.where(valid_indices)[0])

        save_path = os.path.join(root_path, 'result/optCUDA_after_merge.ply')
        o3d.io.write_point_cloud("/home/eric/github/data/optCUDA_after_merge.ply", merged_pcd)

        return  merged_pcd

    def _save_to_memmap(self, pcd, points_path='points_memmap.dat', colors_path='colors_memmap.dat'):
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32)

        points_memmap = np.memmap(points_path, dtype=np.float32, mode='w+', shape=points.shape)
        colors_memmap = np.memmap(colors_path, dtype=np.float32, mode='w+', shape=colors.shape)

        points_memmap[:] = points[:]
        colors_memmap[:] = colors[:]

        points_memmap.flush()
        colors_memmap.flush()

        loaded_points = np.memmap('points_memmap.dat', dtype=np.float32, mode='r', shape=points.shape)
        loaded_colors = np.memmap('colors_memmap.dat', dtype=np.float32, mode='r', shape=colors.shape)


        return loaded_points, loaded_colors

    def log_point_cloud(self, root_path) -> None:
        """Log the project's point cloud."""
        # points = self.project.point_cloud_objs[0].nodes[0]
        # rr.log("world/points", rr.Points3D(points.position, colors=points.color), static=True)


        merged_pcd = self._merge_ply_files(root_path)
        loaded_points, loaded_colors = self._save_to_memmap(merged_pcd)

        rr.log("world/points", rr.Points3D(loaded_points, colors=loaded_colors), static=True)

    def _read_cameras(self, jpeg_quality: int | None, root_path: str) -> None:
        image_path = os.path.join(root_path, 'result/color')
        matrix_path = os.path.join(root_path, 'result/ob_in_cam')

        cal_path = os.path.join(root_path, 'result/cam_K.txt')
        cam_K = np.loadtxt(cal_path)

        image_files = [f for f in os.listdir(image_path) if f.endswith(('.jpg', '.png'))]
        matrix_files = [f for f in os.listdir(matrix_path) if f.endswith('.txt')]

        def extract_number(file_name):
            return int(os.path.splitext(file_name)[0])

        image_files.sort(key=extract_number)
        matrix_files.sort(key=extract_number)

        for i, image_file in enumerate(image_files):
            image_file_path = os.path.join(image_path, image_file)
            matrix_file_path = os.path.join(matrix_path, matrix_files[i])

            if not os.path.exists(matrix_file_path):
                continue

            entity = f"world/cameras/{i}"

            # 이미지 로드
            img = Image.open(image_file_path)

            # 4x4 행렬 로드
            matrix = np.loadtxt(matrix_file_path)
            matrix = np.linalg.inv(matrix)

            if matrix.shape != (4, 4):
                print(f"Error: Matrix shape is not 4x4 for file {matrix_file_path}")
                continue

            rotation_matrix = matrix[:3, :3]
            translation_vector = matrix[:3, 3]

            if self.log_as_frames:
                rr.set_time_sequence("image", i)
                entity = "world/cameras"
            else:
                entity = f"world/cameras/{i}"

            rr.log(entity, rr.Transform3D(translation=translation_vector, mat3x3=rotation_matrix))
            rr.log(entity + "/image",
                rr.Pinhole(
                    width=img.size[0],
                    height=img.size[1],
                    focal_length=[cam_K[0][0], cam_K[1][1]],
                    principal_point=[cam_K[0][2], cam_K[1][2]],
                    #principal_point=[img.size[0]/2, img.size[1]/2],
                    #camera_xyz=rr.ViewCoordinates.RUB,
                ),
            )
            rr.log(entity + "/image", rr.Image(np.array(img)).compress(jpeg_quality=95))
            #rr.log(entity + "/image/rgb", rr.Image(np.array(img)).compress(jpeg_quality=95))

    def log_calibrated_cameras(self, jpeg_quality: int | None) -> None:
        """
        Log the project's calibrated cameras as individual frames.

        Logging all cameras in a single frame is also possible, but clutter the default view with too many image views.
        """
        sensor_map = {sensor.id: sensor for sensor in self.project.input_cameras.sensors}
        calib_sensor_map = {sensor.id: sensor for sensor in self.project.calibration.calibrated_cameras.sensors}

        for i, (camera, calib_camera) in enumerate(
            zip(
                self.project.camera_list.cameras,
                self.project.calibration.calibrated_cameras.cameras,
            )
        ):
            if not str(camera.uri).endswith(".jpg"):
                continue

            if self.log_as_frames:
                rr.set_time_sequence("image", i)
                entity = "world/cameras"
            else:
                entity = f"world/cameras/{i}"

            sensor = sensor_map[calib_camera.sensor_id]
            calib_sensor = calib_sensor_map[calib_camera.sensor_id]

            # Specification for the omega, phi, kappa angles:
            # https://pix4d.github.io/opf-spec/specification/calibrated_cameras.html#calibrated-camera
            omega, phi, kappa = tuple(np.deg2rad(a) for a in calib_camera.orientation_deg)
            rot = (
                np.array([
                    [1, 0, 0],
                    [0, np.cos(omega), -np.sin(omega)],
                    [0, np.sin(omega), np.cos(omega)],
                ])
                @ np.array([
                    [np.cos(phi), 0, np.sin(phi)],
                    [0, 1, 0],
                    [-np.sin(phi), 0, np.cos(phi)],
                ])
                @ np.array([
                    [np.cos(kappa), -np.sin(kappa), 0],
                    [np.sin(kappa), np.cos(kappa), 0],
                    [0, 0, 1],
                ])
            )

            rr.log(entity, rr.Transform3D(translation=calib_camera.position, mat3x3=rot))

            assert calib_sensor.internals.type == "perspective"

            # RUB coordinate system specified in https://pix4d.github.io/opf-spec/specification/projected_input_cameras.html#coordinate-system-specification
            rr.log(
                entity + "/image",
                rr.Pinhole(
                    resolution=sensor.image_size_px,
                    focal_length=calib_sensor.internals.focal_length_px,
                    principal_point=calib_sensor.internals.principal_point_px
                ),
            )

            if jpeg_quality is not None:
                with Image.open(self.path.parent / camera.uri) as img:
                    rr.log(entity + "/image/rgb", rr.Image(np.array(img)).compress(jpeg_quality=jpeg_quality))
            else:
                rr.log(entity + "/image/rgb", rr.ImageEncoded(path=self.path.parent / camera.uri))


def main() -> None:
    logging.getLogger().addHandler(rr.LoggingHandler())
    logging.getLogger().setLevel("INFO")

    parser = argparse.ArgumentParser(
        description="Load an Open Photogrammetry Format (OPF) project and display the cameras and point cloud."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS.keys(),
        default="hd",
        help="Run on a demo image automatically downloaded",
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Log all cameras globally instead of as individual frames in the timeline.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="If specified, compress the camera images with the given JPEG quality.",
    )

    rr.script_add_args(parser)

    args, unknown = parser.parse_known_args()
    for arg in unknown:
        logging.warning(f"unknown arg: {arg}")


    print("args.dataset", args.dataset)
    # load the data set
    project = OPFProject.from_dataset(args.dataset, log_as_frames=not args.no_frames)

    root_path = "/home/eric/github/data/slam/object/HD_hull"

    # display everything in Rerun
    rr.script_setup(args, "rerun_example_open_photogrammetry_format")
    rr.log("description", rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN), timeless=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    project.log_point_cloud(root_path)
    project._read_cameras(jpeg_quality=args.jpeg_quality, root_path=root_path)
    #project.log_calibrated_cameras(jpeg_quality=args.jpeg_quality)
    rr.script_teardown(args)


if __name__ == "__main__":
    main()
