import datetime
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from PIL import Image
from pypcd4.pypcd4 import PointCloud
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from .susc_datamodels import (
    CameraCalibration,
    EgoPose,
    LidarPose,
    parse_label,
)
from .utils import (
    expand_scene_ranges,
    generate_token,
)


class Susc2NuscConverter:
    def __init__(
        self,
        susc_root: Path = Path("data/suscape_scenes"),
        output_root: Path = Path("output/susc_nusc"),
        version: str = "v1.0-mini",
        scenes: list[str] | None = None,
        fetch_map: bool = False,
    ) -> None:
        """Convert SUScape scenes to NuScenes format.

        Args:
            susc_root: Path to the root directory of the SUScape dataset containing scene folders.
            output_root: Path where the converted NuScenes dataset will be saved.
            version: NuScenes version string (e.g., v1.0-mini).
            scenes: List of scenes to convert. Supports range format like 'scene-000001_000005' or individual names. None means all scenes.
            fetch_map: Whether to automatically fetch OpenStreetMap data for the processed scenes.
        """  # noqa: E501
        self.susc_root = Path(susc_root)
        self.output_root = Path(output_root)
        self.version = version
        self.scenes = expand_scene_ranges(scenes)
        self.fetch_map = fetch_map

        self.nusc_root = self.output_root / self.version
        self.nusc_root.mkdir(parents=True, exist_ok=True)

        # Initialize tables
        self.table_names = [
            "category",
            "attribute",
            "visibility",
            "instance",
            "sensor",
            "calibrated_sensor",
            "ego_pose",
            "log",
            "scene",
            "sample",
            "sample_data",
            "sample_annotation",
            "map",
        ]
        self.tables = {name: [] for name in self.table_names}

        self.category_mapping = {}  # name -> token
        self.attribute_mapping = {}  # name -> token
        self.visibility_mapping = {}  # level -> token
        self.sensor_mapping = {}  # name -> token
        self.log_mapping = {}  # log_name -> token
        self.instance_mapping = {}  # instance_id -> token

        # Coordinate transform matrix (Rotation 90 deg around Z + axis)
        # Target: x=Towards, y=Left, z=Up
        self.coord_transform = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        self.s2n_category_mapping = {
            "Car": "vehicle.car",
            "Pedestrian": "human.pedestrian.adult",
            "Van": "vehicle.truck",
            "Bus": "vehicle.bus.rigid",
            "Truck": "vehicle.truck",
            "ScooterRider": "human.pedestrian.adult",
            "Scooter": "vehicle.motorcycle",
            "BicycleRider": "human.pedestrian.adult",
            "Bicycle": "vehicle.bicycle",
            "Motorcycle": "vehicle.motorcycle",
            "MotorcyleRider": "human.pedestrian.adult",
            "PoliceCar": "vehicle.emergency.police",
            "TourCar": "vehicle.car",
            "RoadWorker": "human.pedestrian.construction_worker",
            "Child": "human.pedestrian.child",
            "Cone": "movable_object.trafficcone",
            "FireHydrant": "ignore",  # What's this?
            "SaftyTriangle": "movable_object.trafficcone",  # Maybe
            "PlatformCart": "movable_object.pushable_pullable",
            "ConstructionCart": "movable_object.pushable_pullable",
            "RoadBarrel": "movable_object.barrier",
            "TrafficBarrier": "movable_object.barrier",
            "LongVehicle": "vehicle.truck",
            "BicycleGroup": "ignore",  # Maybe
            "ConcreteTruck": "vehicle.construction",
            "Tram": "ignore",  # Maybe
            "Excavator": "vehicle.construction",
            "Animal": "animal",
            "TrashCan": "movable_object.pushable_pullable",
            "ForkLift": "vehicle.construction",
            "Trimotorcycle": "vehicle.motorcycle",  # 3-wheeler
            "FreightTricycle": "vehicle.motorcycle",  # 3-wheeler
            "Crane": "vehicle.construction",
            "RoadRoller": "vehicle.construction",
            "Bulldozer": "vehicle.construction",
            "DontCare": "ignore",
            "Misc": "ignore",
        }

        # Camera mapping
        self.camera_mapping = {
            "front": "CAM_FRONT",
            "front_left": "CAM_FRONT_LEFT",
            "front_right": "CAM_FRONT_RIGHT",
            "rear": "CAM_BACK",
            "rear_left": "CAM_BACK_LEFT",
            "rear_right": "CAM_BACK_RIGHT",
        }

    def _initialize_static_tables(self):
        """Initialize Category, Attribute, Visibility, Sensor, Map tables."""
        # 1. Category Table
        mapped_categories = set()  # ensure no duplicate categories added
        for susc_cat_name, nusc_cat_name in self.s2n_category_mapping.items():
            if nusc_cat_name == "ignore":
                continue
            if nusc_cat_name in mapped_categories:
                continue
            mapped_categories.add(nusc_cat_name)
            token = generate_token(f"category_{nusc_cat_name}")
            self.tables["category"].append(
                {
                    "token": token,
                    "name": nusc_cat_name,
                    "description": f"Mapped from SUScape {susc_cat_name}",
                    "index": len(self.tables["category"]),
                }
            )
            self.category_mapping[nusc_cat_name] = token

        # 2. Attribute Table (Placeholder)
        attributes = [
            "vehicle.moving",
            "vehicle.stopped",
            "cycle.with_rider",
            "pedestrian.moving",
        ]
        for attr in attributes:
            token = generate_token(f"attribute_{attr}")
            self.tables["attribute"].append(
                {"token": token, "name": attr, "description": ""}
            )
            self.attribute_mapping[attr] = token

        # 3. Visibility Table
        visibilities = ["1", "2", "3", "4"]  # 0-40%, 40-60%, 60-80%, 80-100%
        for vis in visibilities:
            token = generate_token(f"visibility_{vis}")
            self.tables["visibility"].append(
                {"token": token, "level": vis, "description": f"Visibility level {vis}"}
            )
            self.visibility_mapping[vis] = token

        # 4. Sensor Table
        sensors = ["LIDAR_TOP"] + list(self.camera_mapping.values())
        for sensor in sensors:
            token = generate_token(f"sensor_{sensor}")
            self.tables["sensor"].append(
                {
                    "token": token,
                    "channel": sensor,
                    "modality": "lidar" if "LIDAR" in sensor else "camera",
                }
            )
            self.sensor_mapping[sensor] = token

        # 5. Map Table (Placeholder)
        # We don't have map data, but we can create a dummy entry
        self.tables["map"].append(
            {
                "token": generate_token("map_static_placeholder"),
                "log_tokens": [],  # Fill later
                "category": "semantic_prior",
                "filename": "",
            }
        )

    def convert(self):
        start_time = time.time()
        self._initialize_static_tables()

        if self.scenes is None:
            self.scenes = [d.name for d in self.susc_root.iterdir() if d.is_dir()]

        for scene_name in tqdm(self.scenes, desc="Converting Scenes"):
            self.process_scene(scene_name)

        all_log_tokens = [l["token"] for l in self.tables["log"]]  # noqa: E741
        self.tables["map"][0]["log_tokens"] = all_log_tokens

        self.save_tables()
        logger.info(f"Total conversion time: {time.time() - start_time:.2f}s")

    def process_scene(self, scene_name: str):
        scene_path = self.susc_root / scene_name
        if not scene_path.exists():
            logger.error(f"Scene {scene_name} not found, skipping.")
            return

        lidar_pose_dir = scene_path / "lidar_pose"
        frames = sorted([f.stem for f in lidar_pose_dir.glob("*.json")])

        if not frames:
            logger.error(f"No frames found for {scene_name}")
            return

        # Frame name is float timestamp e.g. "1630374714.000"
        try:
            ts_float = float(frames[0])
            dt = datetime.datetime.fromtimestamp(ts_float)
            date_captured = dt.strftime("%Y-%m-%d")
        except:  # noqa: E722
            date_captured = "2023-01-01"  # Fallback

        # 1. Create Log Entry
        log_token = generate_token(f"log_{scene_name}")
        self.tables["log"].append(
            {
                "token": log_token,
                "logfile": scene_name,
                "vehicle": "byd001",
                "date_captured": date_captured,
                "location": "china-shenzhen",
            }
        )
        self.log_mapping[scene_name] = log_token

        # 2. Setup Calibrated Sensors
        calib_sensors = self._process_calibration(scene_path, scene_name)

        # 3. Process Frames
        scene_token = generate_token(f"scene_{scene_name}")

        sample_tokens = [generate_token(f"sample_{scene_name}_{f}") for f in frames]

        gps_points = []
        for idx, frame_name in enumerate(frames):
            sample_token = sample_tokens[idx]
            prev_token = sample_tokens[idx - 1] if idx > 0 else ""
            next_token = sample_tokens[idx + 1] if idx < len(frames) - 1 else ""

            timestamp = int(float(frame_name) * 1e6)

            sample_record = {
                "token": sample_token,
                "timestamp": timestamp,
                "prev": prev_token,
                "next": next_token,
                "scene_token": scene_token,
                # "data": {},
                # "anns": [],
            }
            self.tables["sample"].append(sample_record)

            ego_pose_token, ego_trans, ego_rot, lat, lng = self._process_ego_pose(
                scene_path,
                frame_name,
                timestamp,
                scene_name,
            )

            if lat is not None and lng is not None:
                gps_points.append((lat, lng))

            self._process_sample_data(
                scene_path,
                frame_name,
                sample_token,
                ego_pose_token,
                calib_sensors,
                timestamp,
                scene_name,
                sample_record,
            )

            self._process_annotations(
                scene_path,
                frame_name,
                sample_token,
                scene_name,
                sample_record,
                ego_trans,
                ego_rot,
            )

        # Fetch Map after processing all frames (trajectory coverage)
        if self.fetch_map and gps_points:
            lats = [p[0] for p in gps_points]
            lngs = [p[1] for p in gps_points]
            north = max(lats)
            south = min(lats)
            east = max(lngs)
            west = min(lngs)
            self.fetch_osm_map(scene_name, north, south, east, west)
        elif self.fetch_map:
            logger.error(f"Skipping OSM fetch: No GPS data collected for {scene_name}")

        self.tables["scene"].append(
            {
                "token": scene_token,
                "log_token": log_token,
                "nbr_samples": len(frames),
                "first_sample_token": sample_tokens[0],
                "last_sample_token": sample_tokens[-1],
                "name": scene_name,
                "description": f"Converted from SUScape {scene_name}",
            }
        )

    def _process_calibration(self, scene_path: Path, scene_name: str) -> dict[str, str]:
        """
        Reads calibration from `calib/camera/{cam}.json`.
        Use LiDAR frame as vehicle frame.
        """
        calib_tokens = {}

        # Lidar = Vehicle
        lidar_token = generate_token(f"calib_{scene_name}_LIDAR_TOP")
        self.tables["calibrated_sensor"].append(
            {
                "token": lidar_token,
                "sensor_token": self.sensor_mapping["LIDAR_TOP"],
                "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],  # w, x, y, z
                "camera_intrinsic": [],
            }
        )
        calib_tokens["LIDAR_TOP"] = lidar_token

        # Cameras = Lidar -> Camera
        cam_calib_dir = scene_path / "calib" / "camera"
        for susc_cam, nusc_cam in self.camera_mapping.items():
            calib_file = cam_calib_dir / f"{susc_cam}.json"
            if not calib_file.exists():
                logger.warning(f"Missing calib for {susc_cam} in {scene_name}")
                continue

            with open(calib_file) as f:
                data = CameraCalibration.model_validate_json(f.read())

            # Inverse T_cl (Camera->LiDAR) to get T_sv (Vehicle->Sensor/Camera)
            extrinsic_flat = data.extrinsic
            intrinsic = data.intrinsic

            T_cl = np.array(extrinsic_flat).reshape(4, 4)

            T_sv = np.linalg.inv(T_cl)

            trans = T_sv[:3, 3]
            rot_mat = T_sv[:3, :3]

            trans = self.coord_transform @ trans
            rot_mat = self.coord_transform @ rot_mat

            quat = Quaternion(matrix=rot_mat)

            token = generate_token(f"calib_{scene_name}_{nusc_cam}")
            self.tables["calibrated_sensor"].append(
                {
                    "token": token,
                    "sensor_token": self.sensor_mapping[nusc_cam],
                    "translation": trans.tolist(),
                    "rotation": quat.elements.tolist(),  # w, x, y, z
                    "camera_intrinsic": [intrinsic[:3], intrinsic[3:6], intrinsic[6:]],
                }
            )
            calib_tokens[nusc_cam] = token

        return calib_tokens

    def _process_ego_pose(
        self, scene_path: Path, frame_name: str, timestamp: int, scene_name: str
    ) -> tuple[str, np.ndarray, Quaternion, float | None, float | None]:
        """
        Read lidar pose as ego pose from `lidar_pose/{frame_name}.json`.
        Read GPS infos from `ego_pose/{frame_name}.json`.
        """
        lidar_pose_file = scene_path / "lidar_pose" / f"{frame_name}.json"
        with open(lidar_pose_file) as f:
            lidar_data = LidarPose.model_validate_json(f.read())

        pose_mat = np.array(lidar_data.lidarPose).reshape(4, 4)

        trans_transformed = pose_mat[:3, 3]
        rot_transformed = pose_mat[:3, :3]

        trans_original = trans_transformed

        rot_mat = rot_transformed @ self.coord_transform.T

        quat = Quaternion(matrix=rot_mat)

        token = generate_token(f"ego_pose_{scene_name}_{timestamp}")
        self.tables["ego_pose"].append(
            {
                "token": token,
                "timestamp": timestamp,
                "rotation": quat.elements.tolist(),
                "translation": trans_original.tolist(),
            }
        )

        # Read SUScape GPS Info from so called ego_pose/ if exists
        lat, lng = None, None
        ego_pose_file = scene_path / "ego_pose" / f"{frame_name}.json"
        if ego_pose_file.exists():
            try:
                with open(ego_pose_file) as f:
                    ego_data = EgoPose.model_validate_json(f.read())
                lat, lng = ego_data.lat, ego_data.lng
            except Exception as e:
                logger.warning(f"Failed to parse ego_pose for {frame_name}: {e}")

        # Update the record with lat/lng if available
        if lat is not None and lng is not None:
            self.tables["ego_pose"][-1]["lat"] = lat
            self.tables["ego_pose"][-1]["lng"] = lng

        return token, trans_original, quat, lat, lng

    def _process_sample_data(
        self,
        scene_path,
        frame_name,
        sample_token,
        ego_pose_token,
        calib_sensors,
        timestamp,
        scene_name,
        sample_record,
    ):
        # 1. Lidar
        lidar_src = scene_path / "lidar" / f"{frame_name}.pcd"
        if lidar_src.exists():
            lidar_token = generate_token(
                f"sample_data_{scene_name}_{timestamp}_LIDAR_TOP"
            )
            lidar_filename = f"{scene_name}__LIDAR_TOP__{timestamp}.pcd.bin"
            rel_path = f"sweeps/LIDAR_TOP/{lidar_filename}"

            pc = PointCloud.from_path(lidar_src)
            points = pc.pc_data.copy()
            xyz = np.stack([points["x"], points["y"], points["z"]], axis=1)

            xyz_original = (self.coord_transform @ xyz.T).T

            if "intensity" in pc.fields:
                intensity = points["intensity"]
            else:
                logger.warning(
                    f"No intensity in point cloud {lidar_src}, filling with zeros."
                )
                intensity = np.zeros(len(xyz))

            data_bin = np.zeros((len(xyz), 5), dtype=np.float32)
            data_bin[:, 0] = xyz_original[:, 0]
            data_bin[:, 1] = xyz_original[:, 1]
            data_bin[:, 2] = xyz_original[:, 2]
            data_bin[:, 3] = intensity

            dest_path = self.output_root / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            data_bin.tofile(dest_path)

            self.tables["sample_data"].append(
                {
                    "token": lidar_token,
                    "sample_token": sample_token,
                    "ego_pose_token": ego_pose_token,
                    "calibrated_sensor_token": calib_sensors["LIDAR_TOP"],
                    "timestamp": timestamp,
                    "fileformat": "pcd",
                    "is_key_frame": True,  # TODO how to pick key frame?
                    "height": 0,  # Lidar has no height
                    "width": 0,  # Lidar has no width
                    "filename": rel_path,
                    "prev": "",  # will be filled later
                    "next": "",  # will be filled later
                }
            )

            # sample_record["data"]["LIDAR_TOP"] = lidar_token

        # 2. Cameras
        for susc_cam, nusc_cam in self.camera_mapping.items():
            cam_src = scene_path / "camera" / susc_cam / f"{frame_name}.jpg"
            if cam_src.exists():
                cam_token = generate_token(
                    f"sample_data_{scene_name}_{timestamp}_{nusc_cam}"
                )
                cam_filename = f"{scene_name}__{nusc_cam}__{timestamp}.jpg"
                rel_path = f"sweeps/{nusc_cam}/{cam_filename}"

                dest_path = self.output_root / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cam_src, dest_path)

                with Image.open(cam_src) as img:
                    w, h = img.size

                self.tables["sample_data"].append(
                    {
                        "token": cam_token,
                        "sample_token": sample_token,
                        "ego_pose_token": ego_pose_token,
                        "calibrated_sensor_token": calib_sensors[nusc_cam],
                        "timestamp": timestamp,
                        "fileformat": "jpg",
                        "is_key_frame": True,  # TODO how to pick a key frame?
                        "height": h,
                        "width": w,
                        "filename": rel_path,
                        "prev": "",  # will be filled later
                        "next": "",  # will be filled later
                    }
                )

                # sample_record["data"][nusc_cam] = cam_token

    def _process_annotations(
        self,
        scene_path,
        frame_name,
        sample_token,
        scene_name,
        sample_record,
        ego_trans,
        ego_rot,
    ):
        label_file = scene_path / "label" / f"{frame_name}.json"
        if not label_file.exists():
            return

        with open(label_file) as f:
            data = parse_label(f.read())

        timestamp = int(float(frame_name) * 1e6)

        for obj in data.objs:
            # Map Category
            cat_name = self.s2n_category_mapping.get(obj.obj_type, "ignore")
            if cat_name == "ignore" or cat_name not in self.category_mapping:
                continue

            inst_susc_id = obj.obj_id
            inst_key = f"{scene_name}_{inst_susc_id}"

            if inst_key not in self.instance_mapping:
                inst_token = generate_token(f"instance_{inst_key}")
                self.instance_mapping[inst_key] = inst_token
                self.tables["instance"].append(
                    {
                        "token": inst_token,
                        "category_token": self.category_mapping[cat_name],
                        "nbr_annotations": 0,  # TODO calculate nbr_annotations
                        # TODO first/last annotation tokens should be filled later
                        "first_annotation_token": "",
                        "last_annotation_token": "",
                    }
                )
            else:
                inst_token = self.instance_mapping[inst_key]

            # 1. Transform Local Position/Rotation to aligned Local Frame
            pos = np.array([obj.psr.position.x, obj.psr.position.y, obj.psr.position.z])
            pos_local_aligned = self.coord_transform @ pos

            size_orig = [obj.psr.scale.y, obj.psr.scale.x, obj.psr.scale.z]

            r_susc = R.from_euler(
                "xyz",
                [obj.psr.rotation.x, obj.psr.rotation.y, obj.psr.rotation.z],
                degrees=False,
            )
            mat_susc = r_susc.as_matrix()
            mat_local_aligned = self.coord_transform @ mat_susc

            # 2. Transform Aligned Local to Global using Ego Pose
            # P_global = R_ego * P_local + T_ego
            # R_global = R_ego * R_local

            pos_global = ego_rot.rotate(pos_local_aligned) + ego_trans

            rot_local_quat = Quaternion(matrix=mat_local_aligned)
            rot_global_quat = ego_rot * rot_local_quat

            ann_token = generate_token(
                f"annotation_{scene_name}_{timestamp}_{inst_susc_id}"
            )
            # TODO Visibility, Attributes are not labeled in SUSC
            self.tables["sample_annotation"].append(
                {
                    "token": ann_token,
                    "sample_token": sample_token,
                    "instance_token": inst_token,
                    "visibility_token": self.visibility_mapping["4"],
                    "attribute_tokens": [],
                    "translation": pos_global.tolist(),
                    "size": size_orig,
                    "rotation": rot_global_quat.elements.tolist(),
                    "prev": "",  # will be filled later
                    "next": "",  # will be filled later
                    "num_lidar_pts": 0,  # TODO calculate num_lidar_pts
                    "num_radar_pts": 0,  # TODO calculate num_radar_pts
                }
            )

            # sample_record["anns"].append(ann_token)

    def save_tables(self):
        self._link_sample_data()
        self._link_annotations()

        for name, records in self.tables.items():
            out_path = self.nusc_root / f"{name}.json"
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2)

    def _link_sample_data(self):
        """
        Link sample data records to each other based on calibrated sensor token.
        """
        data_by_sensor = defaultdict(list)
        for rec in self.tables["sample_data"]:
            data_by_sensor[rec["calibrated_sensor_token"]].append(rec)

        for _calib_token, recs in data_by_sensor.items():
            recs.sort(key=lambda x: x["timestamp"])
            for i in range(len(recs)):
                if i > 0:
                    recs[i]["prev"] = recs[i - 1]["token"]
                if i < len(recs) - 1:
                    recs[i]["next"] = recs[i + 1]["token"]

    def _link_annotations(self):
        """
        Link annotations records to each other based on instance token.
        Also populate instance statistics.
        """
        instance_lookup = {rec["token"]: rec for rec in self.tables["instance"]}

        ann_by_inst = defaultdict(list)
        for rec in self.tables["sample_annotation"]:
            ann_by_inst[rec["instance_token"]].append(rec)

        # get sample timestamp from sample table
        sample_time = {s["token"]: s["timestamp"] for s in self.tables["sample"]}

        for inst_token, recs in ann_by_inst.items():
            recs.sort(key=lambda x: sample_time.get(x["sample_token"], 0))

            for i in range(len(recs)):
                if i > 0:
                    recs[i]["prev"] = recs[i - 1]["token"]
                if i < len(recs) - 1:
                    recs[i]["next"] = recs[i + 1]["token"]

            if inst_token in instance_lookup:
                inst_rec = instance_lookup[inst_token]
                inst_rec["nbr_annotations"] = len(recs)
                inst_rec["first_annotation_token"] = recs[0]["token"]
                inst_rec["last_annotation_token"] = recs[-1]["token"]

    def fetch_osm_map(
        self, scene_name: str, north: float, south: float, east: float, west: float
    ) -> None:
        """
        Fetch OSM map for the given bounding box and save it to the maps folder.
        """
        try:
            import osmnx as ox
        except ImportError as e:
            logger.error("osmnx is required for usage with --fetch-map")
            raise e

        map_dir = self.output_root / "maps"
        map_dir.mkdir(parents=True, exist_ok=True)

        map_filename = f"{scene_name}.osm"
        map_path = map_dir / map_filename

        if map_path.exists():
            logger.info(f"Map already exists: {map_path}")
            return

        try:
            logger.info(
                f"Fetching OSM map for {scene_name} bbox: N{north}, S{south}, E{east}, W{west}..."  # noqa: E501
            )
            # Add a small buffer to the bounding box (e.g., 0.001 degrees ~ 100m)
            buffer = 0.001
            G = ox.graph_from_bbox(
                bbox=(north + buffer, south - buffer, east + buffer, west - buffer),
                network_type="drive",
                simplify=False,
            )

            # Save to .osm (XML) format
            ox.settings.all_oneway = True
            ox.save_graph_xml(G, filepath=map_path)
            # TODO maybe convert osm map to semantic map to match nuscenes? but how?
            logger.info(f"Saved OSM map to {map_path}")

        except Exception as e:
            logger.error(f"Failed to fetch OSM map for {scene_name}: {e}")


def main() -> None:
    tyro.cli(Susc2NuscConverter).convert()


if __name__ == "__main__":
    main()
