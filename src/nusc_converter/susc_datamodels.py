from pydantic import AliasChoices, BaseModel, Field, TypeAdapter, field_validator


class LidarPose(BaseModel):
    lidarPose: list[float] = Field(min_length=16, max_length=16)


class EgoPose(BaseModel):
    lat: float
    lng: float
    # north_vel, east_vel, up_vel, roll, pitch, azimuth, x, y, z, etc.


class CameraCalibration(BaseModel):
    extrinsic: list[float] = Field(
        validation_alias=AliasChoices("extrinsic", "lidar_to_camera")
    )
    intrinsic: list[float]

    @field_validator("extrinsic")
    @classmethod
    def check_extrinsic(cls, v):
        if not v:
            raise ValueError("one of extrinsic / lidar_to_camera must be provided")
        return v


class XYZ(BaseModel):
    x: float
    y: float
    z: float


class PSR(BaseModel):
    position: XYZ
    rotation: XYZ
    scale: XYZ


class LabelObject(BaseModel):
    obj_id: str
    obj_type: str
    psr: PSR


class Label(BaseModel):
    frame: str | None = None  # some labels have and some don't
    objs: list[LabelObject]


Label_v2 = TypeAdapter(list[LabelObject])


def parse_label(json_content: str | bytes) -> Label:
    """
    Different formats of label files exist.
    v1: {"frame": "...", "objs": [...]}, the frame may be missing
    v2: [...]  (list of objects directly)
    """
    try:
        return Label.model_validate_json(json_content)
    except Exception:
        try:
            objs = Label_v2.validate_json(json_content)
            return Label(frame=None, objs=objs)
        except Exception as e:
            raise ValueError(f"Failed to parse label: {e}") from e


if __name__ == "__main__":
    from pathlib import Path

    scene_root = Path("data/suscape_scenes")

    for scene_dir in scene_root.iterdir():
        if not scene_dir.is_dir():
            continue
        label_dir = scene_dir / "label"
        ego_pose_dir = scene_dir / "ego_pose"
        lidar_pose_dir = scene_dir / "lidar_pose"

        it = label_dir.iterdir()

        label_file, ego_pose_file, lidar_pose_file = None, None, None
        while True:
            label_file = it.__next__()
            ego_pose_file = scene_dir / "ego_pose" / label_file.name
            lidar_pose_file = scene_dir / "lidar_pose" / label_file.name
            if (
                label_file.exists()
                and ego_pose_file.exists()
                and lidar_pose_file.exists()
            ):
                break

        assert (
            label_file is not None
            and ego_pose_file is not None
            and lidar_pose_file is not None
        )
        try:
            timestamp = float(label_file.stem)
            timestamp_str = label_file.stem
        except ValueError:
            continue

        label = parse_label(label_file.read_text())
        ego_pose = EgoPose.model_validate_json(ego_pose_file.read_text())

        lidar_pose = LidarPose.model_validate_json(lidar_pose_file.read_text())
        print(f"Scene: {scene_dir.name}, Valided Timestamp: {timestamp}")
