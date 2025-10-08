"""
Data loading and processing for nuScenes dataset
"""
import os
from typing import List, Dict, Tuple, Optional
from nuscenes import NuScenes

from core.config import ConfigManager
from utils import quaternion_to_yaw


class SceneData:
    """Container for scene data"""

    def __init__(self, scene_token: str, scene_name: str, description: str,
                 first_sample_token: str, last_sample_token: str):
        self.scene_token = scene_token
        self.scene_name = scene_name
        self.description = description
        self.first_sample_token = first_sample_token
        self.last_sample_token = last_sample_token
        self.frames = []

    def add_frame(self, frame_data: Dict):
        """Add frame data to scene"""
        self.frames.append(frame_data)

    def to_dict(self) -> Dict:
        """Convert scene data to dictionary"""
        return {
            "scene_info": {
                "name": self.scene_name,
                "description": self.description,
                "first_sample_token": self.first_sample_token,
                "last_sample_token": self.last_sample_token
            },
            "frames": self.frames,
            "metadata": {
                "total_frames": len(self.frames)
            }
        }


class FrameData:
    """Container for frame data"""

    def __init__(self, frame_index: int, sample_token: str, image_path: str,
                 timestamp: int, camera_params: Dict, ego_info: Dict):
        self.frame_index = frame_index
        self.sample_token = sample_token
        self.image_path = image_path
        self.timestamp = timestamp
        self.camera_params = camera_params
        self.ego_info = ego_info
        self.inference = {}
        self.token_usage = {}
        self.time_usage = {}

    def set_inference_data(self, scene_description: str, driving_intent: str,
                          pred_actions_str: str, prompts: Dict, tokens: Dict, times: Dict):
        """Set inference results"""
        self.inference = {
            "scene_description": scene_description,
            "driving_intent": driving_intent,
            "pred_actions_str": pred_actions_str,
            "scene_prompt": prompts.get("scene", ""),
            "intent_prompt": prompts.get("intent", ""),
            "waypoint_prompt": prompts.get("waypoint", "")
        }
        self.token_usage = tokens
        self.time_usage = times

    def to_dict(self) -> Dict:
        """Convert frame data to dictionary"""
        return {
            "frame_index": self.frame_index,
            "sample_token": self.sample_token,
            "image_name": os.path.basename(self.image_path),
            "timestamp": self.timestamp,
            "camera_params": self.camera_params,
            "ego_info": self.ego_info,
            "inference": self.inference,
            "token_usage": self.token_usage,
            "time_usage": self.time_usage
        }


class NuScenesDataLoader:
    """Handles loading and processing of nuScenes data"""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.nusc = None

    def initialize_dataset(self):
        """Initialize nuScenes dataset"""
        data_config = self.config.data_config
        self.nusc = NuScenes(
            version=data_config["version"],
            dataroot=data_config["root"],
            verbose=True
        )

    def get_scenes_to_process(self, scene_name: Optional[str] = None) -> List[SceneData]:
        """Get list of scenes to process"""
        if scene_name:
            # Find specific scene
            scenes = [scene for scene in self.nusc.scene if scene["name"] == scene_name]
            if not scenes:
                raise ValueError(f"Scene '{scene_name}' not found in dataset")
        else:
            # Get all scenes
            scenes = self.nusc.scene

        scene_data_list = []
        for scene in scenes:
            scene_data = SceneData(
                scene_token=scene["token"],
                scene_name=scene["name"],
                description=scene["description"],
                first_sample_token=scene["first_sample_token"],
                last_sample_token=scene["last_sample_token"]
            )
            scene_data_list.append(scene_data)

        return scene_data_list

    def load_scene_frames(self, scene_data: SceneData) -> SceneData:
        """Load all frames for a scene"""
        camera_params = []
        front_camera_images = []
        ego_positions = []
        ego_headings = []
        timestamps = []
        sample_tokens = []

        curr_sample_token = scene_data.first_sample_token

        # Retrieve all frames in the scene
        while curr_sample_token:
            sample = self.nusc.get("sample", curr_sample_token)
            sample_tokens.append(curr_sample_token)

            cam_front_data = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            front_camera_images.append(
                os.path.join(self.nusc.dataroot, cam_front_data["filename"])
            )

            # Get the camera parameters
            camera_params.append(
                self.nusc.get("calibrated_sensor", cam_front_data["calibrated_sensor_token"])
            )

            # Get ego vehicle state
            ego_state = self.nusc.get("ego_pose", cam_front_data["ego_pose_token"])
            ego_positions.append(tuple(ego_state["translation"][0:2]))
            ego_headings.append(quaternion_to_yaw(ego_state["rotation"]))
            timestamps.append(ego_state["timestamp"])

            # Move to next sample or exit loop if at the end
            curr_sample_token = (
                sample["next"] if curr_sample_token != scene_data.last_sample_token else None
            )

        # Store frame data in scene
        scene_data.camera_params = camera_params
        scene_data.front_camera_images = front_camera_images
        scene_data.ego_positions = ego_positions
        scene_data.ego_headings = ego_headings
        scene_data.timestamps = timestamps
        scene_data.sample_tokens = sample_tokens

        return scene_data

    def prepare_frame_data(self, scene_data: SceneData, frame_idx: int,
                          obs_len: int, fut_len: int) -> FrameData:
        """Prepare data for a specific frame"""
        cur_index = frame_idx + obs_len + 1

        image_path = scene_data.front_camera_images[cur_index]
        sample_token = scene_data.sample_tokens[cur_index]
        camera_param = scene_data.camera_params[cur_index]

        # Get current position and heading
        cur_pos = scene_data.ego_positions[cur_index]
        cur_heading = scene_data.ego_headings[cur_index]

        # Get observation data (past positions and timestamps)
        obs_pos = scene_data.ego_positions[cur_index - obs_len - 1 : cur_index + 1]
        obs_time = scene_data.timestamps[cur_index - obs_len - 1 : cur_index + 1]

        # Get future positions (ground truth)
        fut_pos = scene_data.ego_positions[cur_index - 1 : cur_index + fut_len + 1]
        # Remove extra indices used for speed and curvature calculation
        fut_pos = fut_pos[2:] if len(fut_pos) > fut_len else fut_pos

        # Transform to ego frame
        from utils import global_to_ego_frame
        obs_pos = global_to_ego_frame(cur_pos, cur_heading, obs_pos)
        fut_pos = global_to_ego_frame(cur_pos, cur_heading, fut_pos)

        frame_data = FrameData(
            frame_index=frame_idx,
            sample_token=sample_token,
            image_path=image_path,
            timestamp=scene_data.timestamps[cur_index],
            camera_params={
                "rotation": camera_param["rotation"],
                "translation": camera_param["translation"],
                "camera_intrinsic": camera_param["camera_intrinsic"]
            },
            ego_info={
                "position": cur_pos,
                "heading": cur_heading,
                "obs_positions": obs_pos,
                "gt_positions": fut_pos,
            }
        )

        return frame_data