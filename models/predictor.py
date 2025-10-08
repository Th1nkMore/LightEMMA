"""
Prediction engine for LightEMMA
"""
import time
from typing import Dict, List, Tuple, Optional

from core.config import ConfigManager
from data.loader import FrameData
from vlm import ModelHandler
from utils import compute_speed, compute_curvature, format_long_text


class PromptGenerator:
    """Generates prompts for different stages of prediction"""

    @staticmethod
    def generate_scene_description_prompt() -> str:
        """Generate scene description prompt"""
        return (
            "You are an autonomous driving labeller. "
            "You have access to the front-view camera image. "
            "You must observe and analyze the movements of vehicles and pedestrians, "
            "lane markings, traffic lights, and any relevant objects in the scene. "
            "describe what you observe, but do not infer the ego's action. "
            "generate your response in plain text in one paragraph without any formating. "
        )

    @staticmethod
    def generate_driving_intent_prompt(scene_description: str, prev_speed: List[float],
                                      prev_curvatures: List[float]) -> str:
        """Generate driving intent analysis prompt"""
        return (
            "You are an autonomous driving labeller. "
            "You have access to the front-view camera image. "
            "The scene is described as follows: "
            f"{scene_description} "
            "The ego vehicle's speed for the past 3 seconds with 0.5 sec resolution is"
            f"{prev_speed} m/s (last index is the most recent) "
            "The ego vehicle's curvature for the past 3 seconds with 0.5 sec resolution is"
            f"{prev_curvatures} (last index is the most recent) "
            "A positive curvature indicates the ego is turning left."
            "A negative curvature indicates the ego is turning right. "
            "What was the ego's previous intent? "
            "Was it accelerating (by how much), decelerating (by how much), or maintaining speed? "
            "Was it turning left (by how much), turning right (by how much), or following the lane? "
            "Taking into account the ego's previous intent, how should it drive in the next 3 seconds? "
            "Should the ego accelerate (by how much), decelerate (by how much), or maintain speed? "
            "Should the ego turn left (by how much), turn right (by how much), or follow the lane?  "
            "Generate your response in plain text in one paragraph without any formating. "
        )

    @staticmethod
    def generate_trajectory_prediction_prompt(scene_description: str, driving_intent: str,
                                           prev_speed: List[float], prev_curvatures: List[float]) -> str:
        """Generate trajectory prediction prompt"""
        return (
            "You are an autonomous driving labeller. "
            "You have access to the front-view camera image. "
            "The scene is described as follows: "
            f"{scene_description} "
            "The ego vehicle's speed for the past 3 seconds with 0.5 sec resolution is"
            f"{prev_speed} m/s (last index is the most recent) "
            "The ego vehicle's curvature for the past 3 seconds with 0.5 sec resolution is"
            f"{prev_curvatures} (last index is the most recent) "
            "A positive curvature indicates the ego is turning left."
            "A negative curvature indicates the ego is turning right. "
            "The high-level driving instructions are as follows: "
            f"{driving_intent} "
            "Predict the speed and curvature for the next 6 waypoints, with 0.5-second resolution. "
            "The predicted speed and curvature changes must obey the physical constraints of the vehicle. "
            "Predict Exactly 6 pairs of speed and curvature, in the format:"
            "[(v1, c1), (v2, c2), (v3, c3), (v4, c4), (v5, c5), (v6, c6)]. "
            "ONLY return the answers in the required format, do not include punctuation or text."
        )


class TrajectoryPredictor:
    """Handles trajectory prediction using VLM"""

    def __init__(self, config: ConfigManager, model_name: str):
        self.config = config
        self.model_name = model_name
        self.model_handler = None
        self.prompt_generator = PromptGenerator()

    def initialize_model(self):
        """Initialize the VLM model"""
        self.model_handler = ModelHandler(self.model_name, self.config.config)
        self.model_handler.initialize_model()
        print(f"Initialized model: {self.model_name}")

    def predict_frame(self, frame_data: FrameData) -> FrameData:
        """Predict trajectory for a single frame"""
        try:
            # Calculate historical speed and curvature
            obs_positions = frame_data.ego_info["obs_positions"]
            obs_timestamps = []  # This would need to be passed or calculated

            # For now, we'll assume timestamps are available in frame_data
            # This is a simplification - in practice, we'd need to pass timestamps
            prev_speed = [0.0] * len(obs_positions)  # Placeholder
            prev_curvatures = [0.0] * len(obs_positions)  # Placeholder

            # Generate prompts
            scene_prompt = self.prompt_generator.generate_scene_description_prompt()
            intent_prompt = self.prompt_generator.generate_driving_intent_prompt(
                "", prev_speed, prev_curvatures  # scene_description would be passed
            )
            trajectory_prompt = self.prompt_generator.generate_trajectory_prediction_prompt(
                "", "", prev_speed, prev_curvatures  # scene_description and intent would be passed
            )

            # Get VLM responses
            scene_description, scene_tokens, scene_time = self.model_handler.get_response(
                prompt=scene_prompt, image_path=frame_data.image_path
            )

            driving_intent, intent_tokens, intent_time = self.model_handler.get_response(
                prompt=intent_prompt, image_path=frame_data.image_path
            )

            pred_actions_str, trajectory_tokens, trajectory_time = self.model_handler.get_response(
                prompt=trajectory_prompt, image_path=frame_data.image_path
            )

            # Store results in frame data
            prompts = {
                "scene": format_long_text(scene_prompt),
                "intent": format_long_text(intent_prompt),
                "waypoint": format_long_text(trajectory_prompt)
            }

            tokens = {
                "scene_prompt": scene_tokens,
                "intent_prompt": intent_tokens,
                "waypoint_prompt": trajectory_tokens
            }

            times = {
                "scene_prompt": scene_time,
                "intent_prompt": intent_time,
                "waypoint_prompt": trajectory_time
            }

            frame_data.set_inference_data(
                scene_description=format_long_text(scene_description),
                driving_intent=format_long_text(driving_intent),
                pred_actions_str=pred_actions_str,
                prompts=prompts,
                tokens=tokens,
                times=times
            )

            return frame_data

        except Exception as e:
            print(f"Error predicting frame {frame_data.frame_index}: {e}")
            return frame_data


class PredictionPipeline:
    """Orchestrates the complete prediction pipeline"""

    def __init__(self, config: ConfigManager, model_name: str):
        self.config = config
        self.model_name = model_name
        self.data_loader = None
        self.predictor = TrajectoryPredictor(config, model_name)

    def set_data_loader(self, data_loader):
        """Set the data loader"""
        self.data_loader = data_loader

    def initialize(self):
        """Initialize all components"""
        self.predictor.initialize_model()

    def process_scene(self, scene_data, obs_len: int, fut_len: int) -> Dict:
        """Process a complete scene"""
        print(f"Processing scene '{scene_data.scene_name}': {scene_data.description}")

        # Load scene frames
        scene_data = self.data_loader.load_scene_frames(scene_data)

        num_frames = len(scene_data.front_camera_images)
        ttl_len = obs_len + fut_len + self.config.get("prediction.ext_len", 2)

        if num_frames < ttl_len:
            print(f"Skipping '{scene_data.scene_name}', insufficient frames ({num_frames} < {ttl_len}).")
            return scene_data.to_dict()

        # Process each frame
        for i in range(0, num_frames - ttl_len, 1):
            try:
                frame_data = self.data_loader.prepare_frame_data(scene_data, i, obs_len, fut_len)
                frame_data = self.predictor.predict_frame(frame_data)
                scene_data.add_frame(frame_data.to_dict())

            except Exception as e:
                print(f"Error processing frame {i} in {scene_data.scene_name}: {e}")
                continue

        return scene_data.to_dict()