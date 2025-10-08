"""
Evaluation module for LightEMMA predictions
"""
import os
import glob
import numpy as np
from typing import Dict, List, Tuple, Optional

from core.config import ConfigManager
from utils import load_json_file, extract_driving_action, integrate_driving_commands, compute_metrics, OverlayTrajectory


class Evaluator:
    """Handles evaluation of prediction results"""

    def __init__(self, config: ConfigManager):
        self.config = config

    def evaluate_single_model(self, results_dir: str, error_handling: bool = False,
                            visualize: bool = False) -> Dict:
        """Evaluate a single model"""
        scene_files = glob.glob(os.path.join(results_dir, "*.json"))

        if not scene_files:
            raise ValueError(f"No scene files found in {results_dir}")

        return self._evaluate_scenes(scene_files, error_handling, visualize, results_dir)

    def evaluate_multiple_models(self, results_base_dir: str) -> Dict:
        """Evaluate multiple models excluding error frames"""
        model_dirs = [d for d in os.listdir(results_base_dir)
                     if os.path.isdir(os.path.join(results_base_dir, d))]

        if not model_dirs:
            raise ValueError(f"No model directories found in {results_base_dir}")

        # Collect all error frames across models
        error_set = set()
        all_results = {}

        for model_dir in model_dirs:
            model_path = os.path.join(results_base_dir, model_dir, "output")
            if not os.path.exists(model_path):
                continue

            scene_files = glob.glob(os.path.join(model_path, "*.json"))
            model_error_set, _ = self._collect_errors(scene_files, model_dir)
            error_set.update(model_error_set)

            # Evaluate this model excluding error frames
            results = self._evaluate_scenes(scene_files, error_handling=True,
                                          visualize=False, error_set=error_set)
            all_results[model_dir] = results

        return all_results

    def _collect_errors(self, scene_files: List[str], model_name: str) -> Tuple[set, List]:
        """Collect error frames for a model"""
        error_set = set()
        error_entries = []

        for scene_file in scene_files:
            scene_data = load_json_file(scene_file)
            scene_name = scene_data["scene_info"]["name"]

            for frame in scene_data["frames"]:
                frame_index = frame["frame_index"]
                pred_actions_str = frame["inference"]["pred_actions_str"]
                pred_actions = extract_driving_action(pred_actions_str, error_handling=True)

                if not pred_actions:
                    error_set.add((scene_name, frame_index))
                    error_entries.append((model_name, scene_name, frame_index, pred_actions_str))

        return error_set, error_entries

    def _evaluate_scenes(self, scene_files: List[str], error_handling: bool = False,
                        visualize: bool = False, viz_dir: Optional[str] = None,
                        error_set: Optional[set] = None) -> Dict:
        """Evaluate multiple scenes"""
        total_frames = 0
        successful_frames = 0
        parse_error_frames = 0

        ade_1s, ade_2s, ade_3s, ade_avg, fde, miss_rate = [], [], [], [], [], []

        total_token_usage = {
            "scene_prompt": {"input": 0, "output": 0},
            "intent_prompt": {"input": 0, "output": 0},
            "waypoint_prompt": {"input": 0, "output": 0},
            "total": {"input": 0, "output": 0},
        }
        total_time_usage = {
            "scene_prompt": 0,
            "intent_prompt": 0,
            "waypoint_prompt": 0,
            "total": 0,
        }

        error_entries = []

        for scene_file in scene_files:
            scene_data = load_json_file(scene_file)
            scene_name = scene_data["scene_info"]["name"]

            for frame in scene_data["frames"]:
                total_frames += 1
                frame_index = frame["frame_index"]

                # Skip error frames if error_set is provided
                if error_set and (scene_name, frame_index) in error_set:
                    continue

                gt_positions = frame["ego_info"]["gt_positions"]
                pred_actions_str = frame["inference"]["pred_actions_str"]
                pred_actions = extract_driving_action(pred_actions_str, error_handling)

                if not pred_actions:
                    parse_error_frames += 1
                    error_entries.append((scene_name, frame_index, pred_actions_str))
                    continue

                pred_trajectory = integrate_driving_commands(pred_actions, dt=0.5)
                metrics = compute_metrics(pred_trajectory, gt_positions)
                successful_frames += 1

                # Collect metrics
                if metrics["ADE_1s"] is not None: ade_1s.append(metrics["ADE_1s"])
                if metrics["ADE_2s"] is not None: ade_2s.append(metrics["ADE_2s"])
                if metrics["ADE_3s"] is not None: ade_3s.append(metrics["ADE_3s"])
                if metrics["ADE_avg"] is not None: ade_avg.append(metrics["ADE_avg"])
                if metrics["FDE"] is not None: fde.append(metrics["FDE"])
                if metrics["missRate_2"] is not None: miss_rate.append(metrics["missRate_2"])

                # Collect token and time usage
                for k, v in frame["token_usage"].items():
                    total_token_usage[k]["input"] += v["input"]
                    total_token_usage[k]["output"] += v["output"]
                    total_token_usage["total"]["input"] += v["input"]
                    total_token_usage["total"]["output"] += v["output"]

                for k, v in frame["time_usage"].items():
                    total_time_usage[k] += v
                    total_time_usage["total"] += v

                # Generate visualizations if requested
                if visualize and viz_dir:
                    self._generate_visualization(frame, scene_name, frame_index,
                                               gt_positions, pred_trajectory, viz_dir)

        # Calculate averages
        avg_token_usage = {
            k: {
                "input": v["input"] / successful_frames if successful_frames > 0 else 0,
                "output": v["output"] / successful_frames if successful_frames > 0 else 0,
            }
            for k, v in total_token_usage.items()
        }

        avg_time_usage = {
            k: v / successful_frames if successful_frames > 0 else 0
            for k, v in total_time_usage.items()
        }

        return {
            "summary": {
                "total_frames": total_frames,
                "successful_frames": successful_frames,
                "parse_error_frames": parse_error_frames,
                "success_rate": successful_frames / total_frames if total_frames > 0 else 0
            },
            "metrics": {
                "ADE_1s": np.mean(ade_1s).item() if ade_1s else None,
                "ADE_2s": np.mean(ade_2s).item() if ade_2s else None,
                "ADE_3s": np.mean(ade_3s).item() if ade_3s else None,
                "ADE_avg": np.mean(ade_avg).item() if ade_avg else None,
                "FDE": np.mean(fde).item() if fde else None,
                "missRate_2": np.mean(miss_rate).item() if miss_rate else None,
            },
            "usage": {
                "token_usage": avg_token_usage,
                "time_usage": avg_time_usage
            },
            "errors": error_entries
        }

    def _generate_visualization(self, frame: Dict, scene_name: str, frame_index: int,
                              gt_positions: List, pred_trajectory: List, viz_dir: str):
        """Generate visualization for a frame"""
        data_config = self.config.data_config
        image_path = os.path.join(data_config["root"], "samples/CAM_FRONT", frame["image_name"])

        if os.path.exists(image_path):
            camera_params = frame["camera_params"]
            viz_filename = f"{scene_name}_frame{frame_index}.png"
            viz_path = os.path.join(viz_dir, viz_filename)

            OverlayTrajectory(
                img_path=image_path,
                wp_world1=gt_positions,
                wp_world2=pred_trajectory,
                cam_to_ego=camera_params,
                ego_pos=(0, 0),
                ego_heading=0.0,
                save_path=viz_path,
            )