#!/usr/bin/env python3
"""
LightEMMA Prediction Script - Modular Version
"""
import os
import argparse
import datetime

from core.config import ConfigManager
from data.loader import NuScenesDataLoader
from models.predictor import PredictionPipeline
from utils import save_dict_to_json


def parse_args():
    parser = argparse.ArgumentParser(description="LightEMMA: End-to-End Autonomous Driving")
    parser.add_argument("--model", type=str, default="chatgpt-4o-latest",
                        help="Options: gpt-series, claude-series, gemini-series, "
                        "qwen2.5-7b, qwen2.5-72b, llama-3.2-11b, llama-3.2-90b")
    parser.add_argument("--continue_dir", type=str, default=None,
                        help="Path to the directory with previously processed scene JSON files to resume processing")
    parser.add_argument("--scene", type=str, default=None,
                        help="Optional: Specific scene name to process.")
    parser.add_argument("--config", type=str, default="MyConfig.yaml",
                        help="Path to configuration file")
    return parser.parse_args()


def setup_output_directory(config: ConfigManager, model_name: str, continue_dir: str = None) -> str:
    """Setup output directory for results"""
    if continue_dir:
        results_dir = continue_dir
        print(f"Continuing from existing directory: {results_dir}")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = f"{config.get('data.results', 'results')}/{model_name}_{timestamp}/output"
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created new results directory: {results_dir}")

    return results_dir


def run_prediction():
    """Main prediction function"""
    # Parse arguments and load configuration
    args = parse_args()
    config = ConfigManager(args.config)

    # Load prediction parameters
    pred_config = config.prediction_config
    OBS_LEN = pred_config["obs_len"]
    FUT_LEN = pred_config["fut_len"]

    # Initialize components
    data_loader = NuScenesDataLoader(config)
    data_loader.initialize_dataset()

    predictor = PredictionPipeline(config, args.model)
    predictor.set_data_loader(data_loader)
    predictor.initialize()

    # Setup output directory
    results_dir = setup_output_directory(config, args.model, args.continue_dir)

    # Get scenes to process
    scenes_to_process = data_loader.get_scenes_to_process(args.scene)

    if args.scene:
        print(f"Processing specific scene: {args.scene}")
    else:
        print(f"Processing all {len(scenes_to_process)} scenes")

    # Process each scene
    for scene_data in scenes_to_process:
        scene_name = scene_data.scene_name

        # Skip if already processed in continuation mode
        output_path = os.path.join(results_dir, f"{scene_name}.json")
        if os.path.exists(output_path):
            print(f"Skipping already processed scene: {scene_name}")
            continue

        try:
            # Process the scene
            scene_result = predictor.process_scene(scene_data, OBS_LEN, FUT_LEN)

            # Update metadata
            scene_result["metadata"]["model"] = args.model
            scene_result["metadata"]["timestamp"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            scene_result["metadata"]["total_frames"] = len(scene_result["frames"])

            # Save scene data
            scene_file_path = os.path.join(results_dir, f"{scene_name}.json")
            save_dict_to_json(scene_result, scene_file_path)
            print(f"Scene data saved to {scene_file_path} with {len(scene_result['frames'])} frames")

        except Exception as e:
            print(f"Error processing scene {scene_name}: {e}")
            continue


if __name__ == "__main__":
    run_prediction()