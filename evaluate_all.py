#!/usr/bin/env python3
"""
LightEMMA Multi-Model Evaluation Script - Modular Version
"""
import os
import argparse

from core.config import ConfigManager
from evaluation.evaluator import Evaluator
from utils import save_dict_to_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all models excluding error frames")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory with model result folders")
    parser.add_argument("--config", type=str, default="MyConfig.yaml",
                        help="Path to configuration file")
    return parser.parse_args()


def run_multi_model_evaluation():
    """Main multi-model evaluation function"""
    args = parse_args()
    config = ConfigManager(args.config)

    # Initialize evaluator
    evaluator = Evaluator(config)

    # Run multi-model evaluation
    print(f"Evaluating all models in: {args.results_dir}")
    results = evaluator.evaluate_multiple_models(args.results_dir)

    # Print summary for each model
    print("\n" + "="*80)
    print("MULTI-MODEL EVALUATION SUMMARY")
    print("="*80)

    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}")
        print("-" * (len(model_name) + 7))

        summary = model_results["summary"]
        metrics = model_results["metrics"]

        print(f"Total frames: {summary['total_frames']}")
        print(f"Successful frames: {summary['successful_frames']}")
        print(f"Parse error frames: {summary['parse_error_frames']}")
        print(".1f")

        print("Metrics:")
        for metric_name, value in metrics.items():
            if value is not None:
                print(".4f")
            else:
                print(f"  {metric_name}: N/A")

    # Save overall results
    overall_results_path = os.path.join(args.results_dir, "multi_model_evaluation.json")
    save_dict_to_json(results, overall_results_path)
    print(f"\nOverall results saved to {overall_results_path}")


if __name__ == "__main__":
    run_multi_model_evaluation()
