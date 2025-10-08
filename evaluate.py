#!/usr/bin/env python3
"""
LightEMMA Evaluation Script - Modular Version
"""
import os
import argparse

from core.config import ConfigManager
from evaluation.evaluator import Evaluator
from utils import save_dict_to_json


def parse_args():
    parser = argparse.ArgumentParser(description="LightEMMA: Evaluation")
    parser.add_argument("--results_dir", type=str, default='results/gpt-4o')
    parser.add_argument("--error_handling", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--config", type=str, default="MyConfig.yaml",
                        help="Path to configuration file")
    return parser.parse_args()


def save_evaluation_results(results: dict, results_dir: str, filename: str = "analysis.json"):
    """Save evaluation results to file"""
    analysis_path = os.path.join(results_dir, filename)
    save_dict_to_json(results, analysis_path)
    print(f"Evaluation results saved to {analysis_path}")


def run_evaluation():
    """Main evaluation function"""
    args = parse_args()
    config = ConfigManager(args.config)

    # Initialize evaluator
    evaluator = Evaluator(config)

    # Run evaluation
    print(f"Evaluating results in: {args.results_dir}")
    results = evaluator.evaluate_single_model(
        results_dir=args.results_dir,
        error_handling=args.error_handling,
        visualize=args.visualize
    )

    # Print summary
    summary = results["summary"]
    metrics = results["metrics"]

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total frames: {summary['total_frames']}")
    print(f"Successful frames: {summary['successful_frames']}")
    print(f"Parse error frames: {summary['parse_error_frames']}")
    print(".1f")

    print("\nMETRICS:")
    for metric_name, value in metrics.items():
        if value is not None:
            print(".4f")
        else:
            print(f"{metric_name}: N/A")

    # Save results
    save_evaluation_results(results, args.results_dir)


if __name__ == "__main__":
    run_evaluation()
