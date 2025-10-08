# LightEMMA AI Coding Instructions# LightEMMA AI Coding Instructions



## Architecture Overview## Architecture Overview

LightEMMA is an end-to-end autonomous driving framework using vision-language models (VLMs) for zero-shot trajectory prediction on nuScenes data. The system processes front-camera images through a chain-of-thought reasoning pipeline:LightEMMA is an end-to-end autonomous driving framework using vision-language models (VLMs) for zero-shot trajectory prediction on nuScenes data. The system processes front-camera images through a chain-of-thought reasoning pipeline:



1. **Scene Description**: Generate detailed driving environment descriptions1. **Scene Description**: Generate detailed driving environment descriptions

2. **Driving Intent Analysis**: Analyze ego vehicle's historical actions to predict maneuvers2. **Driving Intent Analysis**: Analyze ego vehicle's historical actions to predict maneuvers

3. **Trajectory Prediction**: Convert intentions into speed/curvature predictions3. **Trajectory Prediction**: Convert intentions into speed/curvature predictions



## Core Components## Core Components

- `vlm.py`: Unified ModelHandler class supporting 12 VLMs (GPT, Claude, Gemini, Qwen, DeepSeek, LLaMA) with dynamic imports- `vlm.py`: Unified ModelHandler class supporting 12 VLMs (GPT, Claude, Gemini, Qwen, DeepSeek, LLaMA) with dynamic imports

- `predict.py`: Frame-by-frame scene processing with sliding window (obs_len=6, fut_len=6, ext_len=2)- `predict.py`: Frame-by-frame scene processing with sliding window (obs_len=6, fut_len=6, ext_len=2)

- `evaluate.py`: Single-model evaluation with L2 distances, ADE/FDE metrics- `evaluate.py`: Single-model evaluation with L2 distances, ADE/FDE metrics

- `evaluate_all.py`: Multi-model comparison excluding error frames for fair benchmarking- `utils.py`: Coordinate transformations (global↔ego frame), trajectory calculations, visualization

- `utils.py`: Coordinate transformations (global↔ego frame), trajectory calculations, visualization- `config.yaml`: API keys, dataset paths, prediction parameters

- `config.yaml`: API keys, dataset paths, prediction parameters

## Key Patterns & Conventions

## Key Patterns & Conventions

### Model Integration

### Model IntegrationUse dynamic imports in `ModelHandler.initialize_model()` to avoid package conflicts:

Use dynamic imports in `ModelHandler.initialize_model()` to avoid package conflicts:```python

```pythonif "qwen" in self.model_name:

if "qwen" in self.model_name:    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor    # Initialize model with device_map="auto" for GPU acceleration

    # Initialize model with device_map="auto" for GPU acceleration```

```

### Data Processing

### Coordinate System Transformations- Transform coordinates from global to ego frame using `utils.global_to_ego_frame()`

Always transform coordinates from global to ego frame for predictions:- Process scenes frame-by-frame with sliding window: `obs_pos = ego_positions[cur_index - OBS_LEN - 1 : cur_index + 1]`

```python- Extract image IDs with regex: `re.search(r"(\d+)(?=\.jpg$)", image_path)`

# Transform positions to ego-centric frame

obs_pos = global_to_ego_frame(cur_pos, cur_heading, obs_pos)### Results Structure

fut_pos = global_to_ego_frame(cur_pos, cur_heading, fut_pos)Store predictions in timestamped directories:

``````

results/{model}_{timestamp}/

### Sliding Window Processing├── output/     # JSON predictions per scene

Process scenes frame-by-frame with sliding window for temporal context:├── frame/      # Visualization images

```python└── analysis/   # Evaluation metrics

# Extract observation window (past positions for speed/curvature calculation)```

obs_pos = ego_positions[cur_index - OBS_LEN - 1 : cur_index + 1]

obs_time = timestamps[cur_index - OBS_LEN - 1 : cur_index + 1]### Chain-of-Thought Prompting

```Structure prompts with three distinct phases in `predict.py`:

1. Scene description from current frame

### Chain-of-Thought Prompting2. Historical action analysis

Structure prompts with three distinct phases in `predict.py`:3. Future trajectory prediction with speed/curvature values

1. Scene description from current frame

2. Historical action analysis with speed/curvature data## Critical Workflows

3. Future trajectory prediction with exact format requirements: `[(v1, c1), (v2, c2), (v3, c3), (v4, c4), (v5, c5), (v6, c6)]`

### Prediction Pipeline

### Error Handling & Parsing```bash

Handle malformed VLM outputs with robust parsing:# Process all scenes

```pythonpython predict.py --model gpt-4o --all_scenes

# Use error_handling=True for regex-based extraction of numbers

pred_actions = extract_driving_action(pred_actions_str, error_handling=True)# Continue interrupted run

```python predict.py --model gpt-4o --continue_dir results/gpt-4o_20250415-123



### Speed & Curvature Filtering# Single scene with object detection

Apply physical constraints to predictions:python predict.py --model qwen2.5-7b --scene scene-0103 --enable_object_detection

```python```

# Filter out unrealistic speeds and curvatures

speed = 0.0 if speed < 0.2 else speed### Evaluation Pipeline

if curvature > 0.2 or curvature < -0.2:```bash

    curvature = 0.0# Single model evaluation

```python evaluate.py --results_dir results/gpt-4o_20250415-123



### Results Structure# Multi-model comparison

Store predictions in timestamped directories:python evaluate_all.py

``````

results/{model}_{timestamp}/

├── output/     # JSON predictions per scene### Configuration Setup

├── frame/      # Visualization images (when --visualize used)Update `config.yaml` with:

└── analysis/   # Evaluation metrics (generated by evaluate.py)- API keys for commercial models (OpenAI, Anthropic, Gemini)

```- HuggingFace token for model downloads

- nuScenes dataset paths (use v1.0-mini for development)

## Critical Workflows- Model cache directory



### Prediction Pipeline## Development Notes

```bash- Use `device_map="auto"` for automatic GPU distribution on multi-GPU systems

# Process all scenes- Handle quaternion-to-yaw conversions with `utils.quaternion_to_yaw()`

python predict.py --model gpt-4o --all_scenes- Filter predictions with speed threshold: `speed = 0.0 if speed < 0.2 else speed`

- Enable object detection only for Qwen models: `enable_object_detection = "qwen2.5" in args.model.lower()`

# Continue interrupted run (uses existing timestamped directory)

python predict.py --model gpt-4o --continue_dir results/gpt-4o_20250415-123## File References

- `vlm.py`: Model initialization and API response handling

# Single scene with object detection (Qwen models only)- `utils.py`: Trajectory math and coordinate transformations

python predict.py --model qwen2.5-7b --scene scene-0103 --enable_object_detection- `predict.py`: Main prediction loop and scene processing

```- `evaluate.py`: Metrics calculation and error analysis

- `config.yaml`: All configuration parameters</content>

### Evaluation Pipeline<parameter name="filePath">/home/jianyulai/LightEMMA/.github/copilot-instructions.md
```bash
# Single model evaluation
python evaluate.py --results_dir results/gpt-4o_20250415-123

# Enable error handling for malformed VLM outputs
python evaluate.py --results_dir results/gpt-4o --error_handling

# Generate trajectory visualizations
python evaluate.py --results_dir results/gpt-4o --error_handling --visualize

# Multi-model comparison (excludes error frames for fair comparison)
python evaluate_all.py --results_dir results
```

### Configuration Setup
Update `config.yaml` with:
- API keys for commercial models (OpenAI, Anthropic, Gemini)
- HuggingFace token for model downloads
- nuScenes dataset paths (use v1.0-mini for development)
- Model cache directory for local models

### Model-Specific Setup
```bash
# For local models, install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Set environment variables for HuggingFace
export HUGGINGFACE_HUB_TOKEN="your-token"
export HF_HOME="/path/to/model/cache"
```

## Development Notes
- Use `device_map="auto"` for automatic GPU distribution on multi-GPU systems
- Handle quaternion-to-yaw conversions with `utils.quaternion_to_yaw()`
- Filter predictions with speed threshold: `speed = 0.0 if speed < 0.2 else speed`
- Enable object detection only for Qwen models: `enable_object_detection = "qwen2.5" in args.model.lower()`
- DeepSeek models are discontinued due to stability issues
- Process scenes sequentially; each scene generates one JSON file with all frames
- Frame processing can fail - use try/except blocks and continue processing

## Integration Points
- **nuScenes API**: Used for scene/sample data loading and camera parameters
- **HuggingFace Transformers**: For local model loading and tokenization
- **OpenAI/Anthropic/Google APIs**: For commercial model inference
- **PyTorch**: For local model inference with CUDA acceleration
- **Matplotlib**: For trajectory visualization overlays

## File References
- `vlm.py`: Model initialization and API response handling
- `utils.py`: Trajectory math and coordinate transformations
- `predict.py`: Main prediction loop and scene processing
- `evaluate.py`: Metrics calculation and error analysis
- `evaluate_all.py`: Cross-model comparison logic
- `config.yaml`: All configuration parameters