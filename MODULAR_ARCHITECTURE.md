# LightEMMA - Modular Architecture

## Overview

LightEMMA has been refactored into a modular architecture for better maintainability and extensibility. The codebase is now organized into logical modules with clear separation of concerns.

## New Architecture

### Directory Structure
```
LightEMMA/
├── core/                    # Core system components
│   ├── __init__.py
│   └── config.py           # Configuration management
├── data/                   # Data loading and processing
│   ├── __init__.py
│   └── loader.py           # NuScenes data loader
├── models/                 # Model implementations
│   ├── __init__.py
│   └── predictor.py        # Prediction pipeline
├── evaluation/             # Evaluation components
│   ├── __init__.py
│   └── evaluator.py        # Evaluation logic
├── predict.py              # Main prediction script (refactored)
├── evaluate.py             # Single model evaluation (refactored)
├── evaluate_all.py         # Multi-model evaluation (refactored)
├── utils.py                # Utility functions
├── config.yaml             # Configuration file
└── MyConfig.yaml           # Alternative configuration
```

### Core Modules

#### 1. `core/config.py` - Configuration Management
- **ConfigManager**: Centralized configuration loading and validation
- Dot notation access: `config.get('data.version')`
- Type-safe configuration updates
- YAML file persistence

#### 2. `data/loader.py` - Data Loading
- **NuScenesDataLoader**: Handles dataset initialization and scene loading
- **SceneData**: Container for scene information and frames
- **FrameData**: Container for individual frame data
- Coordinate transformations and data preparation

#### 3. `models/predictor.py` - Prediction Pipeline
- **TrajectoryPredictor**: Handles VLM inference and prediction
- **PredictionPipeline**: Orchestrates the complete prediction process
- **PromptGenerator**: Generates prompts for different reasoning stages
- Modular prompt management and inference logic

#### 4. `evaluation/evaluator.py` - Evaluation System
- **Evaluator**: Comprehensive evaluation of prediction results
- Single-model and multi-model evaluation support
- Metrics calculation and visualization generation
- Error handling and frame exclusion logic

### Key Improvements

#### 1. Separation of Concerns
- **Data Layer**: Isolated data loading and preprocessing
- **Model Layer**: Clean prediction logic with VLM abstraction
- **Evaluation Layer**: Dedicated metrics calculation and reporting
- **Configuration Layer**: Centralized settings management

#### 2. Modularity Benefits
- **Easier Testing**: Each component can be tested independently
- **Better Maintainability**: Changes to one module don't affect others
- **Extensibility**: New models, data sources, or evaluation metrics can be added easily
- **Code Reusability**: Components can be reused across different scripts

#### 3. Error Handling
- Robust error handling throughout the pipeline
- Graceful degradation when frames fail
- Detailed error logging and reporting

#### 4. Configuration Management
- Flexible configuration system
- Environment-specific settings
- Runtime configuration updates

### Migration Guide

#### For Prediction
```python
# Old approach
python predict.py --model gpt-4o --scene scene-0103

# New approach (same interface)
python predict.py --model gpt-4o --scene scene-0103
```

#### For Evaluation
```python
# Old approach
python evaluate.py --results_dir results/gpt-4o --error_handling

# New approach (same interface)
python evaluate.py --results_dir results/gpt-4o --error_handling
```

### Development Benefits

1. **Easier Debugging**: Isolated components make it easier to identify and fix issues
2. **Parallel Development**: Multiple developers can work on different modules simultaneously
3. **Code Reviews**: Smaller, focused modules are easier to review
4. **Testing**: Unit tests can be written for individual components
5. **Documentation**: Each module has a clear, single responsibility

### Future Extensions

The modular architecture makes it easy to add:
- New VLM models (just extend ModelHandler)
- New datasets (implement new DataLoader)
- New evaluation metrics (extend Evaluator)
- New prediction strategies (modify PredictionPipeline)
- Web interfaces (add API layer)
- Distributed processing (add orchestration layer)

### Backwards Compatibility

The refactored code maintains the same command-line interfaces and output formats, ensuring existing workflows continue to work without changes.