# Toolken: Tool-Augmented Language Models with Function Calling

This repository contains the implementation of **Toolken**, a novel approach for training language models to perform tool selection and function calling in mathematical reasoning tasks. The project implements joint token-function prediction architectures with reranking mechanisms to improve tool selection accuracy.

## Overview

Toolken extends standard language models by adding a lightweight function head that enables the model to predict both regular tokens and function calls simultaneously. The approach includes:

- **Joint Token-Function Prediction**: Models learn to predict tokens and function calls in a unified probability space
- **Reranker-based Tool Selection**: A secondary reranking mechanism improves tool selection accuracy
- **Mathematical Reasoning Focus**: Evaluated on datasets like FuncQA and GSM8K-XL

## Project Structure

```
toolken/
├── funcqa-experiments/           # Main FuncQA experiments (along with some GSM8K-XL experiments)
│   ├── eval.py                  # Evaluation scripts
│   ├── inference_funcqa.py      # Base inference
│   ├── inference_funcqa_reranker.py  # Reranker-based inference
│   ├── train.py                 # Training script
│   ├── train_reranker.py        # Reranker training
│   ├── tokenize_funcqa.py       # Data preprocessing
│   ├── data/                    # Dataset files
│   ├── checkpoints_head_only/   # Model checkpoints
│   └── outputs/                 # Results and predictions
├── gsm8k-xl-experiments/        # GSM8K-XL experiments
├── virtualhome-experiments/     # VirtualHome task experiments
├── emre_code/                   # Extended implementation
├── davide/                      # Additional experiments
├── eren/                        # Training scripts
├── evaluation/                  # Evaluation metrics
├── funchub/                     # Function implementations
└── plots/                       # Visualization scripts
```

## Key Components

### 1. Training Pipeline

The training consists of two main phases:

#### Phase 1: Function Head Training
- **Script**: `funcqa-experiments/train.py`
- **Purpose**: Train a lightweight function prediction head on top of frozen language models
- **Models Supported**: Gemma, Llama, Qwen, SmolLM, Phi

```bash
cd funcqa-experiments
python train.py \
    --model_name_or_path google/gemma-3-4b-pt \
    --dataset funcqa \
    --input_file data/funcqa/train.json \
    --lr 1e-3 \
    --num_epochs 3 \
    --save_dir checkpoints_head_only
```

#### Phase 2: Reranker Training
- **Script**: `funcqa-experiments/train_reranker.py`
- **Purpose**: Train a reranking model to improve tool selection accuracy
- **Input**: Mined tool selection examples from Phase 1

```bash
python train_reranker.py \
    --mined_jsonl miner_half_split.jsonl \
    --func_dict data/funcqa/func_dict.json \
    --save_path reranker_head_best.pt
```

### 2. Inference

#### Base Inference
```bash
python inference_funcqa.py \
    --model_name_or_path google/gemma-3-4b-pt \
    --func_head_path checkpoints_head_only/head_best.pth \
    --test_file data/funcqa/funcqa_oh.json
```

#### Reranker-Enhanced Inference
```bash
python inference_funcqa_reranker.py \
    --model_name_or_path google/gemma-3-4b-pt \
    --func_head_path checkpoints_head_only/head_best.pth \
    --reranker_head_path reranker_head_best.pt \
    --top_k_tools 3
```

### 3. Evaluation

Please visit the eval.py and eval_funcqa.ipynb files for evaluation. Evaluation is conducted through comparing the exact and approximate results found by baseline and toolken model. See the plots for results.

## Datasets

### GSM8K-XL
- **Path**: `gsm8k-xl-experiments/`
- **Description**: Extended Grade School Math dataset with function annotations
- **Size**: Basic arithmetic operations like `<add>`, `<subtract>`, `<multiply>`, `<divide>`

### FuncQA
- **Path**: `funcqa-experiments/data/funcqa/`
- **Description**: Mathematical reasoning tasks requiring function calls
- **Functions**: 13 different arithmetic operations such as `<power>`, `<log>`, etc.

### VirtualHome
- **Path**: `virtualhome-experiments/`
- **Description**: Household task planning requiring tool/action selection

## Model Architecture

### Function Head
- Lightweight linear layer: `hidden_size → K functions`
- Trained on top of frozen base language models
- Joint optimization with token prediction

### Reranker
- Embedding-based reranking over top-k tool candidates
- Reduces tool selection errors through secondary scoring
- Trained on mined tool selection examples

## Key Features

1. **Multi-Model Support**: Works with various base models (Gemma, Llama, Qwen, etc.)
2. **Scalable Training**: Efficient training with frozen base models
3. **Reject/Reranking Mechanism**: Implementation of Toolken+
4. **Comprehensive Evaluation**: Multiple datasets and metrics
5. **Visualization**: Performance plots and analysis tools

## Results

The project includes comprehensive evaluation results showing:
- Performance across different model sizes (1B to 8B parameters)
- Comparison with baseline approaches
- Ablation studies on learning rates and training strategies
- Analysis of tool selection accuracy improvements

Results are visualized in generated plots:
- `performance_vs_size_plot.png`: Model size vs accuracy
- `lr_ablation_study.png`: Learning rate analysis
- `grouped_stacked_performance.png`: Comparative performance

## Requirements

```bash
# Core dependencies
torch>=1.12.0
transformers>=4.20.0
wandb  # For experiment tracking

# Additional dependencies
numpy
matplotlib
tqdm
fire
```

## Installation

```bash
git clone https://github.com/ErenSenoglu/toolken.git
cd toolken
```


## References

[1] S. Hao, T. Liu, Z. Wang, and Z. Hu, “Toolkengpt: Augmenting frozen language models with massive tools via tool embeddings,” ArXiv, vol.
abs/2305.11554, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:258823133

[2] K. Yakovlev, S. Nikolenko, and A. Bout, “Toolken+: Improving llm tool usage with reranking and a reject option,” ArXiv, vol. abs/2410.12004, 2024.
[Online]. Available: https://api.semanticscholar.org/CorpusID:273375338

## Acknowledgments

This work was conducted as part of an Advanced Deep Learning course project, focusing on enhancing language models' capability for tool use and mathematical reasoning.