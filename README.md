This project is to predict the resilience level of c/cpp programs by fin-tuning CodeBERT-cpp model (https://huggingface.co/neulab/codebert-cpp).

We both do classification prediction and regression prediction
This repository is doing regression prediction task.

The Data set is here: https://github.com/hjiang13/DARE

# BERTRegression with Chunked Code Inputs and Pooling Strategies

This repository provides a flexible and modular framework for training a regression model on source code using CodeBERT. It supports token chunking, multiple pooling strategies, overlap control, and optional LoRA-based fine-tuning.

---

## 🔧 Features

- ✅ Supports pooling strategies: `mean`, `max`, `attn`, `lstm`
- ✅ Optional [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) for efficient fine-tuning
- ✅ Encoder freezing for lightweight experiments
- ✅ Configurable overlapping chunk windows
- ✅ Batch script to evaluate all hyperparameter combinations
- ✅ Outputs per-run MSE for tracking performance

---

## 🚀 Quick Start

### Step 1. Install dependencies

```bash
pip install torch transformers peft matplotlib pandas

### Step 2. Prepare dataset
Ensure your dataset is located at:

dataset/SDC_train_resilience_r.jsonl
dataset/SDC_test_resilience_r.jsonl

Each file should be a JSONL where each line is:

json
{"code": "...", "label": float}

🧪 Run a Single Experiment
bash
python BERTRegression_chunck_functional_freeze.py \
  --pooling_type mean \
  --overlap_ratio 0.5 \
  --epochs 5 \
  --batch_size 8 \
  --use_lora \
  --freeze_encoder

  ⚙️ Command-Line Arguments
Argument	Type	Description
--pooling_type	str	Pooling strategy: mean, max, attn, or lstm
--overlap_ratio	float	Overlap between code chunks (0.0 to 1.0)
--use_lora	flag	Enable LoRA-based fine-tuning
--freeze_encoder	flag	Freeze CodeBERT encoder during training
--epochs	int	Number of training epochs
--batch_size	int	Batch size during training

🔁 Run All Experiments
Use the provided bash script to evaluate all combinations:

bash
Copy
Edit
chmod +x run_experiments_with_freeze.sh
./run_experiments_with_freeze.sh
This will iterate over:

Pooling types: mean, max, attn, lstm

Overlap ratios: 0.0, 0.25, 0.5, 0.75

LoRA on / off

Encoder frozen / not frozen

Final MSE results are logged in:

Copy
Edit
mse_results_with_freeze.txt

Pooling | Overlap | LoRA | Frozen | Final MSE
--------|---------|------|--------|-----------
mean    | 0.25    | on   | yes    | 0.1782
lstm    | 0.5     | off  | no     | 0.2019
...

📚 Citation
Please consider citing the following if this project helps your work:

CodeBERT: A Pre-Trained Model for Programming and Natural Languages

LoRA: Low-Rank Adaptation of Large Language Models

👨‍🔬 Maintainer
Developed and maintained by Hailong Jiang, 2025.