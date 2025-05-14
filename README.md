This project is to predict the resilience level of c/cpp programs by fin-tuning CodeBERT-cpp model (https://huggingface.co/neulab/codebert-cpp).

We both do classification prediction and regression prediction
This repository is doing regression prediction task.

The Data set is here: https://github.com/hjiang13/DARE

# eHAppA：BERTRegression with Chunked Code Inputs and Pooling Strategies

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
```
### Step 2. Prepare dataset
Ensure your dataset is located at:
```bash
dataset/SDC_train_resilience_r.jsonl
dataset/SDC_test_resilience_r.jsonl
```
Each file should be a JSONL where each line is:

```json
{"code": "...", "label": float}
```
### 🧪 Step 3. Run a Single Experiment
```bash
python BERTRegression_chunck_functional_freeze.py \
  --pooling_type mean \
  --overlap_ratio 0.5 \
  --epochs 5 \
  --batch_size 8 \
  --use_lora \
  --freeze_encoder
```
**⚙️ Command-Line Arguments**

| Argument           | Type  | Description                                        |
| ------------------ | ----- | -------------------------------------------------- |
| `--pooling_type`   | str   | Pooling strategy: `mean`, `max`, `attn`, or `lstm` |
| `--overlap_ratio`  | float | Overlap between code chunks (0.0 to 1.0)           |
| `--use_lora`       | flag  | Enable LoRA-based fine-tuning                      |
| `--freeze_encoder` | flag  | Freeze CodeBERT encoder during training            |
| `--epochs`         | int   | Number of training epochs                          |
| `--batch_size`     | int   | Batch size during training                         |


### Step 4. 🔁 Run All Experiments
Use the provided bash script to evaluate all combinations:
```bash
chmod +x run_experiments_with_freeze.sh
./run_experiments_with_freeze.sh
```
This will iterate over:

**Pooling types:** `mean`, `max`, `attn`, `lstm`

**Overlap ratios:** `0.0`, `0.25`, `0.5`, `0.75`

**LoRA** `on` / `off`

**Encoder** `frozen` / `not frozen`

Final MSE results are logged in:
```
mse_results_with_freeze.txt
```
**📊 Example Output**
Pooling | Overlap | LoRA | Frozen | Final MSE
--------|---------|------|--------|-----------
mean    | 0.25    | on   | yes    | 0.1782
lstm    | 0.5     | off  | no     | 0.2019
...

### 🧠 Best Practices

✅ Use --freeze_encoder for fast, reproducible testing of pooling and overlap.

✅ Remove --freeze_encoder to evaluate LoRA vs full fine-tuning.

✅ Tune max_chunks using estimate_max_chunks() for longer inputs.


### 📚 Citation

Please consider citing the following if this project helps your work:
HAPPA: A Modular Platform for HPC Application Resilience Analysis with LLMs Embedded
```
@inproceedings{jiang2024happa,
  title={HAppA: A Modular Platform for HPC Application Resilience Analysis with LLMs Embedded},
  author={Jiang, Hailong and Zhu, Jianfeng and Fang, Bo and Barker, Kevin and Chen, Chao and Jin, Ruoming and Guan, Qiang},
  booktitle={2024 43rd International Symposium on Reliable Distributed Systems (SRDS)},
  pages={40--51},
  year={2024},
  organization={IEEE}
}
```
CodeBERT: A Pre-Trained Model for Programming and Natural Languages

LoRA: Low-Rank Adaptation of Large Language Models


### 👨‍🔬 Maintainer
Developed and maintained by Hailong Jiang (jiang13 at kent.edu), 2025.