# vla0-trl: Minimal VLA-0 Reimplementation with TRL

Unofficial reimplementation of [VLA-0](https://github.com/NVlabs/vla0) using TRL's SFTTrainer.

Start building your own VLA from this minimal codebase!

## Comparison

| | Original | This repo |
|--|----------|-----------|
| Training loop | Custom DDP | TRL SFTTrainer |
| Flash Attention | Off | On |
| Lines of code | ~4,600 | ~1,200 |

## Results

We reproduce VLA-0's training with comparable results.

![Training Loss](train_loss.png)

| Task Suite | VLA-0 (paper) | This Repo | Diff |
|------------|---------------|-----------|------|
| libero_spatial | 97.0% | 92.8% | -4.2% |
| libero_object | 97.8% | 94.0% | -3.8% |
| libero_goal | 96.2% | 94.6% | -1.6% |
| libero_10 | 87.6% | 77.0% | -10.6% |
| **Average** | **94.7%** | **89.6%** | **-5.1%** |

**Training**: vla0 with gradient clipping enabled.

**Eval**: 200k step checkpoint, `action_horizon=8`, `ensemble_prediction=8`, 50 episodes per task.

**Note**: The exact cause of the performance gap is unclear, but given the comparable results, it should be resolvable by aligning more implementation details with the original. I also tested configuration without gradient clipping but it did not help. (avg success rate 89.05%)

<!-- TODO: open-source intermediate checkpoints and results -->

## Limitations

- **LIBERO only** — other environments not ported
- **Qwen2-VL only** — other backbones not supported

## Known Issues

### Ensemble Prediction is Non-Functional (inherited from original)

Both the original VLA-0 (`libs/RoboVerse/roboverse/evals/libero/eval.py`) and this refactored implementation have a bug where `--ensemble_prediction` has **no effect** when `action_horizon >= horizon`. The ensemble logic trims previous chunks by `action_horizon` each step (`old_chunk = old_chunk[action_horizon:]`), which produces an empty array when `action_horizon == horizon`. With default settings (`horizon=8`, `action_horizon=8`), ensemble is completely disabled regardless of `--ensemble_prediction` value.

## Installation

<!-- TODO: upgrade lerobot -->

```bash
pip install -e .
pip install git+https://github.com/huggingface/lerobot.git@f39652707caed42a7cd5ab36066da5663b9565eb  # LeRobot dependency

# For evaluation
pip install -e ".[eval]"
```

## Usage

### Train

```bash
# Single GPU
python scripts/train.py --config configs/vla0.yaml

# Multi-GPU
accelerate launch --num_processes=8 scripts/train.py --config configs/vla0.yaml
```

### Eval

<!-- TODO: batch-eval implementation -->

```bash
python scripts/eval.py \
    --model_path ./runs/vla0/checkpoint-xxx \
    --task_suite libero_spatial \
    --action_horizon 8 \
    --ensemble_prediction 8 \
    --torch_compile \
    --skip_evaluated \
    --shard_id 0 --num_shards 10
```

| Argument | Description |
|----------|-------------|
| `--task_suite` | Task suite: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| `--action_horizon` | Execute N actions before re-querying model (default: 1) |
| `--ensemble_prediction` | Average N overlapping action chunks (default: 1 = off) |
| `--torch_compile` | Enable torch.compile for faster inference |
| `--skip_evaluated` | Skip episodes with existing result videos |
| `--shard_id`, `--num_shards` | Parallelize: run shard M of N (e.g., 0/10, 1/10, ...) |
| `--log_dir` | Output directory (default: auto-generated with timestamp) |

Note: When running multiple shards in parallel, specify `--log_dir` explicitly to ensure all shards write to the same directory.

## Configuration

See [`configs/vla0.yaml`](configs/vla0.yaml). Key parameters:

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 4e-5 (5e-6 × 8 GPUs) |
| `num_train_epochs` | 192 |
| `per_device_train_batch_size` | 8 |
| `horizon` | 8 |

## Project Structure

```
├── configs/vla0.yaml       # Training config
├── scripts/
│   ├── train.py            # Training entry
│   └── eval.py             # Evaluation entry
└── src/
    ├── rv_train/           # Dataset, collator, model
    └── rv_eval/            # LIBERO evaluator
```

## Attribution

This is a derivative work of [VLA-0](https://github.com/NVlabs/vla0) by NVIDIA.

Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

```bibtex
@article{goyal2025vla0,
  title={VLA-0: Building State-of-the-Art VLAs with Zero Modification},
  author={Goyal, Ankit and others},
  journal={arXiv preprint arXiv:2503.14734},
  year={2025}
}
```

## See Also

- [MIGRATION.md](MIGRATION.md) — detailed comparison with original implementation
