# vla0-trl: Minimal VLA-0 Reimplementation with TRL

Unofficial reimplementation of [VLA-0](https://github.com/NVlabs/vla0) using [TRL](https://github.com/huggingface/trl)'s SFTTrainer.

Start building your own VLA from this minimal codebase!

## Why This Repo?

| Codebase | Lines of Code | LIBERO Avg |
|----------|---------------|------------|
| [LeRobot](https://github.com/huggingface/lerobot) | ~113,600 | - |
| [OpenVLA-OFT](https://github.com/moojink/openvla-oft) | ~17,800 | 97.1% |
| [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | ~17,500 | - |
| [OpenPI](https://github.com/Physical-Intelligence/openpi) | ~16,900 | 96.9% |
| [OpenVLA](https://github.com/openvla/openvla) | ~14,800 | 76.5% |
| [VLA-0](https://github.com/NVlabs/vla0) | ~5,500 | 94.7% |
| **This repo** | **~1,200** | 89.6% |

<!-- | [UniVLA](https://github.com/OpenDriveLab/UniVLA) | ~23,000 | 95.2% | -->
<!-- | [FLOWER](https://github.com/intuitive-robots/flower_vla_calvin) | ~10,500 | 96.9% | -->

Other repos support multiple environments, hardware drivers, or diverse policies—this one focuses solely on LIBERO training. Not a fair comparison, but if you want to learn VLA internals, this is the simplest starting point.

How is it so short? Thanks to [transformers](https://github.com/huggingface/transformers) for Qwen2-VL, [TRL](https://github.com/huggingface/trl) for SFTTrainer, [LeRobot](https://github.com/huggingface/lerobot) for LeRobotDataset, and [kernels](https://github.com/huggingface/kernels) for Flash Attention—we just wire them together with [VLA-0](https://github.com/NVlabs/vla0)'s action tokenization. Beyond the smaller codebase, we also gain functional advantages: the original VLA-0 relies on custom DDP with mostly manual implementations, whereas we get Flash Attention 2/3 and WandB logging and many other features out of the box.

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

## Installation

<!-- TODO: upgrade lerobot -->

We recommend using [`uv`](https://docs.astral.sh/uv/) for managing dependencies.

```bash
uv venv --python 3.11
uv pip install -e .
# LeRobot dependency
GIT_LFS_SKIP_SMUDGE=1 uv pip install git+https://github.com/huggingface/lerobot.git@f39652707caed42a7cd5ab36066da5663b9565eb

# For evaluation
uv pip install -e ".[eval]"

# Do not forget activating your venv
source .venv/bin/activate
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

### SLURM

For SLURM users, see [`scripts/train.sbatch`](scripts/train.sbatch) and [`scripts/eval.sbatch`](scripts/eval.sbatch). The `eval.sbatch` demonstrates batch evaluation with round-robin shard distribution across multiple GPUs.

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

## Limitations (inherited from VLA-0)

- **LIBERO only** — other environments not ported
- **Qwen2-VL only** — other backbones not supported

## Known Issues

### Ensemble Prediction is Non-Functional (inherited from original)

Both the original VLA-0 (`libs/RoboVerse/roboverse/evals/libero/eval.py`) and this refactored implementation have a bug where `--ensemble_prediction` has **no effect** when `action_horizon >= horizon`. The ensemble logic trims previous chunks by `action_horizon` each step (`old_chunk = old_chunk[action_horizon:]`), which produces an empty array when `action_horizon == horizon`. With default settings (`horizon=8`, `action_horizon=8`), ensemble is completely disabled regardless of `--ensemble_prediction` value.

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
