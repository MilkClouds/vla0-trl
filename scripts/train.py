#!/usr/bin/env python3
"""VLA-0 Training Script using TRL's SFTTrainer."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

from trl import SFTConfig, SFTTrainer, TrlParser

from rv_train.dataset import LiberoDataset
from rv_train.collator import VLACollator
from rv_train.model import load_model_for_training, load_processor


@dataclass
class ModelArguments:
    model_id: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    use_flash_attention: bool = field(default=False)


@dataclass
class DataArguments:
    repo_id: str = field(default="physical-intelligence/libero")
    history: int = field(default=1)
    horizon: int = field(default=8)
    img_size: int = field(default=224)
    crop_ratio: float = field(default=0.875)
    tile_images: bool = field(default=True)
    brightness_aug: float = field(default=0.2)
    contrast_aug: float = field(default=0.2)
    saturation_aug: float = field(default=0.2)
    hue_aug: float = field(default=0.05)


@dataclass
class VLATrainingArguments:
    action_mask_aug_pct: float = field(default=0.4)


def main():
    parser = TrlParser(dataclass_types=[ModelArguments, DataArguments, VLATrainingArguments, SFTConfig])
    model_args, data_args, vla_args, training_args = parser.parse_args_and_config()

    print(f"Loading model: {model_args.model_id}")
    model = load_model_for_training(
        model_id=model_args.model_id,
        use_flash_attention=model_args.use_flash_attention,
    )

    processor = load_processor(
        model_id=model_args.model_id,
        img_size=data_args.img_size,
        num_cams=2,
        tile_images=data_args.tile_images,
    )

    print("Loading dataset...")
    dataset = LiberoDataset(
        repo_id=data_args.repo_id,
        history=data_args.history,
        horizon=data_args.horizon,
        img_size=data_args.img_size,
        crop_ratio=data_args.crop_ratio,
        tile_images=data_args.tile_images,
        brightness_aug=data_args.brightness_aug,
        contrast_aug=data_args.contrast_aug,
        saturation_aug=data_args.saturation_aug,
        hue_aug=data_args.hue_aug,
    )

    # Save stats for inference
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{training_args.output_dir}/dataset_stats.pkl", "wb") as f:
        pickle.dump(dataset.stats, f)

    collator = VLACollator(
        processor=processor,
        action_mask_aug_pct=vla_args.action_mask_aug_pct,
    )

    # VLM-specific settings
    training_args.max_length = None  # Don't truncate images
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(f"{training_args.output_dir}/final")
    processor.save_pretrained(f"{training_args.output_dir}/final")


if __name__ == "__main__":
    main()
