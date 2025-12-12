"""LIBERO dataset for VLA-0 training with SFTTrainer."""

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from einops import rearrange
from typing import Dict, List, Optional, Tuple
import torchvision.transforms.functional as TF


class LiberoDataset(Dataset):
    """LIBERO dataset formatted for VLA-0 training with SFTTrainer.

    Returns samples in the format expected by SFTTrainer for VLMs:
    - messages: conversation format with system, user (with images), assistant
    - images: list of PIL images
    """

    def __init__(
        self,
        repo_id: str = "physical-intelligence/libero",
        history: int = 1,
        horizon: int = 8,
        action_key: str = "actions",
        state_key: str = "state",
        cam_list: Tuple[str, ...] = ("image", "wrist_image"),
        img_size: int = 224,
        crop_ratio: float = 0.875,
        tile_images: bool = True,
        brightness_aug: float = 0.2,
        contrast_aug: float = 0.2,
        saturation_aug: float = 0.2,
        hue_aug: float = 0.05,
        num_bins: int = 1000,
        action_dim: int = 7,
        episodes: Optional[List[int]] = None,
    ):
        self.history = history
        self.horizon = horizon
        self.cam_list = cam_list
        self.img_size = img_size
        self.crop_ratio = crop_ratio
        self.tile_images = tile_images
        self.num_bins = num_bins
        self.action_dim = action_dim

        # Augmentation params
        self.brightness_aug = brightness_aug
        self.contrast_aug = contrast_aug
        self.saturation_aug = saturation_aug
        self.hue_aug = hue_aug

        # Build delta_timestamps for history and horizon
        # Matches original RoboVerse: actions from -history to horizon-1
        fps = 10  # LIBERO dataset fps
        delta_timestamps = {
            state_key: [-x / fps for x in range(history - 1, -1, -1)],
            action_key: [x / fps for x in range(-history, horizon)],
        }
        for cam in cam_list:
            delta_timestamps[cam] = [-x / fps for x in range(history - 1, -1, -1)]

        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            delta_timestamps=delta_timestamps,
            episodes=episodes,
        )

        self.action_key = action_key
        self.state_key = state_key

        # Compute stats
        self.stats = self._compute_stats()

        # System prompt
        self.system_prompt = (
            f"Analyze the input image and predict robot actions for the next "
            f"{horizon} timesteps. Each action has {action_dim} dimensions. "
            f"Output a single sequence of {horizon * action_dim} integers "
            f"(0-{num_bins} each), representing the {horizon} timesteps "
            f"sequentially. Provide only space separated numbers. Nothing else."
        )

    def _compute_stats(self) -> Dict:
        """Compute action statistics for normalization (future actions only)."""
        all_actions = []
        sample_size = min(len(self.dataset), 10000)
        indices = np.random.choice(len(self.dataset), sample_size, replace=False)

        for idx in indices:
            sample = self.dataset[int(idx)]  # Convert np.int64 to int
            all_act = sample[self.action_key].numpy()
            future_actions = all_act[self.history :]  # Only future actions
            all_actions.append(future_actions)

        all_actions = np.concatenate(all_actions, axis=0)
        return {
            "out_ori_act": {
                "min": all_actions.min(axis=0).tolist(),
                "max": all_actions.max(axis=0).tolist(),
            }
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def _process_images(self, sample: Dict) -> List[Image.Image]:
        """Extract and process images from sample."""
        images = []
        for cam in self.cam_list:
            img = sample[cam]  # (C, H, W) or (1, C, H, W)
            if img.ndim == 4:
                img = img[0]
            img = (img * 255).byte()

            # Apply augmentations
            if self.crop_ratio < 1.0:
                h, w = img.shape[-2:]
                crop_h, crop_w = int(h * self.crop_ratio), int(w * self.crop_ratio)
                top = np.random.randint(0, h - crop_h + 1)
                left = np.random.randint(0, w - crop_w + 1)
                img = TF.crop(img, top, left, crop_h, crop_w)

            if self.img_size > 0:
                img = TF.resize(img, [self.img_size, self.img_size])

            # Color augmentation
            img_float = img.float() / 255.0
            if self.brightness_aug > 0:
                img_float = TF.adjust_brightness(
                    img_float, 1 + np.random.uniform(-self.brightness_aug, self.brightness_aug)
                )
            if self.contrast_aug > 0:
                img_float = TF.adjust_contrast(img_float, 1 + np.random.uniform(-self.contrast_aug, self.contrast_aug))
            if self.saturation_aug > 0:
                img_float = TF.adjust_saturation(
                    img_float, 1 + np.random.uniform(-self.saturation_aug, self.saturation_aug)
                )
            if self.hue_aug > 0:
                img_float = TF.adjust_hue(img_float, np.random.uniform(-self.hue_aug, self.hue_aug))

            img = (img_float * 255).byte()
            img = rearrange(img, "c h w -> h w c").numpy()
            images.append(img)

        if self.tile_images and len(images) > 1:
            # Tile images horizontally
            tiled = np.concatenate(images, axis=1)
            return [Image.fromarray(tiled)]

        return [Image.fromarray(img) for img in images]

    def _action_to_text(self, actions: np.ndarray) -> str:
        """Convert actions to discretized text."""
        stats = self.stats["out_ori_act"]
        min_act = np.array(stats["min"])
        max_act = np.array(stats["max"])

        normalized = (actions - min_act) / (max_act - min_act + 1e-8)
        discretized = np.round(normalized * self.num_bins).astype(int)
        discretized = np.clip(discretized, 0, self.num_bins)

        return " ".join(map(str, discretized.flatten().tolist()))

    def __getitem__(self, idx: int) -> Dict:
        sample = self.dataset[idx]

        images = self._process_images(sample)
        instruction = sample["task"]
        # Actions include history, take only future actions (matches original)
        all_actions = sample[self.action_key].numpy()
        actions = all_actions[self.history :]  # Skip history, keep horizon
        action_text = self._action_to_text(actions)

        # Format for SFTTrainer VLM - matches original QwenActor.format_data()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "image"}] + [{"type": "text", "text": instruction}]},
            {"role": "assistant", "content": [{"type": "text", "text": action_text}]},
        ]

        return {"messages": messages, "images": images}
