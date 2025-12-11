"""LIBERO evaluation loop for VLA-0."""

import csv
import os
from typing import Dict, Optional

import imageio
import numpy as np
import torch
import torchvision.transforms.functional as TF
from filelock import FileLock
from PIL import Image
from tqdm import tqdm

from .libero_env import get_evaluation_tasks, init_libero_env

DUMMY_ACTION = [0.0] * 6 + [-1.0]


def flip_image(img: np.ndarray) -> np.ndarray:
    """LIBERO images are flipped; flip them back."""
    return np.ascontiguousarray(img[::-1, ::-1])


def preprocess_obs(
    obs: Dict,
    img_size: int = 224,
    crop_ratio: float = 0.875,
    tile_images: bool = True,
) -> Image.Image:
    """Preprocess LIBERO observation for model input."""
    cams = ["agentview_image", "robot0_eye_in_hand_image"]
    images = []

    for cam in cams:
        img = flip_image(obs[cam])
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Center crop (no random crop during eval)
        if crop_ratio < 1.0:
            h, w = img.shape[-2:]
            crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
            img = TF.crop(img, top, left, crop_h, crop_w)

        if img_size > 0:
            img = TF.resize(img, [img_size, img_size])

        img = (img * 255).byte()
        img = img.permute(1, 2, 0).numpy()
        images.append(img)

    if tile_images:
        tiled = np.concatenate(images, axis=1)
        return Image.fromarray(tiled)

    return [Image.fromarray(img) for img in images]


class LiberoEvaluator:
    """Evaluator for VLA models on LIBERO benchmark."""

    def __init__(
        self,
        model,
        log_dir: str = "./eval_logs",
        save_video: bool = True,
        seed: int = 7,
        action_horizon: int = 1,
        frame_skip: int = 10,
        img_size: int = 224,
        crop_ratio: float = 0.875,
        tile_images: bool = True,
        shard_id: int = 0,
        num_shards: int = 1,
        skip_evaluated: bool = False,
        ensemble_prediction: int = 1,
        ensemble_version: int = 1,
        ensemble_weight: float = 0.5,
    ):
        self.model = model
        self.log_dir = log_dir
        self.save_video = save_video
        self.seed = seed
        self.action_horizon = action_horizon
        self.frame_skip = frame_skip
        self.img_size = img_size
        self.crop_ratio = crop_ratio
        self.tile_images = tile_images
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.skip_evaluated = skip_evaluated
        self.ensemble_prediction = ensemble_prediction
        self.ensemble_version = ensemble_version
        self.ensemble_weight = ensemble_weight

        os.makedirs(log_dir, exist_ok=True)

    def run_episode(
        self,
        env,
        init_state: np.ndarray,
        instruction: str,
        max_steps: int,
    ) -> tuple:
        """Run a single evaluation episode."""
        env.reset()
        obs = env.set_init_state(init_state)

        frames = []
        action_chunk = None
        action_i = 0
        action_horizon = self.action_horizon

        # Ensemble: maintain list of previous action chunks
        old_action_chunks = [] if self.ensemble_prediction > 1 else None

        for t in tqdm(range(max_steps + self.frame_skip), desc="steps", leave=False):
            if t < self.frame_skip:
                obs, _, done, _ = env.step(DUMMY_ACTION)
                continue

            if action_i >= action_horizon or t == self.frame_skip:
                image = preprocess_obs(obs, self.img_size, self.crop_ratio, self.tile_images)
                action_chunk = self.model.predict(image, instruction).numpy()

                # Ensemble prediction: average overlapping action chunks
                if self.ensemble_prediction > 1 and old_action_chunks is not None:
                    old_action_chunks.append(action_chunk.copy())
                    if len(old_action_chunks) > self.ensemble_prediction:
                        old_action_chunks.pop(0)

                    # Weighted average of overlapping chunks
                    if len(old_action_chunks) > 1:
                        ensemble_chunk = np.zeros_like(action_chunk)
                        ensemble_count = np.zeros_like(action_chunk)

                        new_old_chunks = []
                        num_old = len(old_action_chunks)
                        for i, old_chunk in enumerate(old_action_chunks[:-1]):
                            if len(old_chunk) <= action_horizon:
                                continue
                            old_chunk = old_chunk[action_horizon:]
                            new_old_chunks.append(old_chunk)

                            # Version 1: flat 0.5 weight for all old chunks
                            # Version 2: exponential decay (w^(n-i-1) for i-th chunk)
                            if self.ensemble_version == 1:
                                weight = self.ensemble_weight
                            else:  # version 2
                                weight = self.ensemble_weight ** (num_old - i - 1)

                            ensemble_chunk[: len(old_chunk)] += weight * old_chunk
                            ensemble_count[: len(old_chunk)] += weight

                        new_old_chunks.append(old_action_chunks[-1])
                        ensemble_chunk += old_action_chunks[-1]
                        ensemble_count += 1

                        old_action_chunks = new_old_chunks
                        action_chunk = ensemble_chunk / ensemble_count

                action_i = 0
                action_horizon = min(self.action_horizon, len(action_chunk))

            act = action_chunk[action_i].tolist()
            act[-1] = 1.0 if act[-1] > 0 else -1.0

            obs, _, done, _ = env.step(act)
            frames.append(flip_image(obs["agentview_image"]))
            action_i += 1

            if done:
                return True, frames

        return False, frames

    def _append_csv(self, suite_dir: str, task: str, run_idx: int, success: bool):
        """Append result to CSV with file locking for parallel safety."""
        csv_path = f"{suite_dir}/results.csv"
        lock_path = f"{csv_path}.lock"

        with FileLock(lock_path):
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["task", "run_idx", "success"])
                writer.writerow([task, run_idx, int(success)])

    @torch.no_grad()
    def evaluate(
        self,
        task_suite_name: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> Dict:
        """Evaluate on specified tasks."""
        tasks_to_evaluate = get_evaluation_tasks(task_suite_name, task_name)
        results = {"success": 0, "failure": 0}

        for suite_name, tasks in tasks_to_evaluate.items():
            suite_dir = f"{self.log_dir}/{suite_name}"
            os.makedirs(suite_dir, exist_ok=True)
            results[suite_name] = {"success": 0, "failure": 0}

            for task in tasks:
                results[suite_name][task] = {"success": 0, "failure": 0}
                env, init_states, max_steps, instruction = init_libero_env(task, suite_name, self.seed)

                # Sharding: select subset of init_states
                n = len(init_states)
                start = self.shard_id * n // self.num_shards
                end = (self.shard_id + 1) * n // self.num_shards
                indices = range(start, end)

                pbar = tqdm(indices, desc=task)
                for i in pbar:
                    # Skip if already evaluated
                    if self.skip_evaluated:
                        success_path = f"{suite_dir}/run{i}__success__{task}.mp4"
                        failure_path = f"{suite_dir}/run{i}__failure__{task}.mp4"
                        if os.path.exists(success_path):
                            results[suite_name][task]["success"] += 1
                            results[suite_name]["success"] += 1
                            results["success"] += 1
                            continue
                        if os.path.exists(failure_path):
                            results[suite_name][task]["failure"] += 1
                            results[suite_name]["failure"] += 1
                            results["failure"] += 1
                            continue

                    success, frames = self.run_episode(env, init_states[i], instruction, max_steps)

                    results[suite_name][task]["success" if success else "failure"] += 1
                    results[suite_name]["success" if success else "failure"] += 1
                    results["success" if success else "failure"] += 1

                    s, f = results[suite_name][task]["success"], results[suite_name][task]["failure"]
                    pbar.set_postfix_str(f"{s}/{s + f} success")

                    self._append_csv(suite_dir, task, i, success)
                    if self.save_video:
                        suffix = "success" if success else "failure"
                        video_path = f"{suite_dir}/run{i}__{suffix}__{task}.mp4"
                        imageio.mimwrite(video_path, frames, fps=20)

                env.close()

        self._print_results(results)
        return results

    def _print_results(self, results: Dict):
        """Print evaluation results with color coding."""
        G, B, RST = "\033[92m", "\033[1m", "\033[0m"

        print(f"{B}{'=' * 60}{RST}")
        for suite, data in results.items():
            if suite in ("success", "failure"):
                continue
            print(f"{B}--- {suite} ---{RST}")
            for task, task_data in data.items():
                if task in ("success", "failure"):
                    continue
                t = task_data["success"] + task_data["failure"]
                r = task_data["success"] / t * 100 if t > 0 else 0
                print(f"  {task}: {G}{task_data['success']}{RST}/{t} ({r:.1f}%)")
            total = data["success"] + data["failure"]
            rate = data["success"] / total * 100 if total > 0 else 0
            print(f"{B}Suite Total: {G}{data['success']}{RST}/{total} ({rate:.1f}%){RST}")

        print(f"{B}{'=' * 60}{RST}")
        total = results["success"] + results["failure"]
        rate = results["success"] / total * 100 if total > 0 else 0
        print(f"{B}TOTAL: {G}{results['success']}{RST}/{total} ({rate:.1f}%){RST}")
