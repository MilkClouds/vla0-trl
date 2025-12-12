#!/usr/bin/env python3
"""
Migration verification tests comparing original vs refactored VLA-0 code.

Usage:
    cd refactored && pip install -e . && pytest scripts/verify_migration.py -v
"""

import numpy as np
import pytest
import torch
from transformers import AutoProcessor

from rv_train.collator import VLACollator
from rv_train.dataset import LiberoDataset

# --- Test fixtures ---

LIBERO_STATS = {
    "min": np.array([-0.396, -0.729, -0.552, -0.123, -0.073, -0.097, -1.0]),
    "max": np.array([0.474, 0.793, 0.648, 0.061, 0.114, 0.047, 1.0]),
}


@pytest.fixture(scope="module")
def processor():
    return AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


@pytest.fixture(scope="module")
def dataset():
    np.random.seed(42)
    return LiberoDataset(
        repo_id="physical-intelligence/libero",
        episodes=[0],
        horizon=8,
        tile_images=True,
        brightness_aug=0.0,
        contrast_aug=0.0,
        saturation_aug=0.0,
        hue_aug=0.0,
    )


# --- Original implementations (from rv_train/models/qwen/model.py) ---


def _original_discretize(action, min_act, max_act):
    """From rv_train/models/qwen/model.py lines 245-247"""
    normalized = (action - min_act) / (max_act - min_act)
    return np.round(normalized * 1000).astype(int)


def _original_tile(images):
    """From rv_train/models/qwen/model.py lines 892-910"""
    tensors = [torch.from_numpy(img).float() for img in images]
    widths, heights = zip(*(t.shape[:-1] for t in tensors))
    dst = torch.zeros((max(heights), sum(widths), 3))
    x = 0
    for t in tensors:
        dst[: t.shape[0], x : x + t.shape[1], :] = t
        x += t.shape[1]
    return dst.numpy().astype(np.uint8)


def _original_collate(processor, messages, images):
    """
    Replicates rv_train/models/qwen/model.py get_qwen_inputs + label masking.
    Returns (input_ids, labels, attention_mask) for comparison.
    """
    # Apply chat template with add_vision_id=True (from configs/vla0.yaml line 26)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_vision_id=False,  # Original VLA-0 config uses this
    )

    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    )

    labels = inputs["input_ids"].clone()

    # Original label masking logic (rv_train/models/qwen/model.py lines 681-729)
    action_text = messages[2]["content"][0]["text"]
    action_token_len = len(processor.tokenizer(action_text, add_special_tokens=False)["input_ids"])
    nonpad_len = inputs["attention_mask"].sum().item()
    sysuser_len = nonpad_len - action_token_len - 2

    labels[0, :sysuser_len] = -100
    labels[labels == 151643] = -100  # mask pad tokens

    return inputs["input_ids"][0], labels[0], inputs["attention_mask"][0]


# --- Refactored implementations ---


def _refactored_discretize(action, min_act, max_act):
    """From refactored/src/rv_train/dataset.py"""
    normalized = (action - min_act) / (max_act - min_act + 1e-8)
    return np.clip(np.round(normalized * 1000).astype(int), 0, 1000)


def _refactored_tile(images):
    """From refactored/src/rv_train/dataset.py"""
    return np.concatenate(images, axis=1)


# --- Tests ---


class TestActionDiscretization:
    """Verify action discretization produces identical results."""

    @pytest.mark.parametrize(
        "action",
        [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
            np.array([0.1, -0.2, 0.3, 0.0, 0.05, -0.05, 1.0]),
            np.array([-0.3, 0.5, -0.4, 0.05, -0.05, 0.02, -1.0]),
        ],
    )
    def test_discretization_matches(self, action):
        original = _original_discretize(action, LIBERO_STATS["min"], LIBERO_STATS["max"])
        refactored = _refactored_discretize(action, LIBERO_STATS["min"], LIBERO_STATS["max"])
        np.testing.assert_array_equal(original, refactored)


class TestImageTiling:
    """Verify image tiling produces identical results."""

    def test_tiling_matches(self):
        np.random.seed(42)
        img1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        original = _original_tile([img1, img2])
        refactored = _refactored_tile([img1, img2])

        assert original.shape == refactored.shape == (224, 448, 3)
        np.testing.assert_array_equal(original, refactored)


class TestLabelMasking:
    """Verify label masking formula: masked_count == sysuser_len."""

    def test_masking_formula(self, processor, dataset):
        collator = VLACollator(processor=processor, action_mask_aug_pct=0.0)
        batch = collator([dataset[0]])

        action_text = dataset[0]["messages"][2]["content"][0]["text"]
        action_len = len(processor.tokenizer(action_text)["input_ids"])
        nonpad_len = batch["attention_mask"][0].sum().item()
        expected_masked = nonpad_len - action_len - 2

        actual_masked = (batch["labels"][0] == -100).sum().item()
        assert actual_masked == expected_masked


class TestCollatorOutput:
    """Verify collator output matches original implementation."""

    def test_input_ids_match(self, processor, dataset):
        """Compare input_ids: original vs refactored."""
        sample = dataset[0]

        # Original
        orig_input_ids, _, _ = _original_collate(processor, sample["messages"], sample["images"])

        # Refactored
        collator = VLACollator(processor=processor, action_mask_aug_pct=0.0)
        batch = collator([sample])
        refactored_input_ids = batch["input_ids"][0]

        torch.testing.assert_close(orig_input_ids, refactored_input_ids)

    def test_labels_match(self, processor, dataset):
        """Compare labels: original vs refactored."""
        sample = dataset[0]

        # Original
        _, orig_labels, _ = _original_collate(processor, sample["messages"], sample["images"])

        # Refactored
        collator = VLACollator(processor=processor, action_mask_aug_pct=0.0)
        batch = collator([sample])
        refactored_labels = batch["labels"][0]

        torch.testing.assert_close(orig_labels, refactored_labels)

    def test_attention_mask_match(self, processor, dataset):
        """Compare attention_mask: original vs refactored."""
        sample = dataset[0]

        # Original
        _, _, orig_mask = _original_collate(processor, sample["messages"], sample["images"])

        # Refactored
        collator = VLACollator(processor=processor, action_mask_aug_pct=0.0)
        batch = collator([sample])
        refactored_mask = batch["attention_mask"][0]

        torch.testing.assert_close(orig_mask, refactored_mask)

    def test_output_shapes(self, processor, dataset):
        collator = VLACollator(processor=processor, action_mask_aug_pct=0.0)
        batch = collator([dataset[0], dataset[1]])

        assert batch["input_ids"].shape[0] == 2
        assert batch["labels"].shape == batch["input_ids"].shape
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        assert batch["pixel_values"].ndim == 2
        assert batch["image_grid_thw"].shape == (2, 3)

    def test_no_pad_tokens_in_single_sample(self, processor, dataset):
        """Single sample should have no padding."""
        collator = VLACollator(processor=processor, action_mask_aug_pct=0.0)
        batch = collator([dataset[0]])

        pad_id = processor.tokenizer.pad_token_id
        assert (batch["input_ids"] == pad_id).sum() == 0
        assert batch["attention_mask"].sum() == batch["input_ids"].numel()


class TestActionProcessor:
    """Verify ActionProcessor roundtrip."""

    def test_roundtrip_error(self):
        from rv_train.utils import ActionProcessor

        processor = ActionProcessor(num_bins=1000, action_dim=7, horizon=8)
        processor.set_stats({"min": [-1.0] * 7, "max": [1.0] * 7})

        actions = torch.rand(1, 8, 7) * 2 - 1  # Random in [-1, 1]
        text = processor.action_to_text(actions)
        recovered = processor.text_to_action(text)

        error = (actions - recovered).abs().max().item()
        assert error < 0.01, f"Roundtrip error too high: {error}"

    def test_token_count(self):
        from rv_train.utils import ActionProcessor

        processor = ActionProcessor(num_bins=1000, action_dim=7, horizon=8)
        processor.set_stats({"min": [-1.0] * 7, "max": [1.0] * 7})

        actions = torch.rand(1, 8, 7)
        text = processor.action_to_text(actions)
        tokens = text[0].split()
        assert len(tokens) == 8 * 7, f"Expected 56 tokens, got {len(tokens)}"


class TestDatasetFormat:
    """Verify dataset sample format."""

    def test_sample_structure(self, dataset):
        sample = dataset[0]
        assert "messages" in sample
        assert "images" in sample

        messages = sample["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_image_format(self, dataset):
        sample = dataset[0]
        images = sample["images"]
        assert len(images) == 1  # Tiled into single image
        assert images[0].size == (448, 224)  # W x H

    def test_action_token_count(self, dataset):
        sample = dataset[0]
        action_text = sample["messages"][2]["content"][0]["text"]
        tokens = action_text.split()
        assert len(tokens) == 8 * 7  # horizon * action_dim
