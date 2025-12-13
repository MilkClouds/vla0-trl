"""Data collator for VLA training with Qwen2.5-VL."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from qwen_vl_utils import process_vision_info


@dataclass
class VLACollator:
    """Collator for VLA training that handles image + text batching.

    This collator:
    1. Applies chat template to format messages
    2. Processes images through Qwen's vision encoder
    3. Creates labels with masked system/user tokens
    4. Applies action mask augmentation (masking random action tokens)
    """

    processor: Any  # Qwen2_5_VLProcessor
    action_mask_aug_pct: float = 0.4

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(examples)

        # Apply chat template
        texts = []
        image_inputs = []
        action_texts = []

        for example in examples:
            messages = example["messages"]
            images = example["images"]

            # Extract action text (assistant content is [{"type": "text", "text": ...}])
            action_text = messages[-1]["content"][0]["text"]
            action_texts.append(action_text)

            # Format for Qwen processor - inject actual images
            formatted = []
            for msg in messages:
                if msg["role"] == "user":
                    content = []
                    for item in msg["content"]:
                        if item["type"] == "image":
                            content.append({"type": "image", "image": images[0]})
                        else:
                            content.append(item)
                    formatted.append({"role": "user", "content": content})
                else:
                    # system and assistant already in correct format
                    formatted.append(msg)

            text = self.processor.apply_chat_template(
                formatted, tokenize=False, add_generation_prompt=False, add_vision_id=False
            )
            texts.append(text)
            image_inputs.append(process_vision_info(formatted)[0])

        # Tokenize batch
        model_inputs = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

        # Create labels (mask system + user tokens)
        labels = model_inputs["input_ids"].clone()

        for i in range(batch_size):
            # Compute length of system + user portion
            action_tokens = self.processor.tokenizer(action_texts[i], add_special_tokens=False)["input_ids"]
            action_len = len(action_tokens)

            # Total non-pad tokens
            nonpad_len = model_inputs["attention_mask"][i].sum().item()
            # System + user length = total - action - 2 (assistant end tokens)
            sysuser_len = int(nonpad_len - action_len - 2)

            # Mask system + user
            labels[i, :sysuser_len] = -100

            # Apply action mask augmentation (matches original QwenActor logic)
            # BUG: original implementation mask not only action token but also spacebar but I maintain the original logic for now.
            seq_len = labels.size(1)
            if random.random() < 0.1:
                aug_pct = 0.0
            else:
                aug_pct = random.uniform(0.0, self.action_mask_aug_pct)

            mask_len = int(len(action_texts[i]) * aug_pct)
            if mask_len > 0:
                mask_indices = random.sample(range(len(action_texts[i])), mask_len)
                mask_indices = [x + sysuser_len for x in mask_indices]
                mask_indices = [idx for idx in mask_indices if idx < seq_len]
                if mask_indices:
                    labels[i, mask_indices] = -100
                    model_inputs["input_ids"][i, mask_indices] = 30  # '?' token

        # Mask pad tokens (151643 = <|endoftext|>)
        # Note: EOS is 151645 (<|im_end|>), which should NOT be masked for training
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels
        return model_inputs
