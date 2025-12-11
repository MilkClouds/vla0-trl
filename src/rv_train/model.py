"""Model loading utilities for VLA-0."""

import pickle
from typing import Dict, Optional

import torch
from transformers import AutoConfig, LogitsProcessor, Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration


class NumberSpaceOnlyProcessor(LogitsProcessor):
    """Constrains generation to numbers (0-9), spaces, and EOS."""

    def __init__(self, tokenizer):
        self.allowed_tokens = set()
        for i in range(10):
            self.allowed_tokens.add(tokenizer.encode(str(i), add_special_tokens=False)[0])
        self.allowed_tokens.add(tokenizer.encode(" ", add_special_tokens=False)[0])
        if tokenizer.eos_token_id is not None:
            self.allowed_tokens.add(tokenizer.eos_token_id)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for token_id in self.allowed_tokens:
            mask[:, token_id] = 0
        return scores + mask


def load_model_for_training(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    use_flash_attention: bool = False,
) -> Qwen2_5_VLForConditionalGeneration:
    """Load Qwen2.5-VL for full fine-tuning (no LoRA/QLoRA).

    REVIEW: Original code supported LoRA/QLoRA. This version does full fine-tuning
    as per paper's best results. Add LoRA support if needed.
    """
    kwargs = {"torch_dtype": torch.bfloat16}
    if use_flash_attention:
        kwargs["attn_implementation"] = "kernels-community/flash-attn3"  # "flash_attention_2"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    return model


def load_processor(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    img_size: int = 224,
    num_cams: int = 2,
    tile_images: bool = True,
) -> Qwen2_5_VLProcessor:
    """Load Qwen2.5-VL processor with correct pixel settings."""
    pixel_count = img_size * img_size
    if tile_images:
        pixel_count *= num_cams

    return Qwen2_5_VLProcessor.from_pretrained(
        model_id,
        min_pixels=pixel_count,
        max_pixels=pixel_count,
    )


class QwenVLActor:
    """Wrapper for inference with trained VLA model."""

    def __init__(
        self,
        model_path: str,
        stats_path: Optional[str] = None,
        horizon: int = 8,
        action_dim: int = 7,
        num_bins: int = 1000,
        device: str = "cuda",
        torch_compile: bool = False,
    ):
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.device = device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(
            device
        )
        self.model.eval()

        if torch_compile:
            self.model = torch.compile(self.model)

        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        self.logits_processor = NumberSpaceOnlyProcessor(self.processor.tokenizer)

        # Load stats
        if stats_path:
            with open(stats_path, "rb") as f:
                self.stats = pickle.load(f)
        else:
            self.stats = None

        self.system_prompt = (
            f"Analyze the input image and predict robot actions for the next "
            f"{horizon} timesteps. Each action has {action_dim} dimensions. "
            f"Output a single sequence of {horizon * action_dim} integers "
            f"(0-{num_bins} each), representing the {horizon} timesteps "
            f"sequentially. Provide only space separated numbers. Nothing else."
        )

    @torch.no_grad()
    def predict(self, image, instruction: str, temperature: float = 0.1):
        """Predict actions given image and instruction."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )

        from qwen_vl_utils import process_vision_info

        image_inputs = process_vision_info(messages)[0]

        inputs = self.processor(text=[text], images=[image_inputs], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = {"max_new_tokens": 1024, "logits_processor": [self.logits_processor]}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        action_text = self.processor.decode(generated, skip_special_tokens=True)

        return self._text_to_action(action_text)

    def _text_to_action(self, text: str):
        """Convert action text to tensor."""
        if self.stats is None:
            raise ValueError("Stats not loaded")

        stats = self.stats["out_ori_act"]
        min_act = torch.tensor(stats["min"])
        max_act = torch.tensor(stats["max"])

        try:
            tokens = [int(x) for x in text.strip().split()]
            actions = torch.tensor(tokens, dtype=torch.float32)
            actions = actions.reshape(-1, self.action_dim)

            if actions.shape[0] < self.horizon:
                pad = actions[-1:].repeat(self.horizon - actions.shape[0], 1)
                actions = torch.cat([actions, pad], dim=0)
            actions = actions[: self.horizon]

            actions = (actions / self.num_bins) * (max_act - min_act) + min_act
        except Exception:
            actions = ((min_act + max_act) / 2).repeat(self.horizon, 1)

        return actions
