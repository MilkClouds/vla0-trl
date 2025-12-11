"""Action processing utilities for VLA-0."""
import torch
import numpy as np
from typing import List, Dict, Optional


class ActionProcessor:
    """Handles action discretization and text conversion for VLA-0."""
    
    def __init__(
        self,
        num_bins: int = 1000,
        action_dim: int = 7,
        horizon: int = 8,
    ):
        self.num_bins = num_bins
        self.action_dim = action_dim
        self.horizon = horizon
        self.stats: Optional[Dict] = None
    
    def set_stats(self, stats: Dict):
        """Set dataset statistics for normalization."""
        self.stats = stats
        self._min = torch.tensor(stats["min"])
        self._max = torch.tensor(stats["max"])
    
    def action_to_text(self, actions: torch.Tensor) -> List[str]:
        """Convert continuous actions to discretized text."""
        if self.stats is None:
            raise ValueError("Stats not set. Call set_stats() first.")
        
        min_act = self._min.to(actions.device)
        max_act = self._max.to(actions.device)
        
        # Normalize to [0, 1] then scale to [0, num_bins]
        normalized = (actions - min_act) / (max_act - min_act)
        discretized = torch.round(normalized * self.num_bins).long()
        discretized = discretized.reshape(actions.shape[0], -1)
        
        return [" ".join(map(str, x.tolist())) for x in discretized]
    
    def text_to_action(self, action_texts: List[str]) -> torch.Tensor:
        """Convert discretized text back to continuous actions."""
        if self.stats is None:
            raise ValueError("Stats not set. Call set_stats() first.")
        
        bs = len(action_texts)
        try:
            action_texts = [x.strip() for x in action_texts]
            tokens = [[int(x) for x in text.split()] for text in action_texts]
            actions = torch.tensor(tokens, dtype=torch.float32)
            
            # Handle incomplete sequences
            if bs == 1 and len(actions[0]) % self.action_dim != 0:
                valid_len = len(actions[0]) - (len(actions[0]) % self.action_dim)
                actions = actions[:, :valid_len]
            
            actions = actions.reshape(bs, -1, self.action_dim)
            
            # Pad or truncate to horizon
            if actions.shape[1] < self.horizon:
                pad = actions[:, -1:].repeat(1, self.horizon - actions.shape[1], 1)
                actions = torch.cat([actions, pad], dim=1)
            elif actions.shape[1] > self.horizon:
                actions = actions[:, :self.horizon]
            
            # Denormalize
            actions = (actions / self.num_bins) * (self._max - self._min) + self._min
            
        except Exception as e:
            print(f"Error parsing action text: {e}")
            mid = (self._min + self._max) / 2
            actions = mid.repeat(bs, self.horizon, 1)
        
        return actions
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for action prediction."""
        return (
            f"Analyze the input image and predict robot actions for the next "
            f"{self.horizon} timesteps. Each action has {self.action_dim} dimensions. "
            f"Output a single sequence of {self.horizon * self.action_dim} integers "
            f"(0-{self.num_bins} each), representing the {self.horizon} timesteps "
            f"sequentially. Provide only space separated numbers. Nothing else."
        )


def compute_dataset_stats(dataset, key: str = "out_ori_act") -> Dict:
    """Compute min/max statistics for actions in dataset."""
    all_actions = []
    for i in range(min(len(dataset), 10000)):  # Sample up to 10k
        sample = dataset[i]
        if key in sample:
            all_actions.append(sample[key])
    
    all_actions = np.concatenate(all_actions, axis=0)
    return {
        "min": all_actions.min(axis=0).tolist(),
        "max": all_actions.max(axis=0).tolist(),
    }

