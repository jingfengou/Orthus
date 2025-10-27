from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache
from transformers.utils import ModelOutput


@dataclass
class OrthusCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    next_image_latents: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Union[Tuple[Tuple[torch.Tensor, ...], ...], Cache]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
