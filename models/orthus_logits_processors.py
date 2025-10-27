import math
from typing import List, Optional

import torch
from transformers.generation.logits_process import LogitsProcessor


class AllowOnlyTokensAtRelativeOffsetLogitsProcessor(LogitsProcessor):
    r"""
    在触发 token 的相对偏移位置仅允许生成特定 token。
    """

    def __init__(
        self,
        trigger_token_id: int,
        allowed_token_ids: List[int],
        offset: int,
        exclusive: bool = False,
        device: str = "cpu",
    ):
        self.trigger_token_id = trigger_token_id
        self.allowed_token_ids = torch.tensor(allowed_token_ids, device=device)
        self.offset = offset
        self.exclusive = exclusive

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] < self.offset and not self.exclusive:
            return scores

        disallowed_tokens_mask = torch.ones_like(scores, dtype=torch.bool)
        disallowed_tokens_mask[:, self.allowed_token_ids] = False

        if input_ids.shape[1] < self.offset:
            return scores.masked_fill(~disallowed_tokens_mask, torch.finfo(scores.dtype).min)

        trigger_positions = (input_ids[:, -self.offset] == self.trigger_token_id).unsqueeze(-1)

        if self.exclusive:
            return scores.masked_fill(~(disallowed_tokens_mask ^ trigger_positions), torch.finfo(scores.dtype).min)
        return scores.masked_fill(disallowed_tokens_mask & trigger_positions, torch.finfo(scores.dtype).min)


class AllowOnlyTokensInRelativeWindowLogitsProcessor(LogitsProcessor):
    r"""
    在触发 token 的相对窗口内仅允许生成特定 token。
    """

    def __init__(
        self,
        trigger_token_id: int,
        allowed_token_ids: List[int],
        window_width: int,
        exclusive: bool = False,
        device: str = "cpu",
    ):
        self.trigger_token_id = trigger_token_id
        self.allowed_token_ids = torch.tensor(allowed_token_ids, device=device).unsqueeze(0)
        self.window_width = window_width
        self.exclusive = exclusive

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        window_width = min(self.window_width, input_ids.shape[1])
        trigger_positions = (input_ids[:, -window_width:] == self.trigger_token_id).any(dim=1).unsqueeze(-1)

        disallowed_tokens_mask = torch.ones_like(scores, dtype=torch.bool)
        disallowed_tokens_mask[:, self.allowed_token_ids] = False

        if self.exclusive:
            return scores.masked_fill(
                ~(disallowed_tokens_mask ^ trigger_positions),
                torch.finfo(scores.dtype).min,
            )
        return scores.masked_fill(
            disallowed_tokens_mask & trigger_positions,
            torch.finfo(scores.dtype).min,
        )


class SuppressTokensInIndexRangeLogitsProcessor(LogitsProcessor):
    r"""
    在给定的索引范围内屏蔽某些 token。
    """

    def __init__(
        self, suppress_tokens: List[int], start_index: int, end_index: Optional[int] = None, device: str = "cpu"
    ):
        self.suppress_tokens = torch.tensor(suppress_tokens, device=device)
        self.start_index = start_index
        self.end_index = end_index if end_index is not None else math.inf

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        current_index = input_ids.shape[1]
        if self.start_index > current_index or current_index > self.end_index:
            return scores
        suppress_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        suppress_tokens_mask[:, self.suppress_tokens] = True
        return scores.masked_fill(suppress_tokens_mask, torch.finfo(scores.dtype).min)
