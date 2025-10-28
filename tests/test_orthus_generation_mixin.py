import unittest
from typing import List, Optional

import torch

from models.orthus_generation_mixin import OrthusGenerationMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList


class _DummyConfig:
    is_encoder_decoder = False


class _FakeOutputs:
    def __init__(self, logits: torch.Tensor, next_image_latents: Optional[torch.Tensor] = None):
        self.logits = logits
        if next_image_latents is not None:
            self.next_image_latents = next_image_latents

    def clone(self) -> "_FakeOutputs":
        cloned_latents = getattr(self, "next_image_latents", None)
        return _FakeOutputs(
            self.logits.clone(),
            cloned_latents.clone() if cloned_latents is not None else None,
        )


class _FakeOrthusModel(OrthusGenerationMixin):
    def __init__(self):
        super().__init__()
        self.config = _DummyConfig()
        self.latents_per_image = 1024
        self._reset_state()

    def _reset_state(self):
        self._step = 0
        self._loop_iteration = 0
        self._max_iterations = self.latents_per_image + 4  # slack for EOS iteration
        self._last_output: Optional[_FakeOutputs] = None
        self._reuse_cfg_next = False

    def _get_initial_cache_position(self, *args):
        return args[-1]

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False):
        return model_kwargs

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        filtered = dict(model_kwargs)
        filtered.pop("cfg_scale", None)
        return filtered

    def _has_unfinished_sequences(self, this_peer_finished, synced_gpus, **kwargs):
        keep_running = self._loop_iteration < self._max_iterations
        self._loop_iteration += 1
        return keep_running

    def _token_logits(self, token_id: int) -> torch.Tensor:
        vocab_size = 9000
        logits = torch.full((1, 1, vocab_size), -1e9)
        logits[0, 0, token_id] = 0.0
        return logits

    def _build_output(self) -> _FakeOutputs:
        if self._step == 0:
            output = _FakeOutputs(self._token_logits(42))
        elif self._step == 1:
            output = _FakeOutputs(self._token_logits(8197))
            self._reuse_cfg_next = True
        elif 2 <= self._step < 2 + self.latents_per_image:
            latent_index = self._step - 2
            latent_value = float(latent_index + 1)
            next_latent = torch.full((1, 256), latent_value)
            output = _FakeOutputs(self._token_logits(0), next_latent)
        elif self._step == 2 + self.latents_per_image:
            output = _FakeOutputs(self._token_logits(8710))
        else:
            raise RuntimeError(f"Unexpected call step {self._step}")
        self._step += 1
        return output

    def __call__(self, *args, **kwargs):
        mode_value = kwargs.get("mode")
        has_uncon = "model_inputs_uncon" in kwargs
        kwargs.pop("model_inputs_uncon", None)
        # drop generation plumbing arguments we do not emulate
        kwargs.pop("cfg_scale", None)
        kwargs.pop("logits_processor", None)
        kwargs.pop("logits_warper", None)
        kwargs.pop("diff_pos_id", None)
        kwargs.pop("mode", None)
        kwargs.pop("image_latents", None)
        kwargs.pop("return_dict", None)
        kwargs.pop("streamer", None)
        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)

        if self._reuse_cfg_next and not has_uncon and mode_value == "discrete" and self._last_output is not None:
            self._reuse_cfg_next = False
            return self._last_output.clone()

        output = self._build_output()
        self._last_output = output.clone()
        if has_uncon:
            return output, output.clone()
        return output


def _build_generation_config() -> GenerationConfig:
    config = GenerationConfig(
        pad_token_id=0,
        max_length=1200,
        do_sample=False,
        return_dict_in_generate=False,
    )
    config._pad_token_tensor = torch.tensor([0])
    return config


class OrthusGenerationMixinTests(unittest.TestCase):
    def test_sample_orthus_returns_text_and_latents(self):
        model = _FakeOrthusModel()
        input_ids = torch.zeros((1, 1), dtype=torch.long)

        outputs: List[torch.Tensor] = model._sample_orthus(
            input_ids=input_ids,
            image_latents=None,
            logits_processor=LogitsProcessorList(),
            stopping_criteria=StoppingCriteriaList(),
            generation_config=_build_generation_config(),
            synced_gpus=False,
            streamer=None,
            logits_warper=None,
            interleave_output_format=True,
            cfg_scale=1.0,
        )

        self.assertIsInstance(outputs, list)
        self.assertEqual(outputs[0].item(), 42)
        self.assertEqual(outputs[1].item(), 8197)
        self.assertEqual(outputs[-1].item(), 8710)

        latents = [tensor for tensor in outputs[2:-1] if tensor.ndim == 2]
        self.assertEqual(len(latents), model.latents_per_image)
        self.assertTrue(all(tensor.shape == torch.Size([1, 256]) for tensor in latents))
        self.assertTrue(torch.allclose(latents[0], torch.full((1, 256), 1.0)))
        self.assertTrue(torch.allclose(latents[-1], torch.full((1, 256), float(model.latents_per_image))))

    def test_sample_orthus_cfg_returns_text_and_latents(self):
        model = _FakeOrthusModel()
        input_ids = torch.zeros((1, 1), dtype=torch.long)
        generation_config = _build_generation_config()

        outputs: List[torch.Tensor] = model._sample_orthus_cfg(
            input_ids_list=[input_ids.clone(), input_ids.clone()],
            image_latents_list=[None, None],
            logits_processor_list=[LogitsProcessorList(), LogitsProcessorList()],
            stopping_criteria_list=[StoppingCriteriaList(), StoppingCriteriaList()],
            generation_config_list=[generation_config, generation_config],
            synced_gpus_list=[False, False],
            streamer_list=[None, None],
            logits_warper_list=[None, None],
            model_kwargs_list=[{"cfg_scale": 3.0, "interleave_output_format": True}, {"cfg_scale": 3.0}],
        )

        self.assertIsInstance(outputs, list)
        self.assertEqual(outputs[0].item(), 42)
        self.assertEqual(outputs[1].item(), 8197)
        self.assertEqual(outputs[-1].item(), 8710)
        latents = [tensor for tensor in outputs[2:-1] if tensor.ndim == 2]
        self.assertEqual(len(latents), model.latents_per_image)
        self.assertTrue(all(tensor.shape == torch.Size([1, 256]) for tensor in latents))


if __name__ == "__main__":
    unittest.main()
