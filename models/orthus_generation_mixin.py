import copy
from typing import Any, Dict, List, Optional, Union

import torch

from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
    GenerationConfig,
)


class OrthusGenerationMixin:
    def _sample_orthus_cfg(
        self,
        input_ids_list: List[torch.LongTensor],
        image_latents_list: List[torch.LongTensor],
        logits_processor_list: List[LogitsProcessorList],
        stopping_criteria_list: List[StoppingCriteriaList],
        generation_config_list: List[GenerationConfig],
        synced_gpus_list: List[bool],
        streamer_list: List[Optional[BaseStreamer]],
        logits_warper_list: List[Optional[LogitsProcessorList]],
        model_kwargs_list: List[Dict[str, Any]],
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor, List[torch.Tensor]]:
        use_cfg = len(input_ids_list) > 1

        input_ids = input_ids_list[0]
        model_kwargs = model_kwargs_list[0]

        if use_cfg:
            input_id_uncon = copy.deepcopy(input_ids_list[1])
            model_kwargs_uncon = copy.deepcopy(model_kwargs_list[1])

        max_length = generation_config_list[0].max_length
        this_peer_finished = False
        batch_size, cur_len = input_ids_list[0].shape
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids_list[0].device)
        synced_gpus = synced_gpus_list[0] if synced_gpus_list else None

        do_sample_list = [generation_config.do_sample for generation_config in generation_config_list]
        for do_sample, logits_warper in zip(do_sample_list, logits_warper_list):
            if do_sample and logits_warper is not None and not isinstance(logits_warper, LogitsProcessorList):
                raise ValueError(
                    "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                    f"{logits_warper})."
                )
        model_kwargs_list = [
            self._get_initial_cache_position(ids_branch.shape[-1], ids_branch.device, kwargs_branch)
            for ids_branch, kwargs_branch in zip(input_ids_list, model_kwargs_list)
        ]
        interleave_output_format = model_kwargs_list[0].get("interleave_output_format", False)
        if interleave_output_format:
            interleave_output_list: List[torch.Tensor] = []

        mode = "discrete"
        image_latents = None
        collect_image_latents: List[torch.Tensor] = []

        generate_eoi = False
        sum_image_latents_generated = 0

        cfg_flag = False

        intervention_indices_list = model_kwargs.pop("intervention_indices", None)
        target_latents_list = model_kwargs.pop("target_latents_for_intervention", None)
        current_image_index = -1
        is_new_image_generation = False

        streamer = streamer_list[0] if streamer_list else None

        while cur_len < max_length and self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if use_cfg and cfg_flag:
                model_inputs_uncon = self.prepare_inputs_for_generation(input_id_uncon, **model_kwargs_uncon)
                if len(collect_image_latents) == 0:
                    outputs, outputs_uncon = self(
                        **model_inputs,
                        return_dict=True,
                        mode=mode,
                        cfg_scale=model_kwargs["cfg_scale"],
                        logits_processor=logits_processor_list[0],
                        logits_warper=logits_warper_list[0],
                        diff_pos_id=sum_image_latents_generated % 1024,
                        model_inputs_uncon=model_inputs_uncon,
                    )
                else:
                    image_latents = torch.stack(collect_image_latents, dim=1)
                    outputs, outputs_uncon = self(
                        **model_inputs,
                        image_latents=image_latents,
                        return_dict=True,
                        mode=mode,
                        logits_processor=logits_processor_list[0],
                        logits_warper=logits_warper_list[0],
                        diff_pos_id=sum_image_latents_generated % 1024,
                        cfg_scale=model_kwargs["cfg_scale"],
                        model_inputs_uncon=model_inputs_uncon,
                    )
            else:
                if len(collect_image_latents) == 0:
                    outputs = self(
                        **model_inputs,
                        return_dict=True,
                        mode=mode,
                        logits_processor=logits_processor_list[0],
                        logits_warper=logits_warper_list[0],
                        diff_pos_id=sum_image_latents_generated % 1024,
                    )
                else:
                    image_latents = torch.stack(collect_image_latents, dim=1)
                    outputs = self(
                        **model_inputs,
                        image_latents=image_latents,
                        return_dict=True,
                        mode=mode,
                        logits_processor=logits_processor_list[0],
                        logits_warper=logits_warper_list[0],
                        diff_pos_id=sum_image_latents_generated % 1024,
                    )

            if mode == "discrete":
                next_token_logits = outputs.logits[:, -1, :]
                next_token_logits = logits_processor_list[0](input_ids, next_token_logits)
                if do_sample_list[0]:
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                if interleave_output_format:
                    interleave_output_list.append(next_tokens)

                if torch.sum(next_tokens == 8197) == next_tokens.shape[0]:
                    mode = "continuous"
                    cfg_flag = True
                    is_new_image_generation = True

                    if use_cfg:
                        input_id_uncon = copy.deepcopy(input_ids_list[1])
                        model_kwargs_uncon = copy.deepcopy(model_kwargs_list[1])
                        model_inputs_uncon = self.prepare_inputs_for_generation(input_id_uncon, **model_kwargs_uncon)
                        outputs_uncon = self(
                            **model_inputs_uncon,
                            return_dict=True,
                            mode="discrete",
                            cfg_scale=model_kwargs["cfg_scale"],
                            logits_processor=logits_processor_list[1],
                            logits_warper=logits_warper_list[1],
                            diff_pos_id=sum_image_latents_generated % 1024,
                        )
            else:
                if is_new_image_generation:
                    current_image_index += 1
                    is_new_image_generation = False

                next_image_latents = outputs.next_image_latents
                patch_index_within_image = sum_image_latents_generated % 1024
                should_intervene = False

                if (
                    intervention_indices_list is not None
                    and target_latents_list is not None
                    and 0 <= current_image_index < len(intervention_indices_list)
                ):
                    current_intervention_indices = intervention_indices_list[current_image_index]
                    if patch_index_within_image in current_intervention_indices:
                        should_intervene = True

                if should_intervene:
                    current_target_latents = target_latents_list[current_image_index].view(1024, -1)
                    correct_latent = current_target_latents[patch_index_within_image].unsqueeze(0)
                    collect_image_latents.append(correct_latent)
                else:
                    collect_image_latents.append(next_image_latents)

                sum_image_latents_generated += 1
                next_tokens = torch.tensor([8711]).to(input_ids.device)

                if sum_image_latents_generated % 1024 == 0 and not interleave_output_format:
                    generate_eoi = True
                    mode = "discrete"
                    return torch.stack(collect_image_latents, dim=1)[0]
                elif sum_image_latents_generated % 1024 == 0 and interleave_output_format:
                    next_tokens = torch.tensor([8711]).to(input_ids.device)
                    mode = "discrete"
                    interleave_output_list.append(next_image_latents)
                    cfg_flag = False
                    collect_image_latents = []
                elif sum_image_latents_generated % 1024 != 0 and interleave_output_format:
                    next_tokens = torch.tensor([8711]).to(input_ids.device)
                    interleave_output_list.append(next_image_latents)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if torch.sum(next_tokens == 8710) > 0:
                if not interleave_output_format:
                    return input_ids
                return interleave_output_list

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            if use_cfg and cfg_flag:
                input_id_uncon = torch.cat([input_id_uncon, next_tokens[:, None]], dim=-1)
                model_kwargs_uncon = self._update_model_kwargs_for_generation(
                    outputs_uncon,
                    model_kwargs_uncon,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria_list[0](input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            del outputs

        if not interleave_output_format:
            return input_ids
        return interleave_output_list

    def _sample_orthus(
        self,
        input_ids: torch.LongTensor,
        image_latents: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional[BaseStreamer],
        logits_warper: Optional[LogitsProcessorList],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor, List[torch.Tensor]]:
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)

        do_sample = generation_config.do_sample
        if do_sample and logits_warper is not None and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids.shape[-1], input_ids.device, model_kwargs)

        interleave_output_format = model_kwargs.get("interleave_output_format", False)
        if interleave_output_format:
            interleave_output_list: List[torch.Tensor] = []
        mode = "discrete"

        collect_image_latents: List[torch.Tensor] = []
        if image_latents is None:
            collect_image_latents = []
        else:
            image_latents = image_latents.view(image_latents.shape[0], -1, 256)
            collect_image_latents = [image_latent.squeeze(1) for image_latent in torch.split(image_latents, 1, dim=1)]

        generate_eoi = False
        sum_image_latents_generated = 0

        while cur_len < max_length and self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if len(collect_image_latents) == 0:
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    mode=mode,
                    cfg_scale=model_kwargs["cfg_scale"],
                    logits_processor=logits_processor,
                    logits_warper=logits_warper,
                    diff_pos_id=sum_image_latents_generated % 1024,
                )
            else:
                image_latents = torch.stack(collect_image_latents, dim=1)
                outputs = self(
                    **model_inputs,
                    image_latents=image_latents,
                    return_dict=True,
                    mode=mode,
                    logits_processor=logits_processor,
                    logits_warper=logits_warper,
                    diff_pos_id=sum_image_latents_generated % 1024,
                )

            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_logits,)
                if output_logits:
                    raw_logits += (outputs.logits[:, -1, :],)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            if do_sample:
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            if interleave_output_format:
                interleave_output_list.append(next_tokens)

            if mode == "discrete":
                if torch.sum(next_tokens == 8197) == next_tokens.shape[0]:
                    mode = "continuous"
            elif mode == "continuous":
                next_image_latents = outputs.next_image_latents
                collect_image_latents.append(next_image_latents)
                sum_image_latents_generated += 1
                next_tokens = torch.tensor([8711, 8711]).to(input_ids.device)

                if sum_image_latents_generated % 1024 == 0 and not interleave_output_format:
                    generate_eoi = True
                    mode = "discrete"
                    return torch.stack(collect_image_latents, dim=1)[0]
                elif sum_image_latents_generated % 1024 == 0 and interleave_output_format:
                    next_tokens = torch.tensor([8711]).to(input_ids.device)
                    mode = "discrete"
                    interleave_output_list.append(next_image_latents)
                elif sum_image_latents_generated % 1024 != 0 and interleave_output_format:
                    next_tokens = torch.tensor([8711]).to(input_ids.device)
                    interleave_output_list.append(next_image_latents)
            else:
                raise ValueError("Unknown multimodal generation mode.")

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if torch.sum(next_tokens == 8710) > 0:
                if not interleave_output_format:
                    return input_ids
                return interleave_output_list
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        if not interleave_output_format:
            return input_ids
        return interleave_output_list
