    def _sample_orthus(
        self,
        input_ids: torch.LongTensor,
        image_latents: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        logits_warper: Optional[LogitsProcessorList],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)

        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )
        print("return_dict_in_generate",return_dict_in_generate)

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        print("five",scores,raw_logits,decoder_attentions,cross_attentions,decoder_hidden_states)
        

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        
        interleave_output_format = model_kwargs.get('interleave_output_format', False)
        print(f'interleave_output_format: {interleave_output_format}')
        # Initialize special output format if we want to generate interleave data
        if interleave_output_format:
            # output_list consists of [output_ids, output_image_latents, output_ids, ...]
            interleave_output_list = []
        # generate discrete text token at prefill phase
        mode = 'discrete' #generate mode
        # collect_image_latents store the image latents in the prompt & generated image latents
        if image_latents == None: #there is no image in the prompt
            collect_image_latents = []
        else:
            image_latents = image_latents.view(image_latents.shape[0], -1, 256)
            #CHANGE_for_pix2pix
            #collect_image_latents = [image_latent.squeeze(1).repeat(3,1) for image_latent in torch.split(image_latents, 1, dim=1)]
            collect_image_latents = [image_latent.squeeze(1) for image_latent in torch.split(image_latents, 1, dim=1)]

        generate_eoi = False
        sum_image_latents_generated = 0
        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            # print("output_attentions",output_attentions,"output_hidden_states",output_hidden_states)


            # forward pass to get next token(mode: discrete) or next image latents(mode: continuous)
            if len(collect_image_latents) == 0:
                #CHANGE_for_pix2pix
                # outputs = self(**model_inputs, return_dict=True, mode=mode, cfg_scale=model_kwargs['cfg_scale'],cfg_text=model_kwargs['cfg_text'],cfg_image=model_kwargs['cfg_image'], \
                #                 logits_processor=logits_processor, logits_warper=logits_warper, diff_pos_id=sum_image_latents_generated%1024)
                outputs = self(**model_inputs, return_dict=True, mode=mode, cfg_scale=model_kwargs['cfg_scale'], \
                               logits_processor=logits_processor, logits_warper=logits_warper, diff_pos_id=sum_image_latents_generated%1024)
            else:
                image_latents = torch.stack(collect_image_latents, dim=1)
                #print(f'image_latents: {image_latents.shape}')
                #CHANGE_for_pix2pix
                # outputs = self(**model_inputs, image_latents=image_latents, return_dict=True, mode=mode, \
                #      logits_processor=logits_processor, logits_warper=logits_warper, diff_pos_id=sum_image_latents_generated%1024, \
                #                 cfg_scale=model_kwargs['cfg_scale'],cfg_text=model_kwargs['cfg_text'],cfg_image=model_kwargs['cfg_image'])
                outputs = self(**model_inputs, image_latents=image_latents, return_dict=True, mode=mode, \
                    logits_processor=logits_processor, logits_warper=logits_warper, diff_pos_id=sum_image_latents_generated%1024, \
                               cfg_scale=model_kwargs['cfg_scale'])
            
            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # we refer to 'mode' to deal with different type of output
            if mode == 'discrete':
                # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].clone()

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)
                if do_sample:
                    next_token_scores = logits_warper(input_ids, next_token_scores)

                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)
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

                # token selection
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)
                
                if interleave_output_format:
                    interleave_output_list.append(next_tokens)
                # <boi> is the sign for mode changing, only support batch_size=1 for now
                #TODO: support batch_size > 1
                if torch.sum(next_tokens == 8197) == next_tokens.shape[0]:
                    print('Convert generation mode to predict image_latents...')
                    mode = 'continuous'

            elif mode == 'continuous':
                next_image_latents = outputs.next_image_latents
                collect_image_latents.append(next_image_latents)
                sum_image_latents_generated +=1
                # we use <image>(token_id: 8711) to represent image_token, bsz=2 for cfg_generation
                next_tokens = torch.tensor([8711, 8711]).to(input_ids.device)
                
                #CHANGE_for_pix2pix
                # next_tokens = torch.tensor([8711, 8711,8711]).to(input_ids.device)
                
                # if image_latents_generated%1024 ==0, use last_image_latents to generate discrete token <eoi>
                if sum_image_latents_generated%1024 == 0 and not interleave_output_format:
                    generate_eoi = True
                    mode = 'discrete'
                    # only support bsz=1 inputs
                    
                    return torch.stack(collect_image_latents, dim=1)[0]
                elif sum_image_latents_generated%1024 == 0 and interleave_output_format:
                    # we do not support cfg for interleave generation yet
                    next_tokens = torch.tensor([8711]).to(input_ids.device)
                    mode = 'discrete' # able to generate <eoi> token
                    print('Convert generation mode to predict text_tokens...')
                    interleave_output_list.append(next_image_latents)
                elif sum_image_latents_generated%1024 != 0 and interleave_output_format:
                    # we do not support cfg for interleave generation yet
                    next_tokens = torch.tensor([8711]).to(input_ids.device)
                    interleave_output_list.append(next_image_latents)
                    
            else: 
                raise ValueError(
                    "Unknown multimodal generation mode."
                )

            # finished sentences should have their next token be a padding token
            #print("has_eos_stopping_criteria",has_eos_stopping_criteria)
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

#             print(next_tokens)
            # Update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
#             print(f'interleave_output_list: {interleave_output_list}')
            # <SEP> token(8710) generated, stop generating
            if torch.sum(next_tokens==8710) > 0:
                print(f'Token_id 8710(<SEP>) generated, generation ended successfully.')
                if not interleave_output_format:
                    return input_ids
                else:
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

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
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
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        elif not interleave_output_format:
            return input_ids
        else: 
            return interleave_output_list
