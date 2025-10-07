    def _sample_orthus_cfg(
        self,
        input_ids_list: List[torch.LongTensor],
        image_latents_list: List[torch.LongTensor],
        logits_processor_list: List[LogitsProcessorList],
        stopping_criteria_list: List[StoppingCriteriaList],
        generation_config_list: List[GenerationConfig],
        synced_gpus_list: List[bool],
        streamer_list: List[Optional["BaseStreamer"]],
        logits_warper_list: List[Optional[LogitsProcessorList]],
        model_kwargs_list: List[Dict[str, Any]],
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        
        # init values
        max_length = generation_config_list[0].max_length
        this_peer_finished = False
        batch_size, cur_len = input_ids_list[0].shape
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids_list[0].device)
        synced_gpus=None

        do_sample_list=[generation_config.do_sample for generation_config in generation_config_list]
        for (do_sample,logits_warper) in zip(do_sample_list,logits_warper_list):
            if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
                raise ValueError(
                    "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                    f"{logits_warper})."
                )
        model_kwargs_list = [self._get_initial_cache_position(input_ids, model_kwargs) for input_ids, model_kwargs in zip(input_ids_list, model_kwargs_list)]
        interleave_output_format = model_kwargs_list[0].get('interleave_output_format', False)
        # Initialize special output format if we want to generate interleave data
        if interleave_output_format:
            # output_list consists of [output_ids, output_image_latents, output_ids, ...]
            interleave_output_list = []

        # generate discrete text token at prefill phase
        mode = 'discrete' #generate mode
        # collect_image_latents store the image latents in the prompt & generated image latents

        #no image in prompt, so collect_image_latents is empty
        image_latents=None
        collect_image_latents = []

        generate_eoi = False
        sum_image_latents_generated = 0

        input_ids=input_ids_list[0]
        model_kwargs=model_kwargs_list[0]
        do_sample=do_sample_list[0]

        cfg_flag=False

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if cfg_flag:
                model_inputs_uncon = self.prepare_inputs_for_generation(input_id_uncon, **model_kwargs_uncon)
                # forward pass to get next token(mode: discrete) or next image latents(mode: continuous)
                if len(collect_image_latents) == 0:

                    outputs,outputs_uncon = self(**model_inputs, return_dict=True, mode=mode, cfg_scale=model_kwargs['cfg_scale'], \
                                logits_processor=logits_processor_list[0], logits_warper=logits_warper_list[0], diff_pos_id=sum_image_latents_generated%1024, \
                                model_inputs_uncon=model_inputs_uncon)
                else:
                    image_latents = torch.stack(collect_image_latents, dim=1)
                    #print(f'image_latents: {image_latents.shape}')
                    outputs,outputs_uncon = self(**model_inputs, image_latents=image_latents, return_dict=True, mode=mode, \
                        logits_processor=logits_processor_list[0], logits_warper=logits_warper_list[0], diff_pos_id=sum_image_latents_generated%1024, \
                                cfg_scale=model_kwargs['cfg_scale'], model_inputs_uncon=model_inputs_uncon)
            else:
                # forward pass to get next token(mode: discrete) or next image latents(mode: continuous)
                if len(collect_image_latents) == 0:
                    outputs = self(**model_inputs, return_dict=True, mode=mode, cfg_scale=model_kwargs['cfg_scale'], \
                                logits_processor=logits_processor_list[0], logits_warper=logits_warper_list[0], diff_pos_id=sum_image_latents_generated%1024)
                else:
                    image_latents = torch.stack(collect_image_latents, dim=1)
                    #print(f'image_latents: {image_latents.shape}')
                    outputs = self(**model_inputs, image_latents=image_latents, return_dict=True, mode=mode, \
                        logits_processor=logits_processor_list[0], logits_warper=logits_warper_list[0], diff_pos_id=sum_image_latents_generated%1024, \
                                cfg_scale=model_kwargs['cfg_scale'])
            
            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # we refer to 'mode' to deal with different type of output
            if mode == 'discrete':
                # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].clone()

                # pre-process distribution
                next_token_scores = logits_processor_list[0](input_ids, next_token_logits)
                if do_sample:
                    next_token_scores = logits_warper_list[0](input_ids, next_token_scores)

                
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
                #print("next_tokens: ", next_tokens)
                #TODO: support batch_size > 1
                if torch.sum(next_tokens == 8197) == next_tokens.shape[0]:
                    print('Convert generation mode to predict image_latents...')
                    mode = 'continuous'
                    #unconditon branch begin , init vaule for cfg_generation
                    cfg_flag=True
                    input_id_uncon = copy.deepcopy(input_ids_list[1])
                    model_kwargs_uncon = copy.deepcopy(model_kwargs_list[1])
                    #补上boi 和interleave分支对齐
                    model_inputs_uncon = self.prepare_inputs_for_generation(input_id_uncon, **model_kwargs_uncon)
                    outputs_uncon = self(**model_inputs_uncon, return_dict=True, mode="discrete", cfg_scale=model_kwargs['cfg_scale'], \
                        logits_processor=logits_processor_list[1], logits_warper=logits_warper_list[1], diff_pos_id=sum_image_latents_generated%1024)


            elif mode == 'continuous':
                next_image_latents = outputs.next_image_latents
                collect_image_latents.append(next_image_latents)
                sum_image_latents_generated +=1
                # we use <image>(token_id: 8711) to represent image_token, bsz=2 for cfg_generation
                next_tokens = torch.tensor([8711]).to(input_ids.device)
                
                #CHANGE_for_pix2pix
                #next_tokens = torch.tensor([8711, 8711,8711]).to(input_ids.device)
                
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

                    #unconditon branch finish
                    cfg_flag=False

                elif sum_image_latents_generated%1024 != 0 and interleave_output_format:
                    # we do not support cfg for interleave generation yet
                    next_tokens = torch.tensor([8711]).to(input_ids.device)
                    interleave_output_list.append(next_image_latents)
                    
            else: 
                raise ValueError(
                    "Unknown multimodal generation mode."
                )



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

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            if cfg_flag:
                input_id_uncon = torch.cat([input_id_uncon, next_tokens[:, None]], dim=-1)
                model_kwargs_uncon = self._update_model_kwargs_for_generation(
                    outputs_uncon,
                    model_kwargs_uncon,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria_list[0](input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
#while end


        if not interleave_output_format:
            return input_ids
        else: 
            return interleave_output_list