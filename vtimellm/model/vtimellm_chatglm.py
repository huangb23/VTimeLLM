import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers import AutoConfig, AutoModelForCausalLM
from .chatglm import ChatGLMConfig, ChatGLMModel, ChatGLMForConditionalGeneration
from .vtimellm_arch import VTimeLLMMetaModel, VTimeLLMMetaForCausalLM

class VTimeLLMChatGLMConfig(ChatGLMConfig):
    model_type = "VTimeLLM_ChatGLM"

class VTimeLLMChatGLMModel(ChatGLMModel, VTimeLLMMetaModel):
    config_class = VTimeLLMChatGLMConfig

    def __init__(self, config, empty_init=True, device=None):
        super(VTimeLLMChatGLMModel, self).__init__(config, empty_init=empty_init, device=device)

class VTimeLLMChatGLMForCausalLM(ChatGLMForConditionalGeneration, VTimeLLMMetaForCausalLM):
    config_class = VTimeLLMChatGLMConfig

    def __init__(self, config, empty_init=True, device=None):
        super(ChatGLMForConditionalGeneration, self).__init__(config)
        self.transformer = VTimeLLMChatGLMModel(config, empty_init=empty_init, device=device)
        self.max_sequence_length = config.max_length
        self.config = config
        self.quantized = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.transformer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_last_logit: Optional[bool] = False,
        images: Optional[torch.FloatTensor] = None,
    ):

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("VTimeLLM_ChatGLM", VTimeLLMChatGLMConfig)
AutoModelForCausalLM.register(VTimeLLMChatGLMConfig, VTimeLLMChatGLMForCausalLM)
