from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

class Router(nn.Module):
    def __init__(self, config, prefix_config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.prefix_config = prefix_config
        if prefix_config is not None:
            self.temperature = self.prefix_config['temperature']
            self.task_id = 0
            self.router = nn.Parameter(data=torch.empty((
                prefix_config['n_tasks'],
                config.num_hidden_layers,
                prefix_config['n_prompts']
            )).uniform_(-1e-3, 1e-3))

            self.z = nn.Parameter(data=torch.empty((
                config.num_hidden_layers,
                prefix_config['n_prompts'],
                prefix_config['intrinsic_dim']
            )).uniform_(-1e-3, 1e-3))

            bound = 1 / math.sqrt(prefix_config['n_prompt_tokens'] * config.hidden_size)
            self.A = nn.Parameter(data=torch.empty((
                config.num_hidden_layers,
                prefix_config['intrinsic_dim'],
                prefix_config['n_prompt_tokens'] * config.hidden_size
            )).uniform_(-bound, bound))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.prefix_config is not None:
            hidden_shape = hidden_states.shape
            router = self.router[self.task_id].squeeze()  # layer * n_prompts
            if self.training:
                router = RelaxedBernoulli(temperature=self.temperature, logits=router).rsample()  # layer * n_prompts
            else:
                router = torch.sigmoid(router)  # layer * n_prompts
            router = (router / (router.sum(dim=-1, keepdim=True) + 1e-12)).unsqueeze(1)  # layer * 1 * n_prompts
            prompt = torch.bmm(self.z, self.A) if not hasattr(self, 'prompt') else self.prompt
            prompt_embedding = torch.bmm(router, prompt).view(self.config.num_hidden_layers, -1, self.config.hidden_size)  # layer * n_prompt_token * hidden

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.prefix_config is not None:
                prompt_padding = torch.zeros(size=(hidden_shape[0], hidden_shape[1] - self.prefix_config['n_prompt_tokens'] - 1, self.config.hidden_size), device='cuda:0')
                extended_prompt_embedding = torch.cat([prompt_embedding[i].tile(hidden_shape[0], 1, 1), prompt_padding], dim=1)
                pre_padding = torch.zeros(size=(hidden_shape[0], 1, self.config.hidden_size), device='cuda:0')
                extended_prompt_embedding = torch.cat([pre_padding, extended_prompt_embedding], dim=1)  # for <CLS>
                # extended_prompt_embedding = extended_prompt_embedding.repeat(input_shape[0], 1, 1)
                hidden_states = hidden_states + extended_prompt_embedding

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

