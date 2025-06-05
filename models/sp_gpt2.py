import torch
from .pez import PEZSoftPrompt
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Dict, Optional, Union, Tuple


class PTuningGPT2Config(GPT2Config):
    model_type = "ptuning_gpt2"

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=0.00001,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        base_model_name="gpt2",
        n_prefix_tokens: int = 8,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            n_positions,
            n_embd,
            n_layer,
            n_head,
            n_inner,
            activation_function,
            resid_pdrop,
            embd_pdrop,
            attn_pdrop,
            layer_norm_epsilon,
            initializer_range,
            summary_type,
            summary_use_proj,
            summary_activation,
            summary_proj_to_labels,
            summary_first_dropout,
            scale_attn_weights,
            use_cache,
            bos_token_id,
            eos_token_id,
            scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn,
            **kwargs,
        )

        self.n_prefix_tokens = n_prefix_tokens
        self.base_model_name = base_model_name


class PTuningGPT2LMHeadModel(GPT2LMHeadModel):
    config: PTuningGPT2Config

    def __init__(self, config: PTuningGPT2Config):
        super().__init__(config)
        print(f"[SpwmGPT2LMHead] Loading base model: {config.base_model_name}")
        self.transformer = GPT2Model(config)
        self.soft_prompt = PEZSoftPrompt(config.n_prefix_tokens, config.n_embd)

        self.post_init()
        assert torch.allclose(
            self.get_input_embeddings().weight.data,
            self.get_output_embeddings().weight.data,
        )

    def get_sp_embeddings(self) -> torch.Tensor:
        return self.soft_prompt()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        n_prefix_tokens = self.config.n_prefix_tokens

        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        # reset attention mask to cover soft-prompts
        attention_mask = (
            torch.ones(input_ids.size(0), n_prefix_tokens + input_ids.size(1))
            .long()
            .to(input_ids.device)
        )

        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "do_generate": True,
            }
        )

        return model_inputs

    def save_soft_prompts(self, save_path: str):
        state_dict = self.soft_prompt.state_dict()
        torch.save(state_dict, save_path)

    def load_soft_prompts(self, load_path: str):
        state_dict = torch.load(load_path)
        self.soft_prompt.load_state_dict(state_dict)

    def set_soft_prompts(self, state_dict: Dict):
        self.soft_prompt.load_state_dict(state_dict)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        do_generate: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        if do_generate:
            # generation
            if past_key_values is None:
                # append sp embeddings at the first step
                B = input_ids.size(0)
                sp_embeds = self.get_sp_embeddings().repeat(B, 1, 1)
                inputs_embeds = self.transformer.get_input_embeddings()(input_ids)

                # B, (N + L), H
                inputs_embeds = torch.cat([sp_embeds, inputs_embeds], dim=1)
                input_ids = None
            else:
                # otherwise we just follow standard procedure
                # here the input should be the last token id
                # along with previous previous_key_values
                assert input_ids is not None
                inputs_embeds = None
        else:
            # training, simply append sp embeddings
            assert inputs_embeds is None

            B = input_ids.size(0)
            sp_embeds = self.get_sp_embeddings().repeat(B, 1, 1)
            inputs_embeds = self.transformer.get_input_embeddings()(input_ids)

            # B, (N + L), H
            inputs_embeds = torch.cat([sp_embeds, inputs_embeds], dim=1)
            input_ids = None

            # update attention mask and labels to match soft-prompts
            attention_mask = torch.cat(
                [
                    torch.ones(B, self.config.n_prefix_tokens).long().to(inputs_embeds.device),
                    attention_mask,
                ],
                dim=1,
            )
            if labels is not None:
                labels = torch.cat(
                    [
                        torch.ones(B, self.config.n_prefix_tokens)
                        .fill_(-100)  # -100 will be ignored by CrossEntropyLoss
                        .long()
                        .to(inputs_embeds.device),
                        labels,
                    ],
                    dim=1,
                )

        return super().forward(
            input_ids,
            past_key_values,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
