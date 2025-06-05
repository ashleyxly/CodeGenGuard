import torch
import torch.nn as nn
from transformers import CodeGenForCausalLM, CodeGenConfig

from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Dict, Optional, Union, Tuple


class PTuningCodeGenConfig(CodeGenConfig):
    model_type = "spwm_codegen"

    def __init__(
        self,
        vocab_size=50400,
        n_positions=2048,
        n_ctx=2048,
        n_embd=4096,
        n_layer=28,
        n_head=16,
        rotary_dim=64,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        layer_norm_epsilon=0.00001,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=False,
        base_model_name=None,
        n_prefix_tokens: int = 64,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            n_positions,
            n_ctx,
            n_embd,
            n_layer,
            n_head,
            rotary_dim,
            n_inner,
            activation_function,
            resid_pdrop,
            embd_pdrop,
            attn_pdrop,
            layer_norm_epsilon,
            initializer_range,
            use_cache,
            bos_token_id,
            eos_token_id,
            tie_word_embeddings,
            **kwargs,
        )

        self.n_prefix_tokens = n_prefix_tokens
        self.base_model_name = base_model_name


class PTuningCodeGenForCausalLM(CodeGenForCausalLM):
    config: PTuningCodeGenConfig

    def __init__(self, config: PTuningCodeGenConfig):
        super().__init__(config)

        print(f"[SpwmCodeGenForCausalLM] Loading base model: {config.base_model_name}")
        self.transformer = CodeGenForCausalLM(config)
        self.soft_prompt = nn.Embedding(config.n_prefix_tokens, config.n_embd)

        self.post_init()

    def get_sp_embeddings(self) -> torch.Tensor:
        return self.soft_prompt()

    def save_soft_prompts(self, save_path: str):
        state_dict = self.soft_prompt.state_dict()
        torch.save(state_dict, save_path)

    def load_soft_prompts(self, load_path: str):
        state_dict = torch.load(load_path)
        self.soft_prompt.load_state_dict(state_dict)

    def set_soft_prompts(self, state_dict: Dict):
        self.soft_prompt.load_state_dict(state_dict)

    def get_reindexed_position_ids(
        self,
        input_ids: torch.Tensor,
        n_tokens_per_prefix: int = 64,
        n_prefixes: int = 1,
    ):
        tot_len = n_tokens_per_prefix + input_ids.size(1)
        main_position_ids = torch.arange(n_tokens_per_prefix, tot_len).long().unsqueeze(0)
        prefix_position_ids = torch.arange(n_tokens_per_prefix).long().unsqueeze(0)

        position_ids = (
            torch.cat([prefix_position_ids] * n_prefixes + [main_position_ids], dim=1)
            .long()
            .to(input_ids.device)
        )

        return position_ids

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values: torch.Tensor | None = None,
        **kwargs,
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        n_prefix_tokens = self.config.n_prefix_tokens

        # reset attention mask to cover soft-prompts
        attention_mask = (
            torch.ones(input_ids.size(0), n_prefix_tokens + input_ids.size(1))
            .long()
            .to(input_ids.device)
        )

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "do_generate": True,
        }

    def lm_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        do_generate: bool = False,
    ) -> Tuple | CausalLMOutputWithPast:
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

            # update attention mask and labels to match soft-promptsF
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
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
