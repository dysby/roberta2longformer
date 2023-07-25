import copy

import torch
from transformers import LongformerSelfAttention


class BertLongSelfAttention(LongformerSelfAttention):
    """
    from https://github.com/allenai/longformer/issues/215
    For transformers=4.12.5
    For transformers=4.26

    From XLMRobertaSelfAttention

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,

    to

    LongformerSelfAttention

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        [`LongformerSelfAttention`] expects *len(hidden_states)* to be multiple of *attention_window*. Padding to
        *attention_window* happens in [`LongformerModel.forward`] to avoid redoing the padding on each layer.

        The *attention_mask* is changed in [`LongformerModel.forward`] from 0, 1, 2 to:

            - -10000: no attention
            - 0: local attention
            - +10000: global attention
    """

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        # is_global_attn = any(is_index_global_attn.flatten()) PR #5811
        is_global_attn = is_index_global_attn.flatten().any().item()
        return super().forward(
            hidden_states,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )


class T3BertLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )


def bert2longformer(bert_model, bert_tokenizer, attention_window, max_pos):
    config = bert_model.config
    # extend position embeddings
    bert_tokenizer.model_max_length = max_pos
    bert_tokenizer.init_kwargs["model_max_length"] = max_pos
    (
        current_max_pos,
        embed_size,
    ) = bert_model.bert.embeddings.position_embeddings.weight.shape
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = bert_model.bert.embeddings.position_embeddings.weight.new_empty(
        max_pos, embed_size
    )
    # print(new_pos_embed.shape)
    # print(bert_model.bert.embeddings.position_embeddings)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 0
    step = current_max_pos
    while k < max_pos - 1:
        new_pos_embed[
            k : (k + step)
        ] = bert_model.bert.embeddings.position_embeddings.weight
        k += step

    bert_model.bert.embeddings.position_ids = torch.tensor(
        [i for i in range(max_pos)]
    ).reshape(1, max_pos)

    bert_model.bert.embeddings.position_embeddings = torch.nn.Embedding.from_pretrained(
        new_pos_embed
    )

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(bert_model.bert.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn
    # print(bert_model.bert.embeddings.position_ids.shape)
    # logger.info(f"saving model to {save_model_to}")
    # bert_model.save_pretrained(save_model_to)
    # bert_tokenizer.save_pretrained(save_model_to)
    return bert_model, bert_tokenizer  # , new_pos_embed


# def create_long_model(save_model_to, attention_window, max_pos):
#     model = RobertaForMaskedLM.from_pretrained("roberta-base")
#     tokenizer = RobertaTokenizerFast.from_pretrained(
#         "roberta-base", model_max_length=max_pos
#     )
#     config = model.config

#     # extend position embeddings
#     tokenizer.model_max_length = max_pos
#     tokenizer.init_kwargs["model_max_length"] = max_pos
#     (
#         current_max_pos,
#         embed_size,
#     ) = model.roberta.embeddings.position_embeddings.weight.shape
#     max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
#     config.max_position_embeddings = max_pos
#     assert max_pos > current_max_pos
#     # allocate a larger position embedding matrix
#     new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(
#         max_pos, embed_size
#     )
#     # copy position embeddings over and over to initialize the new position embeddings
#     k = 2
#     step = current_max_pos - 2
#     while k < max_pos - 1:
#         new_pos_embed[
#             k : (k + step)
#         ] = model.roberta.embeddings.position_embeddings.weight[2:]
#         k += step
#     model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed
#     model.roberta.embeddings.position_ids.data = torch.tensor(
#         [i for i in range(max_pos)]
#     ).reshape(1, max_pos)

#     # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
#     config.attention_window = [attention_window] * config.num_hidden_layers
#     for i, layer in enumerate(model.roberta.encoder.layer):
#         longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
#         longformer_self_attn.query = layer.attention.self.query
#         longformer_self_attn.key = layer.attention.self.key
#         longformer_self_attn.value = layer.attention.self.value

#         longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
#         longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
#         longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

#         layer.attention.self = longformer_self_attn

#     logger.info(f"saving model to {save_model_to}")
#     model.save_pretrained(save_model_to)
#     tokenizer.save_pretrained(save_model_to)
#     return model, tokenizer
