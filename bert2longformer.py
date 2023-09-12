import copy
from collections import OrderedDict
from tempfile import TemporaryDirectory

import torch
from transformers import LongformerConfig, LongformerModel, LongformerSelfAttention


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


def convert_bert_to_longformer(
    bert_model,
    bert_tokenizer,
    longformer_max_length: int = 4096,
    attention_window: int = 512,
    max_copy_from_index: int = 512,
    generate_new_positions: str = "",  # "", "sinosoidal", "interpolate"
):
    config = bert_model.config
    # extend position embeddings
    bert_tokenizer.model_max_length = longformer_max_length
    bert_tokenizer.init_kwargs["model_max_length"] = longformer_max_length
    (
        current_max_pos,
        embed_size,
    ) = bert_model.embeddings.position_embeddings.weight.shape
    config.max_position_embeddings = longformer_max_length
    assert longformer_max_length > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = bert_model.embeddings.position_embeddings.weight.new_empty(
        longformer_max_length, embed_size
    )
    # print(new_pos_embed.shape)
    # print(bert_model.bert.embeddings.position_embeddings)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 0
    step = current_max_pos
    while k < longformer_max_length - 1:
        new_pos_embed[k : (k + step)] = bert_model.embeddings.position_embeddings.weight
        k += step

    bert_model.embeddings.position_ids = torch.tensor(
        [i for i in range(longformer_max_length)]
    ).reshape(1, longformer_max_length)

    bert_model.embeddings.position_embeddings = torch.nn.Embedding.from_pretrained(
        new_pos_embed
    )

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(bert_model.encoder.layer):
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

    with TemporaryDirectory() as temp_dir:
        bert_model.save_pretrained(temp_dir)
        longformer_model = LongformerModel.from_pretrained(temp_dir)

    return longformer_model, bert_tokenizer  # , new_pos_embed


# TODO: uniform convertion in roberta2longformer
def _future_convert_bert_to_longformer(
    bert_model,
    bert_tokenizer,
    longformer_max_length: int = 4096,
    attention_window: int = 512,
    max_copy_from_index: int = 514,
    generate_new_positions: str = "",  # "", "sinosoidal", "interpolate"
):
    ##################################
    # Create new longformer instance #
    ##################################

    bert_config = bert_model.config.to_dict()
    bert_config.pop("_name_or_path")
    bert_config.pop("architectures")
    longformer_config = LongformerConfig.from_dict(bert_config)
    longformer_config.max_position_embeddings = longformer_max_length
    longformer_config.attention_window = attention_window

    longformer_model = LongformerModel(longformer_config)

    ###############################
    # Create longformer tokenizer #
    ###############################

    # Longformer tokenizers are Roberta tokenizers.
    # But to follow the conventions
    # and to avoid confusion we create a
    # longformer tokenizer class with the state of
    # the original tokenizer.
    # TODO: Bert tokenizer does not have the same size as BertConfig model (intfloat/multilingual-e5-small)
    # with TemporaryDirectory() as temp_dir:
    #     bert_tokenizer.model_max_length = longformer_max_length
    #     bert_tokenizer.save_pretrained(temp_dir)
    #     longformer_tokenizer = LongformerTokenizerFast.from_pretrained(temp_dir)

    bert_tokenizer.model_max_length = longformer_max_length
    longformer_tokenizer = bert_tokenizer
    # longformer_tokenizer.init_kwargs["model_max_length"] = longformer_max_length

    ######################
    # Copy model weights #
    ######################

    # We only copy the encoder weights and resize the embeddings.
    # Pooler weights are kept untouched.

    # ---------#
    # Encoder  #
    # ---------#
    bert_parameters = bert_model.encoder.state_dict()
    longformer_parameters = longformer_model.encoder.state_dict()

    # Load all compatible keys directly and obtain missing keys to handle later
    errors = longformer_model.encoder.load_state_dict(bert_parameters, strict=False)
    assert not errors.unexpected_keys, "Found unexpected keys"
    missing_keys = errors.missing_keys

    # We expect, the keys to be the weights of the global attention modules and
    # reuse roberta's normal attention weights for those modules.
    for longformer_key in missing_keys:
        # Resolve layer properties
        (
            prefix,
            layer_idx,
            layer_class,
            layer_type,
            target,
            params,
        ) = longformer_key.split(".")
        assert layer_class == "attention" or target.endswith(
            "global"
        ), f"Unexpected parameters {longformer_key}."
        # Copy the normal weights attention weights to the global attention layers too
        bert_target_key = ".".join(
            [
                prefix,
                layer_idx,
                layer_class,
                layer_type,
                target.removesuffix("_global"),
                params,
            ]
        )
        bert_weights = bert_parameters[bert_target_key]
        longformer_parameters[longformer_key] = bert_weights

    # Update the state of the longformer model
    longformer_model.encoder.load_state_dict(longformer_parameters, strict=True)

    # ------------#
    # Embeddings  #
    # ------------#
    # There are two types of embeddings:

    # 1. Token embeddings
    # We can simply copy the token embeddings.

    # We have to resize the token embeddings upfront, to make load_state_dict work.
    longformer_model.resize_token_embeddings(len(bert_tokenizer))

    bert_embeddings_parameters = bert_model.embeddings.state_dict()
    embedding_parameters2copy = []

    for key, item in bert_embeddings_parameters.items():
        if "position" not in key:  # and not "token_type_embeddings" in key:
            # if not "position" in key and not "token_type_embeddings" in key:
            embedding_parameters2copy.append((key, item))

    # 2. Positional embeddings
    # The positional embeddings are repeatedly copied over
    # to longformer to match the new `max_seq_length`.
    bert_pos_embs = bert_model.embeddings.state_dict()["position_embeddings.weight"]

    assert (
        bert_pos_embs.size(0) < longformer_max_length
    ), "Longformer sequence length has to be longer than bert original sequence length"

    # Figure out how many time we need to copy the original embeddings
    n_copies = round(longformer_max_length / bert_pos_embs.size(0))

    # Copy the embeddings and handle the last missing ones.
    longformer_pos_embs = bert_pos_embs.repeat((n_copies, 1))
    n_pos_embs_left = longformer_max_length - longformer_pos_embs.size(0)
    longformer_pos_embs = torch.cat(
        [longformer_pos_embs, bert_pos_embs[:n_pos_embs_left]], dim=0
    )

    embedding_parameters2copy.append(
        ("position_embeddings.weight", longformer_pos_embs)
    )

    # Load the embedding weights into the longformer model
    embedding_parameters2copy = OrderedDict(embedding_parameters2copy)
    longformer_model.embeddings.load_state_dict(embedding_parameters2copy, strict=False)

    # # resize_token_embeddings reset padding_idx
    # if getattr(bert_model.embeddings.word_embeddings, "padding_idx"):
    #     longformer_model.embeddings.word_embeddings.padding_idx = (
    #         bert_model.embeddings.word_embeddings.padding_idx
    #     )

    return longformer_model, longformer_tokenizer
