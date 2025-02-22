from collections import OrderedDict
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from transformers import LongformerConfig, LongformerModel, LongformerTokenizerFast


def generate_position_encoding(seq_len, d, n=10000):
    pos = (
        torch.arange(seq_len).float().unsqueeze(1)
    )  # Create positions [0, 1, ..., seq_len-1]
    div_term = torch.exp(
        torch.arange(0, d, 2).float() * (-torch.log(torch.tensor(n)).float() / d)
    )

    # Compute sine and cosine values using broadcasting
    pos_enc = torch.zeros(seq_len, d)
    pos_enc[:, 0::2] = torch.sin(pos * div_term)
    pos_enc[:, 1::2] = torch.cos(pos * div_term)

    return pos_enc


def interpolate_vectors(A, B, m):
    n = len(A)
    step_size = 1.0 / (m + 1)
    intermediate_vectors = []

    for i in range(1, m + 1):
        intermediate_vector = A + (B - A) * i * step_size
        intermediate_vectors.append(intermediate_vector)

    return np.row_stack(intermediate_vectors)


def generate_by_interpolation(absolute_positions):
    max_position = 4098
    m = 7
    # roberta's first position padding
    new_positional_embeddings = absolute_positions[:1, :]
    for i in range(2, absolute_positions.shape[0] - 1):
        mid_points = interpolate_vectors(
            absolute_positions[i, :], absolute_positions[i + 1, :], m
        )
        new_positional_embeddings = np.row_stack(
            [new_positional_embeddings, absolute_positions[i, :]]
        )
        new_positional_embeddings = np.row_stack(
            [new_positional_embeddings, mid_points]
        )
    # add last point
    new_positional_embeddings = np.row_stack(
        [new_positional_embeddings, absolute_positions[-1, :]]
    )

    # add missing positions (m=7)
    if new_positional_embeddings.shape[0] < max_position:
        new_positional_embeddings = np.row_stack(
            [
                new_positional_embeddings,
                np.vstack(
                    [absolute_positions[-1, :]]
                    * (max_position - new_positional_embeddings.shape[0])
                ),
            ]
        )
    # return only max_positions (m=8)
    return torch.Tensor(new_positional_embeddings[:max_position, :])


def convert_roberta_to_longformer(
    roberta_model,
    roberta_tokenizer,
    longformer_max_length: int = 4096,
    attention_window: int = 512,
    max_copy_from_index: int = 514,
    generate_new_positions: str = "",  # "", "sinosoidal", "interpolate"
):
    ##################################
    # Create new longformer instance #
    ##################################
    longformer_config = LongformerConfig(
        max_position_embeddings=longformer_max_length + 2,
        attention_window=attention_window,
        type_vocab_size=roberta_model.config.type_vocab_size,
        layer_norm_eps=1e-05,
        # hidden_size=roberta_model.config.hidden_size,
        # intermediate_size=roberta_model.config.intermediate_size,
        # vocab_size=roberta_model.config.vocab_size,  # did not work with
    )
    longformer_model = LongformerModel(longformer_config)

    ###############################
    # Create longformer tokenizer #
    ###############################

    # Longformer tokenizers are Roberta tokenizers.
    # But to follow the conventions
    # and to avoid confusion we create a
    # longformer tokenizer class with the state of
    # the original tokenizer.
    with TemporaryDirectory() as temp_dir:
        roberta_tokenizer.model_max_length = longformer_max_length
        roberta_tokenizer.save_pretrained(temp_dir)
        longformer_tokenizer = LongformerTokenizerFast.from_pretrained(temp_dir)
    # longformer_tokenizer.init_kwargs["model_max_length"] = longformer_max_length

    ######################
    # Copy model weights #
    ######################

    # We only copy the encoder weights and resize the embeddings.
    # Pooler weights are kept untouched.

    # ---------#
    # Encoder  #
    # ---------#
    roberta_parameters = roberta_model.encoder.state_dict()
    longformer_parameters = longformer_model.encoder.state_dict()

    # Load all compatible keys directly and obtain missing keys to handle later
    errors = longformer_model.encoder.load_state_dict(roberta_parameters, strict=False)
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
        roberta_target_key = ".".join(
            [
                prefix,
                layer_idx,
                layer_class,
                layer_type,
                target.removesuffix("_global"),
                params,
            ]
        )
        roberta_weights = roberta_parameters[roberta_target_key]
        longformer_parameters[longformer_key] = roberta_weights

    # Update the state of the longformer model
    longformer_model.encoder.load_state_dict(longformer_parameters, strict=True)

    # ------------#
    # Embeddings  #
    # ------------#
    # There are two types of embeddings:

    # 1. Token embeddings
    # We can simply copy the token embeddings.

    # We have to resize the token embeddings upfront, to make load_state_dict work.
    longformer_model.resize_token_embeddings(len(roberta_tokenizer))

    roberta_embeddings_parameters = roberta_model.embeddings.state_dict()
    embedding_parameters2copy = []

    for key, item in roberta_embeddings_parameters.items():
        if "position" not in key:  # and not "token_type_embeddings" in key:
            # if not "position" in key and not "token_type_embeddings" in key:
            embedding_parameters2copy.append((key, item))

    # 2. Positional embeddings
    # The positional embeddings are repeatedly copied over
    # to longformer to match the new `max_seq_length`.
    # In special models, it may be useful to copy only a part of the
    # Roberta position embeddings, by setting `max_copy_from_index`.
    # Because the base model may not be trained to the 514 full positions.
    # This happens with sentence transformers that were only trained
    # to sequence length of 128 tokens.

    roberta_pos_embs = roberta_model.embeddings.state_dict()[
        "position_embeddings.weight"
    ][2:max_copy_from_index]
    roberta_pos_embs_extra = roberta_model.embeddings.state_dict()[
        "position_embeddings.weight"
    ][:2]

    assert (
        roberta_pos_embs.size(0) < longformer_max_length
    ), "Longformer sequence length has to be longer than roberta original sequence length"

    # Figure out how many time we need to copy the original embeddings
    n_copies = round(longformer_max_length / roberta_pos_embs.size(0))

    # Copy the embeddings and handle the last missing ones.
    longformer_pos_embs = roberta_pos_embs.repeat((n_copies, 1))
    n_pos_embs_left = longformer_max_length - longformer_pos_embs.size(0)
    longformer_pos_embs = torch.cat(
        [longformer_pos_embs, roberta_pos_embs[:n_pos_embs_left]], dim=0
    )

    # Add the initial extra embeddings from original roberta model
    # Longformer transformers implementation always shift the position_ids from 2 as Roberta.
    # Here we divert from Bigbird and Nystronformer conversion by reserving the first positions embeddings.
    longformer_pos_embs = torch.cat(
        [roberta_pos_embs_extra, longformer_pos_embs], dim=0
    )

    # TODO: local test position_embedding_generation
    # generate_new_positions = "interpolate"

    # test generated position encoding
    if generate_new_positions == "sinosoidal":
        p = generate_position_encoding(4098, 768)
        longformer_pos_embs[max_copy_from_index:, :] = p[max_copy_from_index:, :]
    elif generate_new_positions == "interpolate":
        # TODO: restructure if interpolation is used because previous longformer position
        # embedding computation is not needed.
        absolute_position_embeddings = roberta_model.embeddings.state_dict()[
            "position_embeddings.weight"
        ]
        longformer_pos_embs = generate_by_interpolation(absolute_position_embeddings)

    embedding_parameters2copy.append(
        ("position_embeddings.weight", longformer_pos_embs)
    )

    # Load the embedding weights into the longformer model
    embedding_parameters2copy = OrderedDict(embedding_parameters2copy)
    longformer_model.embeddings.load_state_dict(embedding_parameters2copy, strict=False)

    # resize_token_embeddings reset padding_idx
    if getattr(roberta_model.embeddings.word_embeddings, "padding_idx"):
        longformer_model.embeddings.word_embeddings.padding_idx = (
            roberta_model.embeddings.word_embeddings.padding_idx
        )

    return longformer_model, longformer_tokenizer
