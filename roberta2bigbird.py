from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from transformers import (
    AutoTokenizer,
    BigBirdConfig,
    BigBirdModel,
    PreTrainedTokenizerBase,
    XLMRobertaModel,
    XLMRobertaTokenizerFast,
)


def convert_roberta_to_bigbird(
    roberta_model: XLMRobertaModel,
    roberta_tokenizer: XLMRobertaTokenizerFast,
    bigbird_max_length: int = 50176,
) -> Tuple[BigBirdModel, PreTrainedTokenizerBase]:
    """
    Note: In contrast to most other conversion functions, this function copies a model with language modeling head.
    """
    with TemporaryDirectory() as temp_dir:
        roberta_tokenizer.model_max_length = bigbird_max_length
        roberta_tokenizer.save_pretrained(temp_dir)
        bigbird_tokenizer = AutoTokenizer.from_pretrained(temp_dir)

    bigbird_config = BigBirdConfig.from_dict(roberta_model.config.to_dict())
    bigbird_config.max_position_embeddings = bigbird_max_length + 2
    bigbird_model = BigBirdModel(bigbird_config)

    # Copy encoder weights
    # bigbird_model.base_model.encoder.load_state_dict(roberta_model.base_model.encoder.state_dict(), strict=False)
    roberta_state_dict = roberta_model.state_dict()
    roberta_state_dict.pop("embeddings.position_ids")
    roberta_state_dict["embeddings.position_embeddings.weight"] = torch.rand(
        bigbird_config.max_position_embeddings, bigbird_config.hidden_size
    )
    bigbird_model.load_state_dict(roberta_state_dict, strict=False)

    # ------------#
    # Embeddings  #
    # ------------#
    # There are two types of embeddings:

    # 1. Token embeddings
    # We can simply copy the token embeddings.

    # We have to resize the token embeddings upfront, to make load_state_dict work.
    bigbird_model.resize_token_embeddings(len(roberta_tokenizer))

    roberta_embeddings_parameters = roberta_model.base_model.embeddings.state_dict()
    embedding_parameters2copy = []

    for key, item in roberta_embeddings_parameters.items():
        if not "position" in key and not "token_type_embeddings" in key:
            embedding_parameters2copy.append((key, item))

    # 2. Positional embeddings
    # The positional embeddings are repeatedly copied over
    # to bigbird to match the new max_seq_length
    # In Roberta Model positions [0, 1] are reserved.
    roberta_pos_embs = roberta_model.base_model.embeddings.state_dict()[
        "position_embeddings.weight"
    ][2:]
    roberta_pos_embs_extra = roberta_model.base_model.embeddings.state_dict()[
        "position_embeddings.weight"
    ][:2]

    assert (
        roberta_pos_embs.size(0) <= bigbird_max_length
    ), "bigbird sequence length has to be longer than roberta original sequence length"

    # Figure out how many time we need to copy the original embeddings
    n_copies = round(bigbird_max_length / roberta_pos_embs.size(0))

    # Copy the embeddings and handle the last missing ones.
    bigbird_pos_embs = roberta_pos_embs.repeat((n_copies, 1))
    n_pos_embs_left = bigbird_max_length - bigbird_pos_embs.size(0)
    bigbird_pos_embs = torch.cat(
        [bigbird_pos_embs, roberta_pos_embs[:n_pos_embs_left]], dim=0
    )

    # Add the extra embeddings.
    # Bigbird transformers implementation does not shift position_ids,
    # so we pad the position embeddings at the end.
    bigbird_pos_embs = torch.cat([bigbird_pos_embs, roberta_pos_embs_extra], dim=0)

    embedding_parameters2copy.append(("position_embeddings.weight", bigbird_pos_embs))

    # Load the embedding weights into the bigbird model
    embedding_parameters2copy = OrderedDict(embedding_parameters2copy)
    bigbird_model.base_model.embeddings.load_state_dict(
        embedding_parameters2copy, strict=False
    )

    return bigbird_model, bigbird_tokenizer
