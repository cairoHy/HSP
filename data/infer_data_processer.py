"""
In order to do beam search, we need to multiply needed features by beam_size times in batch axis,
the first axis changed from [batch] to [batch * beam_size],
this is what functions in this file work for.
"""
import numpy as np

from data.batcher import Sample, Batch
from decomp_models.model_utils.copy_mechanism import copy_mechanism_preprocess
from utils.query_preprocess import preprocess_query


def prepare_inf_features(features, params):
    """Expand source data: [batch, ...] => [batch * beam_size, ...] """
    decode_length = params.decode_length
    beam_size = params.beam_size
    batch_size = features.source_ids.shape[0]

    # Expand the inputs
    # [batch, length] => [batch, beam_size, length]
    features.source_ids = np.expand_dims(features.source_ids, 1)
    features.source_ids = np.tile(features.source_ids, [1, beam_size, 1])
    shape = features.source_ids.shape

    # [batch, beam_size, length] => [batch * beam_size, length]
    features.source_ids = np.reshape(features.source_ids, [shape[0] * shape[1], shape[2]])
    # ------------------------------------------------------------------------------------------
    # Expand the inputs annotation
    # [batch, length] => [batch, beam_size, length]
    features.pos_anno = np.expand_dims(features.pos_anno, 1)
    features.pos_anno = np.tile(features.pos_anno, [1, beam_size, 1])
    shape = features.pos_anno.shape

    # [batch, beam_size, length] => [batch * beam_size, length]
    features.pos_anno = np.reshape(features.pos_anno, [shape[0] * shape[1], shape[2]])
    # ------------------------------------------------------------------------------------------
    # Expand the inputs oo
    # [batch, length] => [batch, beam_size, length]
    features.source_ids_oo = np.expand_dims(features.source_ids_oo, 1)
    features.source_ids_oo = np.tile(features.source_ids_oo, [1, beam_size, 1])
    shape = features.source_ids_oo.shape

    # [batch, beam_size, length] => [batch * beam_size, length]
    features.source_ids_oo = np.reshape(features.source_ids_oo, [shape[0] * shape[1], shape[2]])
    # ------------------------------------------------------------------------------------------
    # For source sequence length
    features.source_len = np.expand_dims(features.source_len, 1)
    features.source_len = np.tile(features.source_len, [1, beam_size])
    shape = features.source_len.shape

    max_length = features.source_len + decode_length  # [batch, beam_size]

    # [batch, beam_size] => [batch * beam_size]
    features.source_len = np.reshape(features.source_len, [shape[0] * shape[1]])
    # ------------------------------------------------------------------------------------------
    return features, max_length, batch_size


def prepare_inf_features_stage2(features, params):
    beam_size = params.beam_size
    # ------------------------------------------------------------------------------------------
    # Expand the sketch
    # [batch, length] => [batch, beam_size, length]
    features.sketch_ids = np.expand_dims(features.sketch_ids, 1)
    features.sketch_ids = np.tile(features.sketch_ids, [1, beam_size, 1])
    shape = features.sketch_ids.shape

    # [batch, beam_size, length] => [batch * beam_size, length]
    features.sketch_ids = np.reshape(features.sketch_ids, [shape[0] * shape[1], shape[2]])
    # ------------------------------------------------------------------------------------------
    # For sketch sequence length
    features.sketch_len = np.expand_dims(features.sketch_len, 1)
    features.sketch_len = np.tile(features.sketch_len, [1, beam_size])
    shape = features.sketch_len.shape

    # [batch, beam_size] => [batch * beam_size]
    features.sketch_len = np.reshape(features.sketch_len, [shape[0] * shape[1]])
    # ------------------------------------------------------------------------------------------
    return features


def prepare_inf_features_stage3(features, params):
    beam_size = params.beam_size
    # ------------------------------------------------------------------------------------------
    # Expand the sketch
    # [batch, length] => [batch, beam_size, length]
    features.second_sketch_ids = np.expand_dims(features.second_sketch_ids, 1)
    features.second_sketch_ids = np.tile(features.second_sketch_ids, [1, beam_size, 1])
    shape = features.second_sketch_ids.shape

    # [batch, beam_size, length] => [batch * beam_size, length]
    features.second_sketch_ids = np.reshape(features.second_sketch_ids, [shape[0] * shape[1], shape[2]])
    # ------------------------------------------------------------------------------------------
    # For sketch sequence length
    features.second_sketch_len = np.expand_dims(features.second_sketch_len, 1)
    features.second_sketch_len = np.tile(features.second_sketch_len, [1, beam_size])
    shape = features.second_sketch_len.shape

    # [batch, beam_size] => [batch * beam_size]
    features.second_sketch_len = np.reshape(features.second_sketch_len, [shape[0] * shape[1]])
    # ------------------------------------------------------------------------------------------
    return features


def concat_enc_output(*args):
    """Concat encoder output for next stage inference"""
    concated_output = np.concatenate(args, axis=1)
    return concated_output


def get_interface_input(query, params, vocab, annotator):
    query = preprocess_query(query)
    query_tokens = query.strip().split()
    query_ids = [vocab.get(i, params.unkId) for i in query_tokens]
    query_ids_oo, query_oovs, _ = copy_mechanism_preprocess(query_tokens, '', params, vocab)
    pos_anno = [0 for _ in query_ids] if not params.use_pos else annotator.annotate_question(query)['pos']
    query_ids += [params.eosId]
    pos_anno += [0]
    query_ids_oo += [params.eosId]
    assert len(query_ids) == len(query_ids_oo)
    assert len(query_ids) == len(pos_anno)
    sample = {
        "source": query_tokens,
        "source_ids": query_ids,
        "source_length": len(query_ids),
        "source_ids_oo": query_ids_oo,
        "oov_str": query_oovs,
        "oov_num": len(query_oovs),
        "pos_anno": pos_anno
    }
    return Batch([Sample(sample, params, 'infer')], params, 'infer')
