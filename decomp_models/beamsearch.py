from collections import namedtuple

from tensorflow.python.util import nest

from decomp_models.model_utils.common import *


class BeamSearchState(namedtuple("BeamSearchState",
                                 ("inputs", "state", "finish"))):
    pass


def _beam_search_step(time, func, state, batch_size, beam_size, alpha, eos_id):
    # b = batch_size, beam = beam_size, v = vocab_size
    # q' is the previous decode sequence length, q' = 1 at first time step
    seqs, log_probs = state.inputs[:2]  # [batch, beam, q'], [batch, beam]
    flat_seqs = merge_first_two_dims(seqs)  # [batch * beam, q']
    flat_state = nest.map_structure(lambda x: merge_first_two_dims(x),
                                    state.state)
    step_log_probs, next_state = func(flat_seqs, flat_state)  # [batch * beam, v], state_shape
    step_log_probs = split_first_two_dims(step_log_probs, batch_size,
                                          beam_size)  # [batch, beam, v], current step vocab probs of all beams
    next_state = nest.map_structure(
        lambda x: split_first_two_dims(x, batch_size, beam_size),
        next_state)
    curr_log_probs = tf.expand_dims(log_probs, 2) + step_log_probs  # add current vocab beam with previous one word

    # Apply length penalty
    length_penalty = tf.pow((5.0 + tf.to_float(time + 1)) / 6.0, alpha)
    curr_scores = curr_log_probs / length_penalty
    vocab_size = curr_scores.shape[-1].value or tf.shape(curr_scores)[-1]

    # Select top-k candidates
    curr_scores = tf.reshape(curr_scores, [-1, beam_size * vocab_size])  # [b, beam * v]
    # get indices like: [3, vocab + 7, vocab * 3 + 180, ...], vocab_idx + beam_offset
    top_scores, top_indices = tf.nn.top_k(curr_scores, k=2 * beam_size)  # [b, 2 * beam]
    beam_indices = top_indices // vocab_size  # [b, 2 * beam]
    symbol_indices = top_indices % vocab_size  # [b, 2 * beam]
    # Expand sequences
    # Get previous decoder sequence given beam indices
    candidate_seqs = gather_2d(seqs, beam_indices)  # [b, 2 * beam, q']
    # concat current decode word idx to given sequences
    candidate_seqs = tf.concat([candidate_seqs,
                                tf.expand_dims(symbol_indices, 2)], 2)  # [b, 2 * beam, q' + 1] !

    # Expand sequences
    # Suppress finished sequences, if current decode word is eos
    flags = tf.equal(symbol_indices, eos_id)  # [b, beam]
    # with our 2 * beam results, we set those eos score to -inf
    alive_scores = top_scores + tf.to_float(flags) * tf.float32.min  # [b, 2 * beam]
    # and keep top beam ones
    alive_scores, alive_indices = tf.nn.top_k(alive_scores, beam_size)  # [b, beam]
    # get their correspond vocab ids
    alive_symbols = gather_2d(symbol_indices, alive_indices)  # [b, beam]
    # and their correspond beam indices
    alive_indices = gather_2d(beam_indices, alive_indices)  # [b, beam]
    # get their correspond previous sequences
    alive_seqs = gather_2d(seqs, alive_indices)  # [b, beam, q']
    # concat, ta_da -_-
    alive_seqs = tf.concat([alive_seqs, tf.expand_dims(alive_symbols, 2)], 2)  # [b, beam, q' + 1]
    # we set decoder state to those alive beams
    alive_state = nest.map_structure(
        lambda x: gather_2d(x, alive_indices),
        next_state)
    alive_log_probs = alive_scores * length_penalty

    # Select finished sequences
    prev_fin_flags, prev_fin_seqs, prev_fin_scores = state.finish
    step_fin_scores = top_scores + (1.0 - tf.to_float(flags)) * tf.float32.min  # [b, 2 * beam]
    fin_flags = tf.concat([prev_fin_flags, flags], axis=1)  # [batch, 3 * beam]
    fin_scores = tf.concat([prev_fin_scores, step_fin_scores], axis=1)
    fin_scores, fin_indices = tf.nn.top_k(fin_scores, beam_size)  # [b, beam]
    fin_flags = gather_2d(fin_flags, fin_indices)
    # with previous decode complete sequence, we continue pad 0 in the tail
    # so it will be [..., 0, 0, ..., 0]
    pad_seqs = tf.fill([batch_size, beam_size, 1],
                       tf.constant(eos_id, tf.int32))  # [b, beam, 1]
    prev_fin_seqs = tf.concat([prev_fin_seqs, pad_seqs], axis=2)  # [b, beam, q' + 1]
    # we always keep beam fin_seqs along with their scores and use current candidate to update
    fin_seqs = tf.concat([prev_fin_seqs, candidate_seqs], axis=1)  # [b, 3 * beam, q' + 1]
    fin_seqs = gather_2d(fin_seqs, fin_indices)  # [b, beam?, q' + 1]

    new_state = BeamSearchState(
        inputs=(alive_seqs, alive_log_probs, alive_scores),
        state=alive_state,
        finish=(fin_flags, fin_seqs, fin_scores),
    )

    return time + 1, new_state


def beam_search(func, state, batch_size, beam_size, max_length, alpha,
                bos_id, eos_id):
    init_seqs = tf.fill([batch_size, beam_size, 1], bos_id)  # [b, beam, 1], all bos id
    # the init log prob of first beam is greater than other (beam_size - 1) beams
    # to ensure at first time step(all beam with same input), we have different output with top prob
    init_log_probs = tf.constant([[0.] + [tf.float32.min] * (beam_size - 1)])  # [1, beam]
    init_log_probs = tf.tile(init_log_probs, [batch_size, 1])  # [b, beam]
    init_scores = tf.zeros_like(init_log_probs)  # [b, beam]
    fin_seqs = tf.zeros([batch_size, beam_size, 1], tf.int32)  # [b, beam, 1]
    fin_scores = tf.fill([batch_size, beam_size], tf.float32.min)  # [b, beam]
    fin_flags = tf.zeros([batch_size, beam_size], tf.bool)  # [b, beam]

    state = BeamSearchState(
        inputs=(init_seqs, init_log_probs, init_scores),
        state=state,
        finish=(fin_flags, fin_seqs, fin_scores),
    )

    max_step = tf.reduce_max(max_length)

    def _is_finished(t, s):
        log_probs = s.inputs[1]
        finished_flags = s.finish[0]
        finished_scores = s.finish[2]
        max_lp = tf.pow(((5.0 + tf.to_float(max_step)) / 6.0), alpha)
        best_alive_score = log_probs[:, 0] / max_lp
        worst_finished_score = tf.reduce_min(
            finished_scores * tf.to_float(finished_flags), axis=1)
        add_mask = 1.0 - tf.to_float(tf.reduce_any(finished_flags, 1))
        worst_finished_score += tf.float32.min * add_mask
        # each alive seqs score < lowest finished seqs
        bound_is_met = tf.reduce_all(tf.greater(worst_finished_score,
                                                best_alive_score))
        cond = tf.logical_and(tf.less(t, max_step),
                              tf.logical_not(bound_is_met))

        return cond

    def _loop_fn(t, s):
        outs = _beam_search_step(t, func, s, batch_size, beam_size, alpha, eos_id)
        return outs

    time = tf.constant(0, name="time")
    shape_invariants = BeamSearchState(
        inputs=(tf.TensorShape([None, None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None])),
        state=nest.map_structure(infer_shape_invariants, state.state),
        finish=(tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None]),
                tf.TensorShape([None, None]))
    )
    outputs = tf.while_loop(_is_finished, _loop_fn, [time, state],
                            shape_invariants=[tf.TensorShape([]),
                                              shape_invariants],
                            parallel_iterations=1,
                            back_prop=False)

    final_state = outputs[1]
    alive_seqs = final_state.inputs[0]
    alive_scores = final_state.inputs[2]
    final_flags = final_state.finish[0]
    final_seqs = final_state.finish[1]
    final_scores = final_state.finish[2]

    alive_seqs.set_shape([None, beam_size, None])
    final_seqs.set_shape([None, beam_size, None])

    # if any seq have finalized, select finalized seqs & scores
    final_seqs = tf.where(tf.reduce_any(final_flags, 1), final_seqs,
                          alive_seqs)
    final_scores = tf.where(tf.reduce_any(final_flags, 1), final_scores,
                            alive_scores)

    return final_seqs, final_scores


def create_inference_ops_general(encoder_output, model_infer_fn, batch_size, max_length, params, scope, feature_prefix):
    beam_size = params.beam_size
    top_beams = params.top_beams
    alpha = params.decode_alpha
    bos_id = params.bosId
    eos_id = params.eosId
    if params.use_pos:
        params.hidden_size = params.embed_dim

    decoding_fn = get_inference_fn_general(model_infer_fn, feature_prefix)
    init_state = {
        "encoder": encoder_output,
        "decoder": {
            "layer_%d" % i: {
                "key": tf.zeros([batch_size, 0, params.hidden_size]),
                "value": tf.zeros([batch_size, 0, params.hidden_size])
            }
            for i in range(params.num_decoder_layers)
        }
    }
    # enc: [b, beam, q_1, e],  dec_k & dec_v: [b, beam, 0, e]
    state = nest.map_structure(
        lambda x: tile_to_beam_size(x, beam_size),
        init_state)

    max_length = tf.convert_to_tensor(max_length)  # [b, beam]

    with tf.variable_scope(scope, reuse=True):
        seqs, scores = beam_search(decoding_fn, state, batch_size, beam_size,
                                   max_length, alpha, bos_id, eos_id)  # [b, beam, decode_len], [b, beam]

    return seqs[:, :top_beams, 1:], scores[:, :top_beams]


def get_inference_fn_general(model_infer_fn, feature_prefix="target"):
    def inference_fn(inputs, state):
        local_features = {
            # [bos_id, ...] => [..., 0], remove bos_id in the head and add a zero in the tail
            # with shift left on decoder, we finally get [...] as input to decoder
            # at first time step, the input is [0]
            # at first and second step decode, the input length is 1
            feature_prefix: tf.pad(inputs[:, 1:], [[0, 0], [0, 1]]),
            feature_prefix + "_length": tf.fill([tf.shape(inputs)[0]],
                                                tf.shape(inputs)[1])
        }
        log_prob, new_state = model_infer_fn(local_features, state)
        return log_prob, new_state

    return inference_fn
