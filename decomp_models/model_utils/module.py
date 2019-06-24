from tensorflow.python.util import nest

from decomp_models.model_utils.attention import multihead_attention, compute_copy_weights
from decomp_models.model_utils.common import set_first_dim_shape_return
from decomp_models.model_utils.layer import *


def transformer_encoder(inputs, bias, params, scope=None, reuse=False):
    with tf.variable_scope(scope, default_name="encoder",
                           values=[inputs, bias], reuse=reuse):
        x = inputs
        for layer in range(params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        params.attention_dropout
                    )
                    y = y["outputs"]
                    x = residual_fn(x, y, params.residual_dropout)

                with tf.variable_scope("feed_forward"):
                    y = ffn_layer(
                        layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        params.relu_dropout
                    )
                    x = residual_fn(x, y, params.residual_dropout)

        outputs = layer_process(x, params.layer_preprocess)

        return outputs


def transformer_decoder(inputs, memory, bias, mem_bias, params, state=None, scope=None, reuse=False):
    with tf.variable_scope(scope, default_name="decoder",
                           values=[inputs, memory, bias, mem_bias], reuse=reuse):
        x = inputs
        next_state = {}
        all_att_weights = []
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None

                with tf.variable_scope("self_attention"):
                    y = multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        params.attention_dropout,
                        state=layer_state
                    )

                    if layer_state is not None:
                        state_batch_shape = layer_state["key"].shape[0]
                        y["state"] = nest.map_structure(lambda x: set_first_dim_shape_return(x, state_batch_shape),
                                                        y["state"])
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = residual_fn(x, y, params.residual_dropout)

                with tf.variable_scope("encdec_attention"):
                    att_weights, y = multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        params.attention_dropout,
                        encdec=True
                    )
                    all_att_weights.append(att_weights)
                    y = y["outputs"]
                    x = residual_fn(x, y, params.residual_dropout)

                with tf.variable_scope("feed_forward"):
                    y = ffn_layer(
                        layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        params.relu_dropout,
                    )
                    x = residual_fn(x, y, params.residual_dropout)

        outputs = layer_process(x, params.layer_preprocess)

        if state is not None:
            return all_att_weights, outputs, next_state

        return all_att_weights, outputs


def transformer_concated_decoder(inputs, memory, bias, mem_bias, params, state=None, scope=None,
                                 reuse=False, src_true_mask=None):
    """
    Concat version of decoder, the only difference is the att_weights.
    This version calculate att_weights, mask them using `src_true_mask`
    """
    with tf.variable_scope(scope, default_name="decoder",
                           values=[inputs, memory, bias, mem_bias], reuse=reuse):
        x = inputs
        next_state = {}
        all_x = []
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None

                with tf.variable_scope("self_attention"):
                    y = multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        params.attention_dropout,
                        state=layer_state
                    )

                    if layer_state is not None:
                        state_batch_shape = layer_state["key"].shape[0]
                        y["state"] = nest.map_structure(lambda x: set_first_dim_shape_return(x, state_batch_shape),
                                                        y["state"])
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = residual_fn(x, y, params.residual_dropout)

                with tf.variable_scope("encdec_attention"):
                    _, y = multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        params.attention_dropout,
                        encdec=True
                    )
                    y = y["outputs"]
                    x = residual_fn(x, y, params.residual_dropout)
                    all_x.append(x)

                with tf.variable_scope("feed_forward"):
                    y = ffn_layer(
                        layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        params.relu_dropout,
                    )
                    x = residual_fn(x, y, params.residual_dropout)

        outputs = layer_process(x, params.layer_preprocess)

        copy_query = layer_process(all_x[-1], params.layer_preprocess)

        att_weights = compute_copy_weights(copy_query, memory, mem_bias, params.hidden_size, params.attention_dropout)

        if state is not None:
            return att_weights, outputs, next_state

        return att_weights, outputs
