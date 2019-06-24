import tensorflow as tf

from decomp_models.model_utils.common import infer_shape


def linear(input_data,
           output_size,
           bias=True,
           dtype=None,
           scope=None):
    """
    output = input_data * W + b
    """
    with tf.variable_scope(scope, default_name="linear"):
        input_shape = infer_shape(input_data)
        input_size = input_shape[-1]
        output_shape = tf.concat([input_shape[:-1], [output_size]], axis=0)

        W = tf.get_variable("W", shape=[input_size, output_size], dtype=dtype)
        output = tf.matmul(tf.reshape(input_data, [-1, input_size]), W)

        if bias:
            bias = tf.get_variable("b", shape=[output_size], dtype=dtype)
            output = output + bias

        return tf.reshape(output, output_shape)


def layer_norm(input_data,
               epsilon=1e-6,
               dtype=None,
               scope=None):
    with tf.variable_scope(scope, default_name="layer_norm"):
        input_size = infer_shape(input_data)[-1]

        scale = tf.get_variable("scale", shape=[input_size],
                                initializer=tf.ones_initializer())
        bias = tf.get_variable("bias", shape=[input_size],
                               initializer=tf.zeros_initializer)

        mean = tf.reduce_mean(input_data, -1, True)
        variance = tf.reduce_mean(tf.square(input_data - mean), -1, True)

        input_norm = (input_data - mean) * tf.rsqrt(variance + epsilon)
        output = input_norm * scale + bias

        return output


def smoothed_softmax_cross_entropy(logits,
                                   labels,
                                   smoothing,
                                   normalize):
    if logits is None or labels is None:
        raise ValueError("Both logits and labels must be provided")

    with tf.name_scope("smoothed_softmax_cross_entropy",
                       values=[logits, labels]):

        labels = tf.reshape(labels, [-1])

        if smoothing is None or smoothing == 0.0:
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )
            return ce

        # label smoothing
        vocab_size = tf.shape(logits)[1]

        n = tf.to_float(vocab_size - 1)
        p = 1.0 - smoothing
        q = smoothing / n

        soft_targets = tf.one_hot(tf.cast(labels, tf.int32), depth=vocab_size,
                                  on_value=p, off_value=q)
        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                           labels=soft_targets)

        if normalize is False:
            return xentropy

        # Normalizing constant is the best cross-entropy value with soft targets. 
        # We subtract it just for readability, makes no difference on learning
        normalizing = -(p * tf.log(p) + n * q * tf.log(q + 1e-20))

        return xentropy - normalizing


def residual_fn(previous_data,
                input_data,
                dropout_rate=None):
    if dropout_rate is not None and dropout_rate > 0.0:
        input_data = tf.nn.dropout(input_data, 1 - dropout_rate)

    return previous_data + input_data


def attention_bias(inputs, mode, inf=-1e9, name=None):
    """ A bias tensor used in attention mechanism
    :param inputs: A tensor
    :param mode: one of "causal", "masking", "proximal" or "distance"
    :param inf: A floating value
    :param name: optional string
    :returns: A 4D tensor with shape [batch, heads, queries, memories]
    """

    with tf.name_scope(name, default_name="attention_bias", values=[inputs]):
        if mode == "causal":
            length = inputs
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            ret = inf * (1.0 - lower_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        elif mode == "masking":
            mask = inputs
            ret = (1.0 - mask) * inf
            return tf.expand_dims(tf.expand_dims(ret, 1), 1)
        elif mode == "proximal":
            length = inputs
            r = tf.to_float(tf.range(length))
            diff = tf.expand_dims(r, 0) - tf.expand_dims(r, 1)
            m = tf.expand_dims(tf.expand_dims(-tf.log(1 + tf.abs(diff)), 0), 0)
            return m
        elif mode == "distance":
            length, distance = inputs
            distance = tf.where(distance > length, 0, distance)
            distance = tf.cast(distance, tf.int64)
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            mask_triangle = 1.0 - tf.matrix_band_part(
                tf.ones([length, length]), distance - 1, 0
            )
            ret = inf * (1.0 - lower_triangle + mask_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        else:
            raise ValueError("Unknown mode %s" % mode)


def layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layer_norm(x)
    else:
        raise ValueError("Unknown mode %s" % mode)


def ffn_layer(inputs, hidden_size, output_size, dropout_rate=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs]):
        with tf.variable_scope("input_layer"):
            hidden = linear(inputs, hidden_size)
            hidden = tf.nn.relu(hidden)

        if dropout_rate is not None and dropout_rate > 0.0:
            hidden = tf.nn.dropout(hidden, 1 - dropout_rate)

        with tf.variable_scope("output_layer"):
            output = linear(hidden, output_size)

        return output
