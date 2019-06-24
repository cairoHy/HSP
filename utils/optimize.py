import tensorflow as tf


def create_train_op(loss, optimizer, global_step, params):
    with tf.name_scope("create_train_op"):
        grads_and_vars = optimizer.compute_gradients(
            loss, colocate_gradients_with_ops=True)
        gradients = [item[0] for item in grads_and_vars]
        variables = [item[1] for item in grads_and_vars]

        # Add summaries
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_norm/gradient_norm",
                          tf.global_norm(gradients))

        # Gradient clipping
        if isinstance(params.clip_grad_norm or None, float) and params.clip_grad_norm > 0:
            gradients, _ = tf.clip_by_global_norm(gradients,
                                                  params.clip_grad_norm)

        # Update variables
        grads_and_vars = list(zip(gradients, variables))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

        return loss, train_op


def smooth_cross_entropy(logits, labels, smoothing):
    # label smoothing
    vocab_size = tf.shape(logits)[1]

    n = tf.to_float(vocab_size - 1)
    p = 1.0 - smoothing
    q = smoothing / n

    soft_targets = tf.one_hot(tf.cast(labels, tf.int32), depth=vocab_size,
                              on_value=p, off_value=q)  # [b, q_2, v + v']

    soft_targets = tf.reshape(soft_targets, [-1, tf.shape(soft_targets)[2]])  # [b * q_2, v + v']
    logits = tf.clip_by_value(logits, 1e-10, 1.0)  # [b * q_2, v + v']

    xentropy = -tf.reduce_sum(soft_targets * tf.log(logits), 1)

    normalizing = -(p * tf.log(p) + n * q * tf.log(q + 1e-20))
    return xentropy
