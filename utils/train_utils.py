import numpy as np
import tensorflow as tf

from decomp_models.model_utils.common import infer_shape


def get_initializer(params):
    if params.initializer == "uniform":
        max_val = 0.1 * params.initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif params.initializer == "normal":
        return tf.random_normal_initializer(0.0, params.initializer_gain)
    elif params.initializer == "orthogonal":
        return tf.orthogonal_initializer(params.initializer_gain)
    elif params.initializer == "normal_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="normal")
    elif params.initializer == "uniform_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="uniform")
    else:
        raise ValueError("Unrecognized initializer: %s" % params.initializer)


def get_learning_rate_decay(learning_rate, global_step, params):
    if params.learning_rate_decay in ["linear_warmup_rsqrt_decay", "noam"]:
        step = tf.to_float(global_step)
        warmup_steps = tf.to_float(params.warmup_steps)
        multiplier = 10 * (params.hidden_size ** -0.5)
        decay = multiplier * tf.minimum((step + 1) * (warmup_steps ** -1.5),
                                        (step + 1) ** -0.5)

        return learning_rate * decay
    elif params.learning_rate_decay == "piecewise_constant":
        return tf.train.piecewise_constant(tf.to_int32(global_step),
                                           params.learning_rate_boundaries,
                                           params.learning_rate_values)
    elif params.learning_rate_decay == "none":
        return learning_rate
    else:
        raise ValueError("Unknown learning_rate_decay")


def session_config(params, is_train=True):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=is_train)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    config.gpu_options.allow_growth = True
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str
    return config


def init_variables():
    return tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


def restore_variables(checkpoint):
    if tf.train.latest_checkpoint(checkpoint) is None:
        return tf.no_op(name="restore_op")

    # Load checkpoints
    tf.logging.info("Loading %s" % checkpoint)
    var_list = tf.train.list_variables(checkpoint)
    reader = tf.train.load_checkpoint(checkpoint)
    values = {}

    for (name, shape) in var_list:
        tensor = reader.get_tensor(name)
        name = name.split(":")[0]
        values[name] = tensor

    var_list = tf.trainable_variables()
    ops = []

    for var in var_list:
        name = var.name.split(":")[0]

        if name in values:
            ops.append(tf.assign(var, values[name]))
        else:
            ops.append(tf.assign(var, tf.zeros(infer_shape(var))))

    return tf.group(*ops, name="restore_op")


# --------------------------------------------------------------------------------eval function-------------------
def write_result_to_file(results, file):
    with open(file, "w", encoding='utf-8') as outfile:
        for decoded in results:
            decoded = str.join(" ", decoded)
            outfile.write("%s\n" % decoded)


# trunct word idx, change those greater than vocab_size to unk
def trunct(x, vocab_size):
    if x >= vocab_size:
        return 1
    else:
        return x


np_trunct = np.vectorize(trunct)

np_d_trunct = lambda x, vocab_size: np_trunct(x, vocab_size).astype(np.int32)


def tf_trunct(x, vocab_size, name=None):
    with tf.name_scope(name, "d_spiky", [x]) as name:
        y = tf.py_func(np_d_trunct,
                       [x, vocab_size],
                       [tf.int32],
                       name=name,
                       stateful=False)
        return y[0]
