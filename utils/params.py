import os
import tensorflow as tf

import utils.vocab as vocab


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    p_name = os.path.join(model_dir, "params.json")
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(p_name) or not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(p_name) as fd:
        tf.logging.info("Restoring hyper parameters from %s" % p_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def import_infer_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def export_params(output_dir, name, params):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)
    with tf.gfile.Open(filename, "w") as fd:
        fd.write(params.to_json())


def collect_params(all_params, params):
    collected = tf.contrib.training.HParams()

    for k in params.values().keys():
        collected.add_hparam(k, getattr(all_params, k))

    return collected


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in params1.values().items():
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in params2.values().items():
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def override_parameters(params, args):
    params.dev_params = args.dev_params
    params.model = args.model
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.vocab = args.vocab or params.vocab
    params.validation = args.validation or params.validation
    params.references = args.references or params.references
    params.parse(args.parameters)

    vocab_dir = os.path.dirname(params.vocab[0])
    params.vocabulary = {
        "source": vocab.load_vocab(params.vocab[0], params),
        "target": vocab.load_vocab(params.vocab[1], params),
        "lf": vocab.load_vocab(os.path.join(vocab_dir, 'vocab-lf.txt'), params),
        "sketch": vocab.load_vocab(os.path.join(vocab_dir, 'vocab-sketch.txt'), params)
    }

    if params.use_pretrained_embedding:
        params.init_word_matrix = vocab.load_word_matrix(args.vocab[0])

    return params


def override_infer_parameters(params, args):
    params.models = args.models
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.vocab = args.vocab
    params.parse(args.parameters)

    vocab_dir = os.path.dirname(params.vocab[0])
    params.vocabulary = {
        "source": vocab.load_vocab(args.vocab[0], params),
        "target": vocab.load_vocab(args.vocab[1], params),
        "lf": vocab.load_vocab(os.path.join(vocab_dir, 'vocab-lf.txt'), params),
        "sketch": vocab.load_vocab(os.path.join(vocab_dir, 'vocab-sketch.txt'), params)
    }

    if params.use_pretrained_embedding:
        params.init_word_matrix = vocab.load_word_matrix(args.vocab[0])

    return params


def parse_infer_params(args, default_params, model_cls_list):
    params_list = [default_params for _ in range(len(model_cls_list))]
    params_list = [
        merge_parameters(params, model_cls.get_parameters())
        for params, model_cls in zip(params_list, model_cls_list)
    ]
    params_list = [
        import_infer_params(args.models[i], args.model, params_list[i])
        for i in range(len(args.models))
    ]
    params_list = [
        override_infer_parameters(params_list[i], args)
        for i in range(len(model_cls_list))
    ]
    return params_list


def parse_params(args, default_params, model_params):
    params = default_params

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = merge_parameters(params, model_params)
    params = import_params(args.output, args.model, params)
    override_parameters(params, args)

    # Export all parameters and model specific parameters
    export_params(params.output, "params.json", params)
    export_params(os.path.join(params.output, 'best/'), "params.json", params)
    export_params(
        params.output,
        "%s.json" % args.model,
        collect_params(params, model_params)
    )
    export_params(
        os.path.join(params.output, 'best/'),
        "%s.json" % args.model,
        collect_params(params, model_params)
    )
    return params


def prepare_dir(params):
    if not os.path.exists(params.output):
        os.makedirs(params.output)
    save_dir = os.path.join(params.output, 'best/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir
