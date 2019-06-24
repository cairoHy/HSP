import argparse
import time
from collections import namedtuple
from shutil import copyfile

import os

import decomp_models
import inference
import utils.optimize as optimize
from data.batcher import Batcher
from evaluate import evaluate
from utils.params import parse_params, prepare_dir
from utils.train_utils import *


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training neural main decomp_models",
        usage="train.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, nargs=2, required=True,
                        help="Path of source and target corpus")
    parser.add_argument("--output", type=str, default="train",
                        help="Path to saved decomp_models")
    parser.add_argument("--vocab", type=str, nargs=2, required=True,
                        help="Path of source and target vocabulary")
    parser.add_argument("--validation", type=str, required=True,
                        help="Path of validation file")
    parser.add_argument("--references", type=str, required=True,
                        help="Path of reference file file")

    parser.add_argument("--model", type=str, default="Transformer",
                        help="Model name")
    parser.add_argument("--metric", type=str, default="f1",
                        help="Model metrics, [f1] or [em]")

    # configuration
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")
    parser.add_argument("--dev_params", type=str, default="",
                        help="Dev hyper parameters during training")

    return parser.parse_args(args)


def default_parameters():
    params = tf.contrib.training.HParams(
        input=["", ""],
        output="",
        model="transformer",
        vocab=["", ""],
        # Default training hyper parameters
        num_threads=8,
        batch_size=4096,
        max_length=256,
        warmup_steps=8000,
        train_steps=100000,
        buffer_size=10000,
        constant_batch_size=False,
        device_list=[0],
        initializer="uniform_unit_scaling",
        initializer_gain=1.0,
        optimizer="Adam",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-9,
        clip_grad_norm=0.0,
        learning_rate=0.2,
        learning_rate_minimum=None,
        learning_rate_decay="linear_warmup_rsqrt_decay",
        learning_rate_boundaries=[0],
        learning_rate_values=[0.0],
        keep_checkpoint_max=10,
        keep_top_checkpoint_max=1,
        print_steps=100,
        # Validation
        eval_steps_begin=6000,  # exist bugs
        eval_steps=400,
        eval_secs=0,
        eval_batch_size=32,
        top_beams=1,  # The number of printed beams
        beam_size=4,
        decode_alpha=0.6,  # word penalty
        decode_length=50,  # max length = source length + decode length, during inference
        validation="",
        references="",
        save_checkpoint_secs=0,
        save_checkpoint_steps=0,
        only_save_trainable=False,  # Set to true if the model is only used to inference
        seed=12345
    )

    return params


# noinspection PyProtectedMember
def main(args):
    # log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.INFO)

    # model & params & dirs
    model_cls = decomp_models.get_decomp_model(args.model)
    params = parse_params(args, default_parameters(), model_cls.get_parameters())
    best_model_dir = prepare_dir(params)

    # tf.set_random_seed(params.seed)

    # define evaluation args & op
    ARGS = namedtuple('Args', ['input', 'output', 'vocab', 'models', 'model', 'parameters', 'log'])
    trans_file = os.path.join(params.output, os.path.basename(params.validation) + '.trans')
    eval_args = ARGS(
        input=params.validation,
        output=trans_file,
        vocab=[params.vocab[0], params.vocab[1]],
        models=[params.output, ],
        model=args.model,
        parameters=params.dev_params,
        log=False
    )

    def eval_op():
        tf.logging.info("Evaluate model on dev set...")
        inference.main(eval_args, verbose=False)
        return eval_args.output, evaluate(pred_file=eval_args.output, ref_file=args.references)

    # Build Graph
    with tf.Graph().as_default():
        # Build input queue
        batcher = Batcher(params, 'train')

        # Build model
        initializer = get_initializer(params)
        model = model_cls(params, args.model, initializer=initializer)

        # Create global step
        global_step = tf.train.get_or_create_global_step()

        # Print trainable parameter and their shape
        all_weights = {v.name: v for v in tf.trainable_variables()}
        for v_name in sorted(list(all_weights)):
            v = all_weights[v_name]
            tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                            str(v.shape).ljust(20))
        # Print parameter number
        total_size = sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
        tf.logging.info("Total trainable variables size: %d", total_size)

        # lr decay
        learning_rate = get_learning_rate_decay(params.learning_rate,
                                                global_step, params)
        if params.learning_rate_minimum:
            lr_min = float(params.learning_rate_minimum)
            learning_rate = tf.maximum(learning_rate, tf.to_float(lr_min))

        learning_rate = tf.convert_to_tensor(learning_rate, dtype=tf.float32)
        tf.summary.scalar("learning_rate", learning_rate)

        # Create optimizer
        if params.optimizer == "Adam":
            opt = tf.train.AdamOptimizer(learning_rate,
                                         beta1=params.adam_beta1,
                                         beta2=params.adam_beta2,
                                         epsilon=params.adam_epsilon)
        elif params.optimizer == "LazyAdam":
            opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate,
                                                   beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2,
                                                   epsilon=params.adam_epsilon)
        else:
            raise RuntimeError("Optimizer %s not supported" % params.optimizer)

        loss, ops = optimize.create_train_op(model.loss, opt, global_step, params)

        init_op = init_variables()

        restore_op = restore_variables(args.output)

        config = session_config(params)

        best_metric = 0
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=params.keep_checkpoint_max)
        best_saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=params.keep_checkpoint_max)
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            sess.run(restore_op)
            t0 = time.time()
            while True:
                batch = batcher.next_batch()
                _, step = sess.run([ops, global_step], model.make_feed_dict(batch))

                if step > params.train_steps:  # terminate
                    tf.logging.info('Train done.')
                    break

                if params.save_checkpoint_steps and step % params.save_checkpoint_steps == 0:  # save checkpoint
                    model.save(saver, sess, params.output, model_prefix='{}-{}'.format(args.model, step))

                if step % params.print_steps == 0:  # print message
                    t1 = time.time()
                    loss = sess.run(model.loss, model.make_feed_dict(batch))
                    tf.logging.info('seconds for training step: %.3f', t1 - t0)
                    t0 = time.time()
                    tf.logging.info('Step {}, Loss {}'.format(step, loss))

                if step % params.eval_steps == 0 and step > params.eval_steps_begin:  # evaluation
                    trans_output, (total, comp_acc, metrics, em) = eval_op()
                    bleu, rouge = metrics['Bleu-4'], metrics['Rouge-L']
                    f1 = 2 * bleu * rouge / (bleu + rouge + 1e-6)
                    metric = f1 if args.metric == 'f1' else em
                    if metric > best_metric:
                        best_metric = metric
                        tf.logging.info("Step {} -> best model acc: {:.3f}, bleu-4: {:.3f}, rouge-L: {:.3f}.".format(
                            step, comp_acc, bleu, rouge))
                        tf.logging.info("Step {} -> best model em: {:.3f}.".format(step, em))
                        copyfile(trans_output, trans_output + '.{}'.format(step))
                        model.save(best_saver, sess, best_model_dir, model_prefix='{}-{}'.format(args.model, step))
                        tf.logging.info("Save best model...")
                    else:
                        tf.logging.info("Step {} -> model acc: {:.3f}, bleu-4: {:.3f}, rouge-L: {:.3f}.".format(
                            step, comp_acc, bleu, rouge))
                        tf.logging.info("Step {} -> model em: {:.3f}.".format(step, em))

            tf.logging.info("Train is end, best Metric: {}".format(best_metric))


if __name__ == "__main__":
    main(parse_args())
