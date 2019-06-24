import argparse

import os
import tensorflow as tf
from enum import IntEnum

import decomp_models
from data.batcher import Batcher
from data.infer_data_processer import prepare_inf_features, concat_enc_output, prepare_inf_features_stage2, \
    prepare_inf_features_stage3
from decomp_models.beamsearch import create_inference_ops_general
from utils.params import parse_infer_params
from utils.train_utils import session_config, write_result_to_file
from utils.vocab import decode_target_ids, decode_target_ids_copy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decompose using existing decomp_models",
        usage="inference.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True,
                        help="Path of input file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    parser.add_argument("--vocab", type=str, nargs=2, required=True,
                        help="Path of source and target vocabulary")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="Path of trained decomp_models")
    parser.add_argument("--model", type=str, default="Transformer",
                        help="Model name")

    # model and configuration
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")
    parser.add_argument("--log", action="store_true",
                        help="Enable log output")

    return parser.parse_args()


def default_parameters():
    params = tf.contrib.training.HParams(
        input=None,
        output=None,
        vocabulary=None,
        # vocabulary specific
        bos="<BOS>",
        eos="<EOS>",
        unk="<UNK>",
        device_list=[0],
        num_threads=8,
        # decoding
        top_beams=1,
        beam_size=8,
        decode_alpha=0.6,
        decode_length=50,
        decode_batch_size=32,
    )

    return params


def main(args, verbose=True):
    # log level
    if verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    msg_level = tf.logging.INFO if verbose else tf.logging.ERROR
    tf.logging.set_verbosity(msg_level)
    # Load configs
    model_cls = decomp_models.get_decomp_model(args.model)
    params_list = parse_infer_params(args, default_parameters(), [model_cls])

    # Build Graph
    with tf.Graph().as_default():
        params = params_list[0]
        # Build input queue
        batcher = Batcher(params, 'infer')

        # Build graph
        model = model_cls(params, args.model, mode='infer')

        # define different run functions
        def run(ops, data):
            return sess.run(ops, model.make_infer_feed_dict(data))

        def run_stage1(ops, data):
            return sess.run(ops, model.make_stage1_infer_feed_dict(data))

        def run_stage2(ops, data):
            return sess.run(ops, model.make_stage2_infer_feed_dict(data))

        # define inference operations
        def infer_op(enc_output, batch_size, max_length):
            return create_inference_ops_general(enc_output, model.decode_infer, batch_size, max_length, params,
                                                model.scope, feature_prefix="target")

        def stage1_infer_op(enc_output, batch_size, max_length):
            return create_inference_ops_general(enc_output, model.decode_infer_stage_1, batch_size, max_length, params,
                                                model.scope, feature_prefix="sketch")

        def stage2_infer_op(enc_output, batch_size, max_length):
            return create_inference_ops_general(enc_output, model.decode_infer_stage_2, batch_size, max_length, params,
                                                model.scope, feature_prefix="second_sketch")

        def single_stage_model_inference(batch_data, params):
            """Inference for single model"""
            # get output of encoder
            encoder_output = run(model.encoder_output, batch_data)
            # prepare decoder data
            batch, max_dec_len, dec_batch_size = prepare_inf_features(batch_data, params)
            # decoder beam search
            decode_seq, decode_score = run(infer_op(encoder_output, dec_batch_size, max_dec_len), batch)
            # [batch, top_beam, len] => [batch, len], [batch, top_beam] => [batch]
            decode_seq, decode_score = decode_seq[:, 0, :].tolist(), decode_score[:, 0]
            return decode_seq, decode_score

        def two_stage_model_inference(batch_data, params):
            """Inference for two stage model"""
            # 1. stage 1
            # get output of encoder, [b, q_1, e]
            encoder_output = run_stage1(model.encoder_output, batch_data)
            # prepare decoder data, tile feature by beam_size times
            batch, max_dec_len, dec_batch_size = prepare_inf_features(batch_data, params)
            # decoder beam search
            sketch_seq, sketch_score = run_stage1(stage1_infer_op(encoder_output, dec_batch_size, max_dec_len), batch)
            # [batch, top_beam, len] => [batch, len], [batch, top_beam] => [batch]
            sketch_seq, sketch_score = sketch_seq[:, 0, :].tolist(), sketch_score[:, 0]
            # 2. stage 2
            # prepare, add stage1 decoder output to feature
            batch.add_sketch_feature(sketch_seq)
            sketch_encoder_output = run(model.sketch_encoder_output, batch)
            # concat src enc output & sketch enc output using SAME WAY in model
            concated_encoder_output = concat_enc_output(encoder_output, sketch_encoder_output)
            batch = prepare_inf_features_stage2(batch, params)
            # decoder beam search
            decode_seq, decode_score = run(infer_op(concated_encoder_output, dec_batch_size, max_dec_len), batch)
            # [batch, top_beam, len] => [batch, len], [batch, top_beam] => [batch]
            decode_seq, decode_score = decode_seq[:, 0, :].tolist(), decode_score[:, 0]
            return decode_seq, decode_score, sketch_seq

        def hierarchical_model_inference(batch_data, params):
            """Inference for hierarchical model"""
            # 1. stage 1
            # get output of encoder, [b, q_1, e]
            encoder_output = run_stage1(model.encoder_output, batch_data)
            # prepare decoder data, tile feature by beam_size times
            batch, max_dec_len, dec_batch_size = prepare_inf_features(batch_data, params)
            # decoder beam search, get sketch-1
            sketch1_seq, sketch1_score = run_stage1(stage1_infer_op(encoder_output, dec_batch_size, max_dec_len), batch)
            # [batch, top_beam, len] => [batch, len], [batch, top_beam] => [batch]
            sketch1_seq, sketch1_score = sketch1_seq[:, 0, :].tolist(), sketch1_score[:, 0]
            # 2. stage 2
            # prepare, add stage1 decoder output to feature
            batch.add_sketch_feature(sketch1_seq)
            sketch1_encoder_output = run_stage2(model.sketch1_encoder_output, batch)
            # concat src enc output & sketch enc output using SAME WAY in model
            concated_sketch1_encoder_output = concat_enc_output(encoder_output, sketch1_encoder_output)
            batch = prepare_inf_features_stage2(batch, params)
            # decoder beam search, get sketch-2
            sketch2_seq, sketch2_score = run_stage2(
                stage2_infer_op(concated_sketch1_encoder_output, dec_batch_size, max_dec_len),
                batch)
            # [batch, top_beam, len] => [batch, len], [batch, top_beam] => [batch]
            sketch2_seq, sketch2_score = sketch2_seq[:, 0, :].tolist(), sketch2_score[:, 0]
            # 3. stage 3
            batch.add_second_sketch_feature(sketch2_seq)
            sketch2_encoder_output = run(model.sketch2_encoder_output, batch)
            # concat src enc output & sketch enc output using SAME WAY in model
            final_encoder_output = concat_enc_output(encoder_output,
                                                     sketch1_encoder_output,
                                                     sketch2_encoder_output)
            batch = prepare_inf_features_stage3(batch, params)
            # decoder beam search, get target seq
            decode_seq, decode_score = run(infer_op(final_encoder_output, dec_batch_size, max_dec_len),
                                           batch)
            # [batch, top_beam, len] => [batch, len], [batch, top_beam] => [batch]
            decode_seq, decode_score = decode_seq[:, 0, :].tolist(), decode_score[:, 0]
            return decode_seq, decode_score, sketch1_seq, sketch2_seq

        class InferType(IntEnum):
            hierarchical = 1
            multi_step = 2
            single = 3

        def determine_infer_type():
            """Determine the inference type, using parameters define in model class"""
            try:
                if params.hierarchical_inference:
                    return InferType.hierarchical
            except AttributeError:
                try:
                    if params.multi_step_inference:
                        return InferType.multi_step
                except AttributeError:
                    return InferType.single

        # Create session
        with tf.Session(config=session_config(params, is_train=False)) as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # Restore
            model.restore(sess, args.models[0])
            tf.logging.set_verbosity(tf.logging.INFO)
            # Prepare
            results, sketch_results = [], []
            sketch1_results, sketch2_results = [], []
            infer_type = determine_infer_type()
            while True:
                # predict one batch
                batch = batcher.next_batch()
                if not batch:
                    break
                # inference ids seq
                if infer_type == InferType.multi_step:
                    decode_seq, decode_score, decode_sketch = two_stage_model_inference(batch, params)
                    decode_sketch_result = decode_target_ids_copy(decode_sketch, batch, params, 'sketch')
                    sketch_results += decode_sketch_result
                elif infer_type == InferType.hierarchical:
                    decode_seq, decode_score, dec_sketch1, dec_sketch2 = hierarchical_model_inference(batch, params)
                    dec_sketch1 = decode_target_ids_copy(dec_sketch1, batch, params, 'sketch')
                    dec_sketch2 = decode_target_ids_copy(dec_sketch2, batch, params, 'sketch')
                    sketch1_results += dec_sketch1
                    sketch2_results += dec_sketch2
                else:
                    decode_seq, decode_score = single_stage_model_inference(batch, params)
                # convert to string
                if params.use_copy:
                    decode_result = decode_target_ids_copy(decode_seq, batch, params)
                else:
                    decode_result = decode_target_ids(decode_seq, params)
                results += decode_result
                tf.logging.log(tf.logging.INFO, "Finished sample %d" % len(results))
        # write results to file
        write_result_to_file(results, params.output)
        if infer_type == InferType.multi_step:
            write_result_to_file(sketch_results, params.output + '.sketch')
        elif infer_type == InferType.hierarchical:
            write_result_to_file(sketch2_results, params.output + '.sketch2')
            write_result_to_file(sketch1_results, params.output + '.sketch1')


if __name__ == "__main__":
    main(parse_args())
