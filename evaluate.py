import argparse
import json
from collections import defaultdict

import os
import pandas as pd
import tensorflow as tf

from utils.bleu_metric.bleu import Bleu
from utils.query_preprocess import preprocess_query
from utils.rouge_metric.rouge import Rouge


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate dev/test set prediction performance.",
    )
    parser.add_argument("--prefix", type=str,
                        default="/data/complexwebquestions/",
                        help="Prefix of target dataset path")
    parser.add_argument("--ori_file", type=str, required=True,
                        help="Path of origin file")
    parser.add_argument("--pred_file", type=str, required=True,
                        help="Path of prediction file")
    parser.add_argument("--ref_file", type=str, required=True,
                        help="Path of reference file")
    parser.add_argument("--calcu_origin", action="store_true",
                        help="Calculate performance of origin paper")
    return parser.parse_args()


def build_pred_ref_dict(ori_file, pred_file, ref_file):
    ref_dict, pred_dict, query_dict, id_dict = {}, {}, {}, {}
    for idx, line in enumerate(open(ref_file, 'r', encoding='utf-8')):
        ref_dict.update({idx: [line.strip(), ]})
    if pred_file:
        for idx, line in enumerate(open(pred_file, 'r', encoding='utf-8')):
            pred_dict.update({idx: [line.strip(), ]})
    for idx, line in enumerate(open(ori_file, 'r', encoding='utf-8')):
        query_dict.update({idx: [line.strip(), ]})
        id_dict.update({idx: idx})  # TODO: change to question_id here
    return ref_dict, pred_dict, query_dict, id_dict


def compute_bleu_rouge(pred_dict, ref_dict, bleu_order=4):
    """
    Compute bleu and rouge scores.
    """
    assert set(pred_dict.keys()) == set(ref_dict.keys()), \
        "missing keys: {}".format(set(ref_dict.keys()) - set(pred_dict.keys()))
    scores = {}
    bleu_scores, _ = Bleu(bleu_order).compute_score(ref_dict, pred_dict)
    for i, bleu_score in enumerate(bleu_scores):
        scores['Bleu-%d' % (i + 1)] = bleu_score
    rouge_score, _ = Rouge().compute_score(ref_dict, pred_dict)
    scores['Rouge-L'] = rouge_score
    return scores


def res_eval_with_type_acc(query_dict, pred_dict, ref_dict, id_dict, prefix=None, save=True):
    """Calculate target bleu/rouge value compared to golden, with/without composition type em."""
    num, acc = 0, 0
    new_ref_dict, new_pred_dict = {}, {}
    for key in ref_dict.keys():
        query, pred, ref, q_id = query_dict[key][0], pred_dict[key][0], ref_dict[key][0], id_dict[key]
        comp_gold, comp_pred = ref.split('#')[0].strip(), pred.split('#')[0].strip()
        num += 1
        if comp_gold != comp_pred:
            continue
        else:
            acc += 1
            new_ref_dict[key] = ref_dict[key]
            new_pred_dict[key] = pred_dict[key]
    print('Comp Type Acc: {}/{} -> {:.2f}%'.format(acc, num, acc / num * 100))
    all_res = []
    for key in new_ref_dict.keys():
        query, pred, ref, q_id = query_dict[key][0], new_pred_dict[key][0], new_ref_dict[key][0], id_dict[key]
        score = compute_bleu_rouge({'1': pred_dict[key]}, {'1': ref_dict[key]})
        sample = {'id': q_id, 'query': query, 'pred': pred, 'ref': ref, 'bleu-4': score['Bleu-4'],
                  'rouge-l': score['Rouge-L']}
        all_res.append(sample)
    scores = compute_bleu_rouge(new_pred_dict, new_ref_dict)
    if save:
        print(scores)
        res_pd = pd.DataFrame(all_res)
        res_pd.to_csv(os.path.join(prefix, 'res-with-type.csv'), encoding='utf-8')
    return num, acc / num * 100, scores


def build_pred_dict_from_ref(ori_pred_file, ref_dict):
    ori_pred_json = json.load(open(ori_pred_file, 'r', encoding='utf-8'))
    ori_pred_dict = {}
    for i in ori_pred_json:
        ori_pred_dict[preprocess_query(i['question']).strip()] = i
    pred_dict = {}
    for key, query in ref_dict.items():
        query = query[0]
        if query.strip() in ori_pred_dict.keys():
            sample = ori_pred_dict[query.strip()]
            pred = sample['comp'] + ' # ' + preprocess_query(sample['split_part1']) + ' # ' + preprocess_query(
                sample['split_part2'])
            pred_dict[key] = [pred, ]
        else:
            print('Warning!!!!')
    return pred_dict


def ori_res_eval(args):
    print('Calculate original paper score...')
    prefix = args.prefix
    ref_file, ori_file = args.ref_file, args.ori_file
    ref_type = os.path.basename(ref_file).split('-')[0]
    ref_file = os.path.join(prefix, ref_file)
    ori_file = os.path.join(prefix, ori_file)
    ori_pred_file = os.path.join(prefix, 'split_points/{}.json'.format(ref_type))
    ref_dict, pred_dict, query_dict, id_dict = build_pred_ref_dict(ori_file, None, ref_file)
    ori_pred_dict = build_pred_dict_from_ref(ori_pred_file, query_dict)
    print('Compute bleu and rouge...')
    scores = compute_bleu_rouge(ori_pred_dict, ref_dict)
    print(scores)


def calculate_exact_match(pred_dict, ref_dict):
    """Calculate target EM value."""
    num, em = 0, 0
    for key in pred_dict.keys():
        num += 1
        if ' '.join(pred_dict[key]).strip() == ' '.join(ref_dict[key]).strip():
            em += 1
    return em / num * 100


def calculate_sketch_performance(sketch_pred_file, *args):
    sketch_type = os.path.basename(sketch_pred_file).split('.')[-1]
    for ref_file in args:
        total, type_acc, acc = 0, 0, 0
        ref_file_name = os.path.basename(ref_file)
        for pred, ref in zip(open(sketch_pred_file, 'r', encoding='utf-8'),
                             open(ref_file, 'r', encoding='utf-8')):
            total += 1
            if pred.strip() == ref.strip():
                acc += 1
            pred_comp = pred.split('#')[0].strip()
            ref_comp = ref.split('#')[0].strip()
            if pred_comp == ref_comp:
                type_acc += 1
        tf.logging.info('Compared to {}, {} type acc is {:.3f} %, {}/{}.'.format(ref_file_name,
                                                                                 sketch_type,
                                                                                 type_acc / total * 100,
                                                                                 type_acc,
                                                                                 total))
        tf.logging.info('Compared to {}, {} em is {:.3f} %, {}/{}.'.format(ref_file_name,
                                                                           sketch_type,
                                                                           acc / total * 100,
                                                                           acc,
                                                                           total))


def calculate_sketch_type_acc(ref_file, pred_file):
    # 1. get all possible ref files, including sketch & sub-query
    sketch_ref_file = os.path.join(os.path.dirname(ref_file), os.path.basename(pred_file).split('-')[0] + '.sketch')
    subq_ref_file = os.path.join(os.path.dirname(ref_file), os.path.basename(pred_file).split('-')[0] + '-tgt.json')
    # 2. list all possible sketch pred files
    sketch_pred_file = pred_file + '.sketch'
    hierarchical_pred_file1 = pred_file + '.sketch1'
    hierarchical_pred_file2 = pred_file + '.sketch2'
    # 3. compare each exist sketch pred file | all ref files
    if os.path.exists(sketch_pred_file):
        calculate_sketch_performance(sketch_pred_file, sketch_ref_file, subq_ref_file)
    if os.path.exists(hierarchical_pred_file1):
        calculate_sketch_performance(hierarchical_pred_file1, sketch_ref_file, subq_ref_file)
    if os.path.exists(hierarchical_pred_file2):
        calculate_sketch_performance(hierarchical_pred_file2, sketch_ref_file, subq_ref_file)


def calculate_exact_match_for_each_q_type(ref_file, pred_file):
    sketch_ref_file = os.path.join(os.path.dirname(ref_file), os.path.basename(pred_file).split('-')[0] + '.sketch')
    question_file = os.path.join(os.path.dirname(ref_file), os.path.basename(pred_file).split('-')[0] + '-src.json')
    question_dict = {}
    for idx, line in enumerate(open(question_file, 'r', encoding='utf-8')):
        question_dict.update({idx: line.strip()})
    question_length_golden = defaultdict(int)
    question_length_num = defaultdict(int)
    ref_dict, pred_dict, sketch_dict, id_dict = build_pred_ref_dict(sketch_ref_file, pred_file, ref_file)
    res = {'conjunction': 0, 'composition': 0, 'superlative': 0, 'comparative': 0}
    type_num = {'conjunction': 0, 'composition': 0, 'superlative': 0, 'comparative': 0}
    num = 0
    for key in pred_dict.keys():
        num += 1
        q_len_span = len(question_dict[key].split())
        if q_len_span < 7: q_len_span = '4-7'
        elif 7 <= q_len_span < 11: q_len_span = '7-10'
        elif 11 <= q_len_span < 15: q_len_span = '11-14'
        elif 15 <= q_len_span < 19: q_len_span = '15-18'
        elif 19 <= q_len_span < 23: q_len_span = '19-22'
        else: q_len_span = '>22'
        question_length_num[q_len_span] += 1
        q_type = sketch_dict[key][0].split('#')[0].strip()
        type_num[q_type] += 1
        if ' '.join(pred_dict[key]).strip() == ' '.join(ref_dict[key]).strip():
            res[q_type] += 1
            question_length_golden[q_len_span] += 1
    for key in res.keys():
        print('{}: {}/{}, {:.2f}%'.format(key, res[key], type_num[key], res[key] / type_num[key] * 100))
    for key in question_length_num.keys():
        print('{}: {}/{}, {:.2f}%'.format(key, question_length_golden[key], question_length_num[key],
                                          question_length_golden[key] / question_length_num[key] * 100))


def res_eval(args):
    """Evaluate after train."""
    if args.calcu_origin:
        ori_res_eval(args)
        return
    prefix = args.prefix
    pred_file, ref_file, ori_file = args.pred_file, args.ref_file, args.ori_file
    pred_file = os.path.join(prefix, pred_file)
    ref_file = os.path.join(prefix, ref_file)
    ori_file = os.path.join(prefix, ori_file)
    ref_dict, pred_dict, query_dict, id_dict = build_pred_ref_dict(ori_file, pred_file, ref_file)
    print('Compute bleu and rouge...')
    all_res = []
    for key in ref_dict.keys():
        query, pred, ref, q_id = query_dict[key][0], pred_dict[key][0], ref_dict[key][0], id_dict[key]
        score = compute_bleu_rouge({'1': pred_dict[key]}, {'1': ref_dict[key]})
        sample = {'id': q_id, 'query': query, 'pred': pred, 'ref': ref, 'bleu-4': score['Bleu-4'],
                  'rouge-l': score['Rouge-L']}
        all_res.append(sample)
    scores = compute_bleu_rouge(pred_dict, ref_dict)
    print(scores)
    res_pd = pd.DataFrame(all_res)
    res_pd.to_csv(os.path.join(prefix, 'res.csv'), encoding='utf-8')
    res_eval_with_type_acc(query_dict, pred_dict, ref_dict, id_dict, prefix)
    em = calculate_exact_match(pred_dict, ref_dict)
    print('EM: {:.3f}%'.format(em))
    # calculate_sketch_type_acc(ref_file, pred_file)
    # calculate_exact_match_for_each_q_type(ref_file, pred_file)


def evaluate(pred_file, ref_file):
    """Evaluation during train process."""
    ref_dict, pred_dict, query_dict, id_dict = build_pred_ref_dict(ref_file, pred_file, ref_file)
    total, acc, scores = res_eval_with_type_acc(query_dict, pred_dict, ref_dict, id_dict, save=False)
    em = calculate_exact_match(pred_dict, ref_dict)
    print('Comp Acc: {:.3f}%\tBleu-4: {:.3f}\tRouge-L: {:.3f}'.format(acc, scores['Bleu-4'], scores['Rouge-L']))
    print('EM: {:.3f}%'.format(em))
    # calculate_sketch_type_acc(ref_file, pred_file)
    # calculate_exact_match_for_each_q_type(ref_file, pred_file)
    return total, acc, scores, em


if __name__ == '__main__':
    res_eval(parse_args())
