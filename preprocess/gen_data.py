"""Build vocab and generate train/dev/test data for query decomposition"""
import argparse
import json
import sys

import os

cur_dir = os.path.dirname(os.path.realpath(__file__))
par_dir = os.path.dirname(cur_dir)
sys.path.append(par_dir)
from utils.query_preprocess import preprocess_query, preprocess_sparql, extract_sketch_from_sparql


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Preprocess, generate data for ComplexWebQ",
    )

    parser.add_argument("--ori_data_path", type=str,
                        default="/data/complexwebquestions/golden_supervision/",
                        help="Prefix of dataset path")

    parser.add_argument("--ori_complex_data_path", type=str,
                        default="/data/complexwebquestions/",
                        help="Prefix of `the` original dataset path, we need sparql in the data")

    parser.add_argument("--target_data_path", type=str,
                        default="/data/complexwebquestions/",
                        help="Prefix of target dataset path")
    return parser.parse_args(args)


class DataGen:
    def __init__(self, args):
        self.ori_files = ['ComplexWebQuestions_train.json', 'ComplexWebQuestions_dev.json',
                          'ComplexWebQuestions_test.json']
        self.files = ['train.json', 'dev.json', 'test.json']
        self.very_ori_prefix = args.ori_complex_data_path
        self.ori_prefix = args.ori_data_path
        self.transformer_data_prefix = args.target_data_path

    def generate(self):
        self.gen_all_transformer_data()
        print('Generation Done.')

    @staticmethod
    def gen_transform_data(ori_data_file, data_file, target_file):
        with open(ori_data_file, 'r', encoding='utf-8') as outfile:
            ori_dataset = json.load(outfile)
        sparql = {}
        for item in ori_dataset:
            sparql[item['ID']] = item['sparql']
        with open(data_file, 'r', encoding='utf-8') as outfile:
            split_dataset = json.load(outfile)

        with open(target_file + '-src.json', 'w', encoding='utf-8') as fout:
            for i, sample in enumerate(split_dataset):
                fout.write(preprocess_query(sample['question']))
                fout.write('\n')
        with open(target_file + '-tgt.json', 'w', encoding='utf-8') as fout:
            for i, sample in enumerate(split_dataset):
                fout.write(sample['comp'] + ' # ')
                fout.write(preprocess_query(str(sample['split_part1'])) + ' # ')
                fout.write(preprocess_query(str(sample['split_part2'])))
                fout.write('\n')
        with open(target_file + '.lf', 'w', encoding='utf-8') as fout:
            for i, sample in enumerate(split_dataset):
                preprocessed_sparql = preprocess_sparql(sparql[sample['ID']])
                fout.write(preprocessed_sparql)
                fout.write('\n')
        with open(target_file + '.sketch', 'w', encoding='utf-8') as fout:
            for i, sample in enumerate(split_dataset):
                sketch = extract_sketch_from_sparql(preprocess_sparql(sparql[sample['ID']]))
                sketch = sample['comp'] + ' # ' + sketch
                fout.write(sketch)
                fout.write('\n')

    def gen_all_transformer_data(self):
        print('Generate data for [transformer]...')
        for in_file, in_ori_file in zip(self.files, self.ori_files):
            ori_file = os.path.join(self.ori_prefix, in_file)
            ver_ori_file = os.path.join(self.very_ori_prefix, in_ori_file)
            out_file = os.path.join(self.transformer_data_prefix, in_file.split('.')[0])
            print('Generating: {}, {} -> ( {}-[src.json, tgt.json, .lf, .sketch] )...'.format(ver_ori_file,
                                                                                              ori_file,
                                                                                              out_file))
            self.gen_transform_data(ver_ori_file, ori_file, out_file)


if __name__ == '__main__':
    data_gen = DataGen(parse_args())
    data_gen.generate()
