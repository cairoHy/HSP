import argparse

import os
from pycorenlp import StanfordCoreNLP
from tqdm import tqdm


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Preprocess data, add POS annotation.",
    )

    # input files
    parser.add_argument("--prefix", type=str,
                        default="/data/complexwebquestions/",
                        help="Prefix of dataset path")
    parser.add_argument("--ner_server_url", type=str, default="http://localhost:9003",
                        help="Url of NER server")

    return parser.parse_args(args)


class Annotator:
    """Annotate NER and split train/dev/test set using Stanford toolkit."""

    def __init__(self, args):
        self.nlp = StanfordCoreNLP(args.ner_server_url)
        self.prefix = args.prefix
        self.src_fils = ['train-src.json', 'dev-src.json', 'test-src.json']
        self.ner_files = ['train-src.json.pos', 'dev-src.json.pos', 'test-src.json.pos']

    def annotate_question(self, question):
        # removing the question mark
        annotator_props = 'pos'

        output = self.nlp.annotate(question, properties={
            'tokenize.whitespace': True,
            "ssplit.eolonly": True,
            'annotators': annotator_props,
            'outputFormat': 'json'
        })
        res = {'pos': [word['pos'] for word in output['sentences'][0]['tokens']]}
        return res

    def annotate_data(self):
        print('Annotate NER label for complexWebQ data...')
        for ori, anno in zip(self.src_fils, self.ner_files):
            try:
                ori = os.path.join(self.prefix, ori)
                anno = os.path.join(self.prefix, 'anno', anno)
                if not os.path.exists(os.path.join(self.prefix, 'anno')):
                    os.makedirs(os.path.join(self.prefix, 'anno'))
                if os.path.isfile(anno):
                    print('Skip exist annotation file {}'.format(anno))
                    continue
                with open(anno, 'w', encoding='utf-8') as anno_out:
                    for line in tqdm(open(ori, 'r', encoding='utf-8')):
                        data = self.annotate_question(' '.join(line.strip().split()))
                        data['pos'] = [str(1) if 'NN' in label else str(0) for label in data['pos']]
                        assert len(data['pos']) == len(line.strip().split()), 'query is: {}, len is: {}'.format(
                            line.strip().split(), data['pos'])
                        anno_out.write(' '.join(data['pos']) + '\n')
            except FileNotFoundError as why:
                print(why)


if __name__ == '__main__':
    annotator = Annotator(parse_args())
    annotator.annotate_data()
