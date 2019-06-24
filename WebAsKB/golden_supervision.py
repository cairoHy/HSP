import json
import unicodedata

from pycorenlp import StanfordCoreNLP
import pandas as pd

from config import *


class GoldenSupervision():
    def __init__(self):
        self.load_data()
        self.nlp = StanfordCoreNLP(config.StanfordCoreNLP_Path)

    def load_data(self):
        # loading webcomplexquestions
        with open(config.complexwebquestions_dir + 'ComplexWebQuestions_' + config.EVALUATION_SET + '.json') as f:
            questions = json.load(f)
        print(len(questions))
        print(pd.DataFrame(questions)['compositionality_type'].value_counts())

        # aliases version
        compWebQ = pd.DataFrame(
            [{'ID': question['ID'], 'question': question['question'], 'webqsp_question': question['webqsp_question'], \
              'machine_question': question['machine_question'], 'comp': question['compositionality_type'], \
              } for question in questions])
        print(compWebQ['comp'].value_counts())

        self.compWebQ = compWebQ.to_dict(orient="rows")

    def calc_split_point(self, question):
        question['question'] = question['question'].replace('?', '').replace('.', '')
        question['machine_question'] = question['machine_question'].replace('?', '').replace('.', '')
        machine_annotations = self.annotat(question['machine_question'], annotators='tokenize')
        webqsp_annotations = self.annotat(question['webqsp_question'], annotators='tokenize')
        question['machine_tokens'] = machine_annotations
        question['webqsp_tokens'] = webqsp_annotations

        # calculating original split point
        org_q_vec = question['webqsp_tokens']
        machine_q_vec = question['machine_tokens']
        org_q_offset = 0

        for word in machine_q_vec:
            if org_q_offset < len(org_q_vec) and org_q_vec[org_q_offset] == word:
                org_q_offset += 1
            else:
                break

        # adding split_point2 for composition
        if question['comp'] == 'composition':
            org_q_offset2 = len(machine_q_vec) - 1
            for word in org_q_vec[::-1]:
                if org_q_offset2 > 0 and machine_q_vec[org_q_offset2] == word:
                    org_q_offset2 -= 1
                else:
                    break
            if org_q_offset2 != len(machine_q_vec) - 1:
                question['split_point2'] = org_q_offset2
            else:
                question['split_point2'] = org_q_offset2

            question['machine_comp_internal'] = ' '.join(
                question['machine_tokens'][org_q_offset:question['split_point2'] + 1])

        question['split_point'] = org_q_offset
        if question['split_point'] == 0:
            question['split_point'] = 1

        org_q_offset = 0
        new_part = []
        for word in question['machine_tokens']:
            if org_q_offset < len(question['webqsp_tokens']) and question['webqsp_tokens'][org_q_offset] == word:
                org_q_offset += 1
            else:
                new_part.append(word)

        question['split_point'] = org_q_offset
        question['new_part'] = ' '.join(new_part)
        return question

    # Generating golden supervision
    def gen_golden_supervision(self):
        qind = 0
        num_q_to_proc = len(self.compWebQ)
        for question in self.compWebQ[0:num_q_to_proc]:

            # print question
            qind += 1
            if qind % 100 == 0:
                print(qind)

            if question['comp'] is None or question['comp'] in ['comparative', 'superlative']:
                continue

            question = self.calc_split_point(question)
            mg_question = question['machine_question'].split()

            if question['split_point'] == 0:
                question['split_point'] = 1

            question['flip_rephrase'] = 0
            if question['comp'] == 'conjunction':
                tokens_anno = self.annotat(' '.join(mg_question))
                question['machine_comp_internal'] = ''
                s = question['split_point']
                question['split_part1'] = ' '.join(mg_question[:s])
                question['split_part2'] = mg_question[s:]
                if question['split_part2'][0] == 'and':  # delete conjunction word
                    question['split_part2'] = question['split_part2'][1:]
                # add wh- and nouns of first part
                head_part = []
                for i in range(len(tokens_anno)):
                    # if we meet a verb, or a that(WDT) in the middle, we break
                    if 'V' in tokens_anno[i]['pos'] or ('WDT' in tokens_anno[i]['pos'] and i != 0):
                        break
                    else:
                        head_part.append(mg_question[i])
                question['split_part2'] = ' '.join(head_part + question['split_part2'])
            else:
                if question['split_point2'] <= question['split_point']:
                    print('found error in split point 2')
                    question['split_point2'] = question['split_point'] = 1
                s1, s2 = question['split_point'], question['split_point2']
                question['split_part1'] = question['machine_comp_internal']
                question['split_part2'] = ' '.join(mg_question[:s1] + ['%composition', ] + mg_question[s2 + 1:])
            # print('{}[{}]\n[{}]\n[{}]\n{}'.format(question['comp'], ' '.join(mg_question),
            #                                     question['split_part1'], question['split_part2'], '-' * 100))

        out = pd.DataFrame(self.compWebQ[0:num_q_to_proc])[
            ['ID', 'comp', 'flip_rephrase', 'split_part1', 'machine_comp_internal', 'split_part2', 'question',
             'machine_question']]

        with open(config.golden_supervision_dir + config.EVALUATION_SET + '.json', 'w') as outfile:
            json.dump(out.to_dict(orient="rows"), outfile, sort_keys=True, indent=4)

    def annotat(self, text, annotators='pos'):
        question = text.replace('?', '')

        text = unicodedata.normalize('NFKD', question).encode('ascii', 'ignore').decode(encoding='UTF-8')

        output = self.nlp.annotate(text, properties={
            'annotators': annotators,
            'outputFormat': 'json'
        })
        try:
            tokens_anno = output['sentences'][0]['tokens']
        except KeyError:
            tokens_anno = [k['word'] for k in output['tokens']]
        return tokens_anno
