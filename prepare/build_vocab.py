import argparse
import collections
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Create Vocabulary")

    parser.add_argument("inputfile", help="Dataset")
    parser.add_argument("outputfile", help="Vocabulary Name")
    parser.add_argument("--vocabsize", default=30000, type=int, help="Vocabulary Size")
    parser.add_argument("--threshold", default=0, type=int, help="Threshold of vocab")
    parser.add_argument("--use_pretrain_emb", default=None, type=str, help="Use Pre-trained word embedding")
    parser.add_argument("--emb_dim", default=300, type=int, help="Pre-trained word embedding dimension")

    return parser.parse_args()


def count_words(filename):
    counter = collections.Counter()

    with open(filename, "r") as fd:
        for line in fd:
            words = line.strip().split()
            words = [word.lower() for word in words]
            counter.update(words)

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, counts = list(zip(*count_pairs))

    return words, counts


def save_vocab(filename, vocab):
    pairs = sorted(vocab.items(), key=lambda x: (x[1], x[0]))
    words = list(zip(*pairs))[0]

    with open(filename, "w", encoding='utf-8') as f:
        for word in words:
            f.write(word + "\n")


def save_pretrain_vocab(vocab, args):
    pretrained_word_matrix = {}
    pretrained_words = []
    pretrained_emb_file = os.path.join(os.path.dirname(args.inputfile), 'embed',
                                       'glove.6B.{}d.txt'.format(args.emb_dim))
    for line in open(pretrained_emb_file, 'r', encoding='utf-8'):
        info = line.strip().split()
        word, emb = info[0], info[1:]
        pretrained_words.append(word)
        pretrained_word_matrix[word] = emb

    vocab_num = 0
    pretrained_vocab_num = 0
    pretrained = 0
    with open(args.outputfile, "w", encoding='utf-8') as f:
        for word in vocab.keys():
            vocab_num += 1
            if word in pretrained_word_matrix:
                f.write(word + "\t" + ' '.join(pretrained_word_matrix[word]) + "\n")
                pretrained += 1
                del pretrained_word_matrix[word]
            else:
                emb = ' '.join(str(float(i)) for i in np.random.uniform(-1, 1, size=args.emb_dim))
                f.write(word + "\t" + str(emb) + "\n")
        for word in pretrained_words:
            if vocab_num > args.vocabsize or pretrained_vocab_num > 0:
                break
            if word in pretrained_word_matrix.keys():
                emb = pretrained_word_matrix[word]
                vocab_num += 1
                pretrained_vocab_num += 1
                f.write(word + "\t" + ' '.join(emb) + "\n")
    print('Total words: {}, {}/{} use pre-trained embedding.'.format(vocab_num, pretrained, vocab_num))


def main(args):
    vocab = {}
    count = 0
    words, counts = count_words(args.inputfile)

    vocab["<EOS>"] = 0  # insert end-of-sentence/unknown/begin-of-sentence symbols
    vocab["<UNK>"] = 1
    vocab["<BOS>"] = 2

    for word, freq in zip(words, counts):
        if args.vocabsize and len(vocab) >= args.vocabsize:
            print('Warning: given vocab size < corpus vocab size.')

        if word in vocab:
            print("Warning: found duplicate token %s, ignored" % word)
            continue

        if freq < args.threshold:
            continue

        vocab[word] = len(vocab)
        count += freq

    if args.use_pretrain_emb and args.use_pretrain_emb == 'True':
        save_pretrain_vocab(vocab, args)
    else:
        output_file = os.path.join(os.path.dirname(args.inputfile), os.path.basename(args.outputfile))
        save_vocab(output_file, vocab)

    print("Unique words: %d" % len(words))
    print("Vocabulary coverage: %4.2f" % (100.0 * count / sum(counts)))


if __name__ == "__main__":
    main(parse_args())
