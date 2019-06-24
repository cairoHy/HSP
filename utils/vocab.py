import numpy as np
import six
import tensorflow as tf


def load_pretrained_vocab(filename):
    vocab = []
    for line in open(filename, 'r', encoding='utf-8'):
        word, emb = line.strip().split('\t')
        vocab.append(word.strip())
    return vocab


def load_word_matrix(filename):
    word_matrix = []
    for line in open(filename, 'r', encoding='utf-8'):
        word, emb = line.strip().split('\t')
        emb = emb.split()
        word_matrix.append(emb)
    word_matrix = np.array(word_matrix)
    return word_matrix


def load_vocab(filename, params):
    if params.use_pretrained_embedding:
        return load_pretrained_vocab(filename)
    vocab = []
    with tf.gfile.GFile(filename) as fd:
        for line in fd:
            word = line.split('\t')[0].strip()
            vocab.append(word)

    return vocab


def load_simple_vocab(filename):
    vocab = []
    with tf.gfile.GFile(filename) as fd:
        for line in fd:
            word = line.strip()
            vocab.append(word)
    return vocab


def decode_target_ids(decode_ids_list, params, name="target"):
    """decode target ids to string(batch)"""
    decoded = []
    vocab = params.vocabulary[name]

    for decode_ids in decode_ids_list:
        syms = []
        for idx in decode_ids:
            if isinstance(idx, six.integer_types):
                sym = vocab[idx]
            else:
                sym = idx

            if sym == params.eos:
                break
            syms.append(sym)
        decoded.append(syms)

    return decoded


def decode_target_ids_copy(decode_ids_list, batch, params, name="target"):
    """decode target ids to string(batch), with copy mechanism"""
    decoded = []
    extra_vocab_list = batch.source_oovs
    vocab = params.vocabulary[name]

    for decode_ids, extra_vocab in zip(decode_ids_list, extra_vocab_list):
        syms = []
        extended_vocab = vocab + extra_vocab
        for idx in decode_ids:
            if isinstance(idx, six.integer_types):
                sym = extended_vocab[idx]
            else:
                sym = idx

            if sym == params.eos:
                break
            syms.append(sym)
        decoded.append(syms)

    return decoded
