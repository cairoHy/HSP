def article2ids(article_words, vocab, params):
    """Map the article words to their ids. Also return a list of OOVs in the article.

    Args:
      article_words: list of words (strings)
      vocab: Vocabulary object
      params:

    Returns:
      ids:
        A list of word ids (integers); OOVs are represented by their temporary article OOV number.
        If the vocabulary size is 50k and the article has 3 OOVs,
        then these temporary OOV numbers will be 50000, 50001, 50002.
      oovs:
        A list of the OOV words in the article (strings),
        in the order corresponding to their temporary article OOV numbers.
    """
    ids = []
    oovs = []
    unk_id = params.unkId
    for w in article_words:
        if not w:
            continue
        i = vocab.get(w, unk_id)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(len(vocab) + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def target2ids(target_words, vocab, article_oovs, params):
    ids = []
    unk_id = params.unkId
    for w in target_words:
        if not w:
            continue
        i = vocab.get(w, params.unkId)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = len(vocab) + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def copy_mechanism_preprocess(src_words, target_words, params, vocab):
    # 1. change target sequence, convert the <UNK> id token to its temporary OOV id
    # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id;
    # also store the in-article OOVs words themselves
    src_ids_extend_vocab, src_oovs = article2ids(src_words, vocab, params)

    # Get a version of the reference summary where in-article OOVs are represented by their temporary article OOV id
    tgt_ids_extend_vocab = target2ids(target_words, vocab, src_oovs, params)

    return src_ids_extend_vocab, src_oovs, tgt_ids_extend_vocab


def _dynamic_padding(batch_data, pad_id):
    """Dynamically pads the batch_data with pad_id"""
    pad_source_len = max(len(each) for each in batch_data['source'])
    pad_target_len = max(len(each) for each in batch_data['target'])
    pad_src_oo_len = max(len(each) for each in batch_data['enc_ids_extend_vocab'])
    batch_data['source'] = [(ids + [pad_id] * (pad_source_len - len(ids)))[: pad_source_len]
                            for ids in batch_data['source']]
    batch_data['target'] = [(ids + [pad_id] * (pad_target_len - len(ids)))[: pad_target_len]
                            for ids in batch_data['target']]
    batch_data['enc_ids_extend_vocab'] = [(ids + [pad_id] * (pad_src_oo_len - len(ids)))[: pad_src_oo_len]
                                          for ids in batch_data['enc_ids_extend_vocab']]
    return batch_data
