import queue
import time
from collections import defaultdict
from threading import Thread

import numpy as np
import os
import tensorflow as tf
from random import shuffle

from decomp_models.model_utils.copy_mechanism import copy_mechanism_preprocess


class Sample(object):
    """Class representing a train/dev/test sample."""

    def __init__(self, sample, params, mode):
        self.params = params
        self.mode = mode

        # Process the source sequence
        self.source_ids = sample['source_ids']
        self.source_len = sample['source_length']
        self.source_ids_oo = sample['source_ids_oo']
        self.source_oovs = sample['oov_str']
        self.source_oov_num = sample['oov_num']
        self.pos_anno = sample['pos_anno']
        # Store the original source strings
        self.original_source = sample['source']

        if self.mode == 'train':
            # Process the target sequence
            self.target_ids = sample['target_ids']
            self.target_ids_oo = sample['target_ids_oo']
            self.target_len = sample['target_length']
            # Store the original target strings
            self.original_target = sample['target']
            # Process logic form
            self.logic_form = sample['logic_form']
            self.lf_len = sample['lf_len']
            # Process sketch
            self.sketch_ids = sample['sketch_ids']
            self.sketch_len = sample['sketch_len']
            # Process sub-query
            self.sub_query_ids = sample['sub_query_ids']
            self.sub_query_len = sample['sub_query_len']
            self.sub_query_label = sample['sub_query_ids_oo']


# noinspection PyAttributeOutsideInit
class Batch(object):
    """Class representing a minibatch of train samples."""

    def __init__(self, sample_list, params, mode):
        """Turns the sample_list into a Batch object."""
        self.params = params
        self.mode = mode
        self.max_lf_len = 80
        self.max_tgt_len = 80
        self.vocab = params.vocabulary['target']
        self.vocab_size = len(self.vocab)
        self.pad_id = params.eosId
        self.init_encoder_seq(sample_list)  # initialize the input to the encoder
        if self.mode == 'train':
            self.init_decoder_seq(sample_list)  # initialize the input and targets for the decoder
        self.store_orig_strings(sample_list)  # store the original strings

    def add_sketch_feature(self, sketch_ids_batch_list):
        """Add sketch to features"""
        # remove 0 in the tail
        sketch_ids_batch_list = [np.trim_zeros(np.array(sketch), 'b').tolist() for sketch in
                                 sketch_ids_batch_list]
        # if copy in sketch, we turn it to unk
        sketch_ids_batch_list = [[ids if ids < self.vocab_size else self.params.unkId for ids in sketch] for sketch in
                                 sketch_ids_batch_list]
        # group
        self.sketch_ids = sketch_ids_batch_list
        self.sketch_len = [len(sketch) for sketch in sketch_ids_batch_list]
        # pad sketch
        max_sketch_len = max(self.sketch_len)
        self.sketch_ids = [(ids + [self.pad_id] * (max_sketch_len - len(ids)))[: max_sketch_len]
                           for ids in self.sketch_ids]
        # convert to np array
        self.sketch_ids = np.array(self.sketch_ids, dtype=np.int32)
        self.sketch_len = np.array(self.sketch_len, dtype=np.int32)

    def add_second_sketch_feature(self, sketch_ids_batch_list):
        """Add second sketch to features"""
        # remove 0 in the tail
        sketch_ids_batch_list = [np.trim_zeros(np.array(sketch), 'b').tolist() for sketch in
                                 sketch_ids_batch_list]
        # if copy in sketch, we turn it to unk
        sketch_ids_batch_list = [[ids if ids < self.vocab_size else self.params.unkId for ids in sketch] for sketch in
                                 sketch_ids_batch_list]
        # group
        self.second_sketch_ids = sketch_ids_batch_list
        self.second_sketch_len = [len(sketch) for sketch in sketch_ids_batch_list]
        # pad sketch
        max_sketch_len = max(self.second_sketch_len)
        self.second_sketch_ids = [(ids + [self.pad_id] * (max_sketch_len - len(ids)))[: max_sketch_len]
                                  for ids in self.second_sketch_ids]
        # convert to np array
        self.second_sketch_ids = np.array(self.second_sketch_ids, dtype=np.int32)
        self.second_sketch_len = np.array(self.second_sketch_len, dtype=np.int32)

    def init_encoder_seq(self, sample_list):
        # group
        self.source_ids = [ex.source_ids for ex in sample_list]
        self.source_len = [ex.source_len for ex in sample_list]
        self.source_ids_oo = [ex.source_ids_oo for ex in sample_list]
        self.source_oov_num = [ex.source_oov_num for ex in sample_list]
        self.pos_anno = [ex.pos_anno for ex in sample_list]

        # pad
        max_src_len = max(self.source_len)
        self.source_ids = [(ids + [self.pad_id] * (max_src_len - len(ids)))[: max_src_len]
                           for ids in self.source_ids]
        self.source_ids_oo = [(ids + [self.pad_id] * (max_src_len - len(ids)))[: max_src_len]
                              for ids in self.source_ids_oo]
        self.pos_anno = [(ids + [self.pad_id] * (max_src_len - len(ids)))[: max_src_len]
                         for ids in self.pos_anno]

        # to numpy array
        self.source_ids = np.array(self.source_ids, dtype=np.int32)
        self.source_ids_oo = np.array(self.source_ids_oo, dtype=np.int32)
        self.source_len = np.array(self.source_len, dtype=np.int32)
        self.pos_anno = np.array(self.pos_anno, dtype=np.int32)

        # Determine the max number of in-article OOVs in this batch
        self.max_oov_num = max([len(ex.source_oovs) for ex in sample_list])
        # Store the in-article OOVs themselves
        self.source_oovs = [ex.source_oovs for ex in sample_list]

    def init_decoder_seq(self, sample_list):
        # group
        self.target_ids = [ex.target_ids for ex in sample_list]
        self.target_len = [ex.target_len for ex in sample_list]
        self.target_ids_oo = [ex.target_ids_oo for ex in sample_list]
        self.logic_form_ids = [ex.logic_form for ex in sample_list]
        self.lf_len = [ex.lf_len for ex in sample_list]
        self.sketch_ids = [ex.sketch_ids for ex in sample_list]
        self.sketch_len = [ex.sketch_len for ex in sample_list]
        self.sub_query_ids = [ex.sub_query_ids for ex in sample_list]
        self.sub_query_len = [ex.sub_query_len for ex in sample_list]
        self.sub_query_label = [ex.sub_query_label for ex in sample_list]

        # pad
        max_tgt_len = min(max(self.target_len), self.max_tgt_len)
        self.target_ids = [(ids + [self.pad_id] * (max_tgt_len - len(ids)))[: max_tgt_len]
                           for ids in self.target_ids]
        self.target_ids_oo = [(ids + [self.pad_id] * (max_tgt_len - len(ids)))[: max_tgt_len]
                              for ids in self.target_ids_oo]

        # pad logic_form
        max_lf_len = min(max(self.source_len), self.max_lf_len)
        self.logic_form_ids = [(ids + [self.pad_id] * (max_lf_len - len(ids)))[: max_lf_len]
                               for ids in self.logic_form_ids]

        # pad sketch
        max_sketch_len = max(self.sketch_len)
        self.sketch_ids = [(ids + [self.pad_id] * (max_sketch_len - len(ids)))[: max_sketch_len]
                           for ids in self.sketch_ids]

        # pad sub_query
        max_sub_query_len = max(self.sub_query_len)
        self.sub_query_ids = [(ids + [self.pad_id] * (max_sub_query_len - len(ids)))[: max_sub_query_len]
                              for ids in self.sub_query_ids]
        self.sub_query_label = [(ids + [self.pad_id] * (max_sub_query_len - len(ids)))[: max_sub_query_len]
                                for ids in self.sub_query_label]

        # to numpy array
        self.target_ids = np.array(self.target_ids, dtype=np.int32)
        self.target_ids_oo = np.array(self.target_ids_oo, dtype=np.int32)
        self.target_len = np.array(self.target_len, dtype=np.int32)
        self.logic_form_ids = np.array(self.logic_form_ids, dtype=np.int32)
        self.lf_len = np.array(self.lf_len, dtype=np.int32)
        self.sketch_ids = np.array(self.sketch_ids, dtype=np.int32)
        self.sketch_len = np.array(self.sketch_len, dtype=np.int32)
        self.sub_query_ids = np.array(self.sub_query_ids, dtype=np.int32)
        self.sub_query_len = np.array(self.sub_query_len, dtype=np.int32)
        self.sub_query_label = np.array(self.sub_query_label, dtype=np.int32)

    def store_orig_strings(self, sample_list):
        """Store the original strings in the Batch object"""
        self.original_source = [ex.original_source for ex in sample_list]  # list of lists
        if self.mode == 'train':
            self.original_target = [ex.original_target for ex in sample_list]  # list of lists


class Batcher(object):
    """A class to generate minibatches of data. Buckets samples together based on length of the encoder sequence."""

    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold

    def __init__(self, params, mode='train'):
        """Initialize the batcher. Start threads that process the data into batches."""
        self.mode = mode
        self.params = params
        self.batch_num = 0
        if mode != 'train':
            self.params.batch_size = self.params.decode_batch_size
        self.vocab = defaultdict(lambda: self.params.unkId)
        self.vocab.update({word: i for i, word in enumerate(self.params.vocabulary["target"])})
        self.lf_vocab = defaultdict(lambda: self.params.unkId)
        self.lf_vocab.update({word: i for i, word in enumerate(self.params.vocabulary["lf"])})
        self.sketch_vocab = defaultdict(lambda: self.params.unkId)
        self.sketch_vocab.update({word: i for i, word in enumerate(self.params.vocabulary["sketch"])})

        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
        self._sample_queue = queue.Queue(self.BATCH_QUEUE_MAX * self.params.batch_size)

        if self.mode != 'train':
            self._num_sample_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch samples
            self._bucketing_cache_size = 1  # this essentially means no bucketing
            self._finished_reading = False  # this will tell us when we're finished reading the dataset
        else:
            self._num_sample_q_threads = 16  # num threads to fill sample queue
            self._num_batch_q_threads = 4  # num threads to fill batch queue
            self._bucketing_cache_size = 100  # how many batches-worth of samples to load into cache before bucketing

        # Start the threads that load the queues
        self._sample_q_threads = []
        for _ in range(self._num_sample_q_threads):
            self._sample_q_threads.append(Thread(target=self.fill_sample_queue))
            self._sample_q_threads[-1].daemon = True
            self._sample_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            target = self.fill_batch_queue if self.mode == 'train' else self.fill_infer_batch_queue
            self._batch_q_threads.append(Thread(target=target))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if self.mode == 'train':
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        """Return a Batch from the batch queue"""
        if self._batch_queue.qsize() == 0:
            if self.mode == 'train':
                tf.logging.warning(
                    'Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i',
                    self._batch_queue.qsize(), self._sample_queue.qsize())
            # During infer, If the batch queue is empty, return None
            if self.mode != 'train' and self._finished_reading:
                return None
        self.batch_num += 1
        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def fill_sample_queue(self):
        """Reads data from file and processes into Examples which are then placed into the sample queue."""
        sample_gen = self.sample_generator() if self.mode == 'train' else self.sample_infer_generator()
        while True:
            try:
                sample = sample_gen.__next__()
            except StopIteration:  # if there are no more samples:
                if self.mode == 'train':
                    tf.logging.info("The sample generator for this sample queue filling thread has exhausted data.")
                    raise Exception("single_pass mode is off but the sample generator is out of data; error.")
                else:
                    self._finished_reading = True
                    break

            sample = Sample(sample, self.params, self.mode)
            self._sample_queue.put(sample)  # place the Sample in the sample queue.

    def fill_batch_queue(self):
        """
        Takes Examples out of sample queue, sorts them by encoder sequence length,
        processes into Batches and places them in the batch queue.
        """
        while True:
            # Get bucketing_cache_size-many batches of Examples into a list, then sort
            inputs = []
            for _ in range(self.params.batch_size * self._bucketing_cache_size):
                inputs.append(self._sample_queue.get())
            inputs = sorted(inputs, key=lambda inp: inp.source_len)  # sort by length of encoder sequence

            # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
            batches = []
            for i in range(0, len(inputs), self.params.batch_size):
                batches.append(inputs[i:i + self.params.batch_size])
            shuffle(batches)
            for b in batches:  # each b is a list of Example objects
                self._batch_queue.put(Batch(b, self.params, self.mode))

    def fill_infer_batch_queue(self):
        inputs = []
        while True:
            try:
                # Get bucketing_cache_size-many batches of Examples into a list
                for _ in range(self.params.decode_batch_size * self._bucketing_cache_size):
                    inputs.append(self._sample_queue.get(timeout=1))
                # Group Samples into batches, place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self.params.decode_batch_size):
                    batches.append(inputs[i:i + self.params.decode_batch_size])
            except queue.Empty:
                if inputs:  # get data in the tail
                    batches = [inputs]
                else:
                    continue
            for b in batches:  # each b is a list of Example objects
                self._batch_queue.put(Batch(b, self.params, self.mode))
            inputs = []

    def watch_threads(self):
        """Watch sample queue and batch queue threads and restart if dead."""
        while True:
            time.sleep(60)
            for idx, t in enumerate(self._sample_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found sample queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_sample_queue)
                    self._sample_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def get_files_opener(self):
        portion = ''
        src, tgt = self.params.input[0], self.params.input[1]
        src = self.params.input[0] if self.mode == 'train' else self.params.input
        src_file, src_dir = os.path.basename(src), os.path.dirname(src)
        src_file_prefix = str(src_file.split('-')[0])
        pos_file = os.path.join(src_dir, 'anno', src_file + '.pos')
        lf_file = os.path.join(src_dir, src_file_prefix + '.lf.shuf')
        sketch_file = os.path.join(src_dir, src_file_prefix + '.sketch.shuf')
        sub_query_file = os.path.join(src_dir, src_file_prefix + '-tgt.json.shuf')
        files = [src, tgt, pos_file, lf_file, sketch_file, sub_query_file] if self.mode == 'train' else [
            self.params.input,
            pos_file]
        if self.mode == 'train':
            files = [file + portion for file in files]
        tf.logging.info('All files input: {}'.format(files))
        files = [open(file, 'r', encoding='utf-8') for file in files]
        return files

    def sample_generator(self):
        """Read data from train file"""
        dataset = []
        files = self.get_files_opener()
        tgt_oov_sample_num = 0
        type_num = {'conjunction': 0, 'composition': 0, 'superlative': 0, 'comparative': 0}
        for source, target, pos_anno, logic_form, sketch, sub_query in zip(*files):
            q_type = sketch.split('#')[0].strip()
            type_num[q_type] += 1
            # strip and split
            source, target, pos_anno = source.strip().split(), target.strip().split(), pos_anno.strip().split()
            sub_query = sub_query.strip().split()
            logic_form, sketch = logic_form.strip().split(), sketch.strip().split()
            # convert to ids format
            source_ids = [self.vocab.get(i, self.params.unkId) for i in source]
            target_ids = [self.vocab.get(i, self.params.unkId) for i in target]
            pos_anno = [int(i) for i in pos_anno]
            logic_form_ids = [self.lf_vocab.get(i, self.params.unkId) for i in logic_form]
            sketch_ids = [self.sketch_vocab.get(i, self.params.unkId) for i in sketch]
            sub_query_ids = [self.vocab.get(i, self.params.unkId) for i in sub_query]
            source_ids_oo, src_oovs, target_ids_oo = copy_mechanism_preprocess(source, target, self.params, self.vocab)
            _, _, sub_query_ids_oo = copy_mechanism_preprocess(source, sub_query, self.params, self.vocab)
            # add eos in the end
            source_ids += [self.params.eosId]
            target_ids += [self.params.eosId]
            source_ids_oo += [self.params.eosId]
            target_ids_oo += [self.params.eosId]
            pos_anno += [0]
            sketch_ids += [self.params.eosId]
            logic_form_ids += [self.params.eosId]
            sub_query_ids += [self.params.eosId]
            sub_query_ids_oo += [self.params.eosId]
            # assertion
            assert len(source_ids) == len(source_ids_oo)
            assert len(target_ids) == len(target_ids_oo)
            assert len(sub_query_ids) == len(sub_query_ids_oo)
            assert len(source_ids) == len(pos_anno), "len source: {}, len pos: {}".format(len(source_ids),
                                                                                          len(pos_anno))
            if self.params.unkId in target_ids_oo:
                tgt_oov_sample_num += 1
            # assert self.params.unkId not in target_ids_oo, "train target label has UNK_id: {}".format(target_ids_oo)
            # append
            dataset.append({
                "source": source,
                "target": target,
                "source_ids": source_ids,
                "target_ids": target_ids,
                "target_ids_oo": target_ids_oo,
                "source_length": len(source_ids),
                "target_length": len(target_ids),
                "source_ids_oo": source_ids_oo,
                "oov_num": len(src_oovs),
                "oov_str": src_oovs,
                "pos_anno": pos_anno,
                "logic_form": logic_form_ids,
                "lf_len": len(logic_form_ids),
                "sketch_ids": sketch_ids,
                "sketch_len": len(sketch_ids),
                "sub_query_ids": sub_query_ids,
                "sub_query_len": len(sub_query_ids),
                "sub_query_ids_oo": sub_query_ids_oo
            })
        for key in type_num.keys():
            tf.logging.info('{}: {}'.format(key, type_num[key]))
        tf.logging.info('Train samples: {}'.format(sum(type_num.values())))
        max_target_len = max(sample['target_length'] for sample in dataset)
        max_source_len = max(sample['source_length'] for sample in dataset)
        max_lf_len = max(sample['lf_len'] for sample in dataset)
        tf.logging.info('Max source length is {}'.format(max_source_len))
        tf.logging.info('Max target length is {}'.format(max_target_len))
        tf.logging.info('Max lf length is {}'.format(max_lf_len))
        tf.logging.info('Total train tgt oov sample {}'.format(tgt_oov_sample_num))

        while True:
            idx = np.random.permutation(len(dataset))
            for i in idx:
                yield dataset[i]

    def sample_infer_generator(self):
        """Read data from inference file"""
        dataset = []
        files = self.get_files_opener()
        for source, pos_anno in zip(*files):
            # strip and split str
            source, pos_anno = source.strip().split(), pos_anno.strip().split()
            # convert to ids format
            source_ids = [self.vocab.get(i, self.params.unkId) for i in source]
            pos_anno = [int(i) for i in pos_anno]
            source_ids_oo, src_oovs, target_ids_oo = copy_mechanism_preprocess(source, '', self.params, self.vocab)
            # add eos in the end
            source_ids += [self.params.eosId]
            source_ids_oo += [self.params.eosId]
            pos_anno += [0]
            # assertion
            assert len(source_ids) == len(source_ids_oo)
            assert len(source_ids) == len(pos_anno)
            # append
            dataset.append({
                "source": source,
                "source_ids": source_ids,
                "source_length": len(source_ids),
                "source_ids_oo": source_ids_oo,
                "oov_num": len(src_oovs),
                "oov_str": src_oovs,
                "pos_anno": pos_anno,
            })
        for i in range(len(dataset)):
            yield dataset[i]
