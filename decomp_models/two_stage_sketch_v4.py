import copy

from decomp_models.model import *
from decomp_models.model_utils.attention import *
from decomp_models.model_utils.embedding import get_embedding_may_pretrain, get_embedding
from decomp_models.model_utils.layer import *
from decomp_models.model_utils.module import transformer_encoder, transformer_decoder
from utils.optimize import smooth_cross_entropy
from utils.train_utils import tf_trunct


# noinspection PyAttributeOutsideInit
class TwoStageSketchV4(Model):
    """
    Based on TwoStageSketchV2, do not use copy for sketch1
    ######################### shape notation def
    # b ---> batch_size
    # q ---> word(token) number
        .........q_1 for source
        .........q_2 for target
        .........q_3 for sketch
    # e ---> embedding dim or encoded feature dim
    # v ---> position of the word in vocabulary
    # beam ---> beam_size
    """

    def __init__(self, params, scope, mode='train', initializer=None):
        self.use_copy_in_sketch1 = False
        self.use_copy_in_sketch2 = False
        self.sketch1_loss_alpha = 0.3
        self.sketch2_loss_alpha = 0.1
        self.initializer = initializer
        self.mode = mode
        super(TwoStageSketchV4, self).__init__(params=params, scope=scope)

    def build_graph(self):
        self._setup_hyper_params()
        with tf.variable_scope(self.scope, initializer=self.initializer):
            self._setup_placeholder()
            self._embed()
            self._encode()
            self._sketch_decode_train_stage_1()
            # -----------------------------------------
            self._first_sketch_encode()
            self.prepare_for_sketch_decode_stage_2()
            self._sketch_decode_train_stage_2()
            # -----------------------------------------
            self._second_sketch_encode()
            self.prepare_for_final_decode()
            self._decode_train()
            self._compute_loss()

    def _setup_hyper_params(self):
        self.params = self.parameters
        self.feature = None
        self.hidden_size = self.params.hidden_size
        self.embed_dim = self.params.embed_dim
        self.decode_hidden_dim = self.params.embed_dim
        self.vocab = self.params.vocabulary['source']
        self.vocab_size = len(self.vocab)
        self.sketch_vocab = self.params.vocabulary['sketch']
        self.sketch_vocab_size = len(self.sketch_vocab)

    def _setup_placeholder(self):
        self.src_seq = tf.placeholder(tf.int32, [None, None], name='src_seq')  # (b, q_1)
        self.tgt_seq = tf.placeholder(tf.int32, [None, None], name='tgt_seq')  # (b, q_2)

        self.src_len = tf.placeholder(tf.int32, [None], name='src_len')  # [b]
        self.tgt_len = tf.placeholder(tf.int32, [None], name='tgt_len')  # [b]
        # copy related placeholder
        self.tgt_label = tf.placeholder(tf.int32, [None, None], name='tgt_label')  # (b, q_2)
        self.max_out_oovs = tf.placeholder(tf.int32, [], name='max_out_oovs')  # []
        self.source_ids_oo = tf.placeholder(tf.int32, [None, None], name='source_ids_oo')  # [b, q_1]
        self.tiled_len = tf.shape(self.tgt_seq)[1]
        # annotation
        self.pos_anno = tf.placeholder(tf.int32, [None, None], name='pos_anno')  # [b, q_1]
        # sketch
        self.first_sketch_seq = tf.placeholder(tf.int32, [None, None], name='first_sketch_seq')  # [b, q_3]
        self.first_sketch_label = tf.placeholder(tf.int32, [None, None], name='first_sketch_label')  # [b, q_3]
        self.first_sketch_len = tf.placeholder(tf.int32, [None], name='first_sketch_len')  # [b]
        # second_stage_sketch
        self.second_sketch_seq = tf.placeholder(tf.int32, [None, None], name='second_sketch_seq')  # [b, q_4]
        self.second_sketch_label = tf.placeholder(tf.int32, [None, None], name='second_sketch_label')  # [b, q_4]
        self.second_sketch_len = tf.placeholder(tf.int32, [None], name='second_sketch_len')  # [b]

    def _embed(self):
        with tf.device('/cpu:0'), tf.variable_scope('embedding_layer'):
            self.word_embeddings = get_embedding_may_pretrain(self.vocab_size, self.embed_dim, self.params,
                                                              'word_embeddings', trainable=True)
            self.pos_embeddings = get_embedding(2, self.params.ner_emb_size, 'pos_embeddings')
            self.sketch_embeddings = self.word_embeddings
        # weight matrix of decoder
        self.sketch_decoder_weights = self.sketch_embeddings  # [v_3, e]
        self.decoder_weights = self.word_embeddings  # [v, e]
        self.target_embeddings = self.word_embeddings

    def _encode_func(self, input_ids, input_length, word_embeddings, scope=None, use_pos_anno=False, pos_anno=None):
        """Given input_ids: [b, q], output: [b, q, e]"""
        with tf.variable_scope(scope):
            src_embed = tf.nn.embedding_lookup(word_embeddings, input_ids) * (self.embed_dim ** 0.5)
            if use_pos_anno:
                pos_embed = tf.nn.embedding_lookup(self.pos_embeddings, pos_anno) * (
                        self.params.ner_emb_size ** 0.5)
                src_embed = tf.concat([src_embed, pos_embed], -1)
            src_mask = tf.sequence_mask(input_length,
                                        maxlen=tf.shape(input_ids)[1],
                                        dtype=tf.float32)
            src_embed = src_embed * tf.expand_dims(src_mask, -1)
            bias_shape = word_embeddings.shape[1]
            if use_pos_anno:
                bias_shape += self.params.ner_emb_size
            bias = tf.get_variable("src_language_bias", [bias_shape])
            src_embed = tf.nn.bias_add(src_embed, bias)
            encoder_input = add_timing_signal(src_embed)
            enc_attn_bias = attention_bias(src_mask, "masking")
            if self.params.residual_dropout > 0:
                encoder_input = tf.nn.dropout(encoder_input, 1 - self.params.residual_dropout)
            encoder_output = transformer_encoder(encoder_input, enc_attn_bias,
                                                 self.params)
            return encoder_output, enc_attn_bias

    def _decode_func(self, input_ids, input_length, word_embeddings, decoder_weights, enc_attn_bias, mode, state,
                     vocab_size,
                     use_copy,
                     expand_source_ids_oo=None, max_out_oovs=None, src_mask=None,
                     scope=None, decoder_fn=None, decode_sketch=False):
        with tf.variable_scope(scope):
            decoder_fn = transformer_decoder if not decoder_fn else decoder_fn
            tiled_len = tf.shape(input_ids)[1]
            encoder_output = state['encoder']
            decoder_prev_state = state['decoder'] if mode != 'train' else None  # during train we don't need state
            decode_params = copy.copy(self.params)
            decode_params.hidden_size = self.decode_hidden_dim
            decode_params.embed_dim = self.decode_hidden_dim
            input_mask = tf.sequence_mask(input_length,
                                          maxlen=tf.shape(input_ids)[1],
                                          dtype=tf.float32)
            input_embed = tf.nn.embedding_lookup(word_embeddings, input_ids) * (decode_params.embed_dim ** 0.5)
            input_embed = input_embed * tf.expand_dims(input_mask, -1)
            dec_attn_bias = attention_bias(tf.shape(input_embed)[1], "causal")
            # Shift left
            decoder_input = tf.pad(input_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            decoder_input = add_timing_signal(decoder_input)  # [b * beam, q', e]
            if self.params.residual_dropout > 0:
                decoder_input = tf.nn.dropout(decoder_input, 1.0 - self.params.residual_dropout)
            if mode != 'train':  # during inference, we just use shift left input
                decoder_input = decoder_input[:, -1:, :]
                dec_attn_bias = dec_attn_bias[:, :, -1:, :]

            decoder_output = decoder_fn(decoder_input, encoder_output, dec_attn_bias,
                                        enc_attn_bias, decode_params, state=decoder_prev_state)
            if mode != 'train':
                all_att_weights, decoder_infer_output, decoder_state = decoder_output
                decoder_infer_output = decoder_infer_output[:, -1, :]  # [b * beam, e]
                # [b * beam, v]
                infer_vocab_logits = tf.matmul(decoder_infer_output, decoder_weights, False, True)
                infer_logits = tf.nn.softmax(infer_vocab_logits)
                # we have tiled source_id_oo before feed, so last argument is set to 1
                if use_copy:
                    infer_logits = self.calculate_final_logits(decoder_infer_output, all_att_weights,
                                                               infer_logits,
                                                               expand_source_ids_oo, max_out_oovs, src_mask, vocab_size,
                                                               1)
                log_prob = tf.log(infer_logits)
                # set_shape for decoder state to keep loop shape invariant
                return log_prob, {'encoder': state['encoder'], 'decoder': decoder_state}
            else:
                all_att_weights, decoder_train_output = decoder_output
                # [b, q_2, e] => [b * q_2, v]
                decoder_train_output = tf.reshape(decoder_train_output, [-1, decode_params.hidden_size])
                vocab_logits = tf.matmul(decoder_train_output, decoder_weights, False, True)  # [b * q_2, v]
                logits = tf.nn.softmax(vocab_logits)  # [b * q_2, v]
                if use_copy:
                    logits = self.calculate_final_logits(decoder_train_output, all_att_weights, logits,
                                                         expand_source_ids_oo, max_out_oovs,
                                                         src_mask,
                                                         vocab_size,
                                                         tiled_len)  # [b * q_2, v + v']
                return logits

    def _encode(self):
        self.encoder_output, self.comp_q_attn_bias = self._encode_func(self.src_seq, self.src_len, self.word_embeddings,
                                                                       scope='comp_q_encoder', use_pos_anno=True,
                                                                       pos_anno=self.pos_anno)

    def _sketch_decode_train_stage_1(self):
        state = {'encoder': self.encoder_output}
        src_mask = tf.sequence_mask(self.src_len,
                                    maxlen=tf.shape(self.src_seq)[1],
                                    dtype=tf.float32)
        self.first_sketch_logits = self._decode_func(
            self.first_sketch_seq, self.first_sketch_len, self.sketch_embeddings,
            self.sketch_decoder_weights,
            self.comp_q_attn_bias, 'train', state, self.sketch_vocab_size, use_copy=self.use_copy_in_sketch1,
            expand_source_ids_oo=self.source_ids_oo,
            max_out_oovs=self.max_out_oovs, src_mask=src_mask,
            scope='first_sketch_decoder', decode_sketch=True)

    def _first_sketch_encode(self):
        sketch_mask = tf.sequence_mask(self.first_sketch_len,
                                       maxlen=tf.shape(self.first_sketch_seq)[1],
                                       dtype=tf.float32)  # [b, q_3]
        pos_anno = tf.zeros_like(sketch_mask, dtype=tf.int32)
        self.sketch1_encoder_output, _ = self._encode_func(self.first_sketch_seq, self.first_sketch_len,
                                                           self.sketch_embeddings,
                                                           scope='first_sketch_encoder',
                                                           use_pos_anno=True, pos_anno=pos_anno)
        self.sketch1_concated_encoder_output = self.concat_padded_seq(self.encoder_output, self.sketch1_encoder_output)

    def prepare_for_sketch_decode_stage_2(self):
        src_mask = tf.sequence_mask(self.src_len,
                                    maxlen=tf.shape(self.src_seq)[1],
                                    dtype=tf.float32)
        first_sketch_mask = tf.sequence_mask(self.first_sketch_len,
                                             maxlen=tf.shape(self.first_sketch_seq)[1],
                                             dtype=tf.float32)
        self.src_sketch1_mask = tf.concat([src_mask, first_sketch_mask], -1)  # [b, q_1 + q_3]
        self.src_sketch1_copy_mask = tf.sequence_mask(self.src_len,
                                                      maxlen=tf.shape(self.sketch1_concated_encoder_output)[1],
                                                      dtype=tf.float32)  # [b, q_1 + q_3]
        self.sketch1_enc_attn_bias = attention_bias(self.src_sketch1_mask, "masking")
        self.src_sketch1_ids_oo = tf.concat([self.source_ids_oo, self.first_sketch_seq], axis=-1)  # [b, q_1 + q_3]

    def _sketch_decode_train_stage_2(self):
        state = {'encoder': self.sketch1_concated_encoder_output}

        def transformer_concated_decoder_internal(inputs, memory, bias, mem_bias, params, state=None, scope=None,
                                                  reuse=False):
            return transformer_decoder(inputs, memory, bias, mem_bias, params, state, scope, reuse)

        self.second_sketch_logits = self._decode_func(
            self.second_sketch_seq, self.second_sketch_len, self.sketch_embeddings, self.decoder_weights,
            self.sketch1_enc_attn_bias, 'train', state, self.sketch_vocab_size, use_copy=self.use_copy_in_sketch2,
            expand_source_ids_oo=self.src_sketch1_ids_oo,
            max_out_oovs=self.max_out_oovs, src_mask=self.src_sketch1_mask,
            decoder_fn=transformer_concated_decoder_internal,
            scope='second_sketch_decoder')

    def _second_sketch_encode(self):
        sketch_mask = tf.sequence_mask(self.second_sketch_len,
                                       maxlen=tf.shape(self.second_sketch_seq)[1],
                                       dtype=tf.float32)  # [b, q_3]
        pos_anno = tf.zeros_like(sketch_mask, dtype=tf.int32)
        self.sketch2_encoder_output, _ = self._encode_func(self.second_sketch_seq, self.second_sketch_len,
                                                           self.sketch_embeddings,
                                                           scope='second_sketch_encoder',
                                                           use_pos_anno=True, pos_anno=pos_anno)
        self.concated_encoder_output = tf.concat([self.encoder_output,
                                                  self.sketch1_encoder_output,
                                                  self.sketch2_encoder_output], axis=1)

    def prepare_for_final_decode(self):
        src_mask = tf.sequence_mask(self.src_len,
                                    maxlen=tf.shape(self.src_seq)[1],
                                    dtype=tf.float32)
        sketch1_mask = tf.sequence_mask(self.first_sketch_len,
                                        maxlen=tf.shape(self.first_sketch_seq)[1],
                                        dtype=tf.float32)
        sketch2_mask = tf.sequence_mask(self.second_sketch_len,
                                        maxlen=tf.shape(self.second_sketch_seq)[1],
                                        dtype=tf.float32)
        self.concat_src_mask = tf.concat([src_mask, sketch1_mask, sketch2_mask], -1)  # [b, q_1 + q_3 + q_4]
        self.final_enc_attn_bias = attention_bias(self.concat_src_mask, "masking")
        self.concat_src_ids_oo = tf.concat([self.source_ids_oo, self.first_sketch_seq, self.second_sketch_seq],
                                           axis=-1)  # [b, q_1 + q_3 + q_4]

    def _decode_train(self):
        """During train, calculate loss of different time-steps in one mini-batch at the same time"""

        # the basic idea is, we use golden sketch during train and in order to copy from source
        # we given true mask of decoder to generate right copy weights
        state = {'encoder': self.concated_encoder_output}

        def transformer_concated_decoder_internal(inputs, memory, bias, mem_bias, params, state=None, scope=None,
                                                  reuse=False):
            return transformer_decoder(inputs, memory, bias, mem_bias, params, state, scope, reuse)

        self.final_logits = self._decode_func(
            self.tgt_seq, self.tgt_len, self.target_embeddings, self.decoder_weights,
            self.final_enc_attn_bias, 'train', state, self.vocab_size, use_copy=True,
            expand_source_ids_oo=self.concat_src_ids_oo,
            max_out_oovs=self.max_out_oovs, src_mask=self.concat_src_mask,
            decoder_fn=transformer_concated_decoder_internal,
            scope='final_decoder')

    def _compute_loss(self):
        # label smoothing, first sketch cross-entropy
        self.first_sketch_ce = smooth_cross_entropy(
            self.first_sketch_logits,
            self.first_sketch_label,
            self.params.label_smoothing)
        self.first_sketch_ce = tf.reshape(self.first_sketch_ce, tf.shape(self.first_sketch_label))  # [batch, q_3]
        first_sketch_mask = tf.sequence_mask(self.first_sketch_len,
                                             maxlen=tf.shape(self.first_sketch_seq)[1],
                                             dtype=tf.float32)
        self.first_sketch_loss = tf.reduce_sum(self.first_sketch_ce * first_sketch_mask) / tf.reduce_sum(
            first_sketch_mask)

        # label smoothing, second sketch cross-entropy
        self.second_sketch_ce = smooth_cross_entropy(
            self.second_sketch_logits,
            self.second_sketch_label,
            self.params.label_smoothing)
        self.second_sketch_ce = tf.reshape(self.second_sketch_ce, tf.shape(self.second_sketch_label))  # [batch, q_4]
        second_sketch_mask = tf.sequence_mask(self.second_sketch_len,
                                              maxlen=tf.shape(self.second_sketch_seq)[1],
                                              dtype=tf.float32)  # [b, q_4]
        self.second_sketch_loss = tf.reduce_sum(self.second_sketch_ce * second_sketch_mask) / tf.reduce_sum(
            second_sketch_mask)

        # label smoothing, target cross-entropy
        self.ce = smooth_cross_entropy(
            self.final_logits,
            self.tgt_label,
            self.params.label_smoothing)
        self.ce = tf.reshape(self.ce, tf.shape(self.tgt_label))  # [batch, q_2]
        tgt_mask = tf.sequence_mask(self.tgt_len,
                                    maxlen=tf.shape(self.tgt_seq)[1],
                                    dtype=tf.float32)  # [b, q_2]
        self.tgt_loss = tf.reduce_sum(self.ce * tgt_mask) / tf.reduce_sum(tgt_mask)

        self.loss = self.tgt_loss + \
                    self.sketch1_loss_alpha * self.first_sketch_loss + \
                    self.sketch2_loss_alpha * self.second_sketch_loss

    def decode_infer_stage_1(self, inputs, state):
        # input: complex query encoder output     output: sub-query sketch
        # state['enc']: [b * beam, q_1, e]  ,   state['dec']: [b * beam, q', e]
        # q' = previous decode output length
        # during infer, following graph are constructed using beam search
        target_sequence = inputs['sketch']  # [b * beam, q']
        # trunct word idx, change those greater than vocab_size to zero
        shape = target_sequence.shape
        target_sequence = tf_trunct(target_sequence, self.sketch_vocab_size)
        target_sequence.set_shape(shape)
        target_length = inputs['sketch_length']  # [b * beam]
        src_mask = tf.sequence_mask(self.src_len,
                                    maxlen=tf.shape(self.src_seq)[1],
                                    dtype=tf.float32)
        return self._decode_func(target_sequence, target_length, self.sketch_embeddings,
                                 self.sketch_decoder_weights,
                                 self.comp_q_attn_bias, 'infer', state, self.sketch_vocab_size,
                                 use_copy=self.use_copy_in_sketch1,
                                 expand_source_ids_oo=self.source_ids_oo,
                                 max_out_oovs=self.max_out_oovs, src_mask=src_mask,
                                 scope='first_sketch_decoder', decode_sketch=True)

    def decode_infer_stage_2(self, inputs, state):
        target_sequence = inputs['second_sketch']  # [b * beam, q']
        # trunct word idx, change those greater than vocab_size to zero
        shape = target_sequence.shape
        target_sequence = tf_trunct(target_sequence, self.sketch_vocab_size)
        target_sequence.set_shape(shape)
        target_length = inputs['second_sketch_length']  # [b * beam]
        return self._decode_func(target_sequence, target_length, self.sketch_embeddings,
                                 self.sketch_decoder_weights,
                                 self.sketch1_enc_attn_bias, 'infer', state, self.sketch_vocab_size,
                                 use_copy=self.use_copy_in_sketch2,
                                 expand_source_ids_oo=self.src_sketch1_ids_oo,
                                 max_out_oovs=self.max_out_oovs, src_mask=self.src_sketch1_mask,
                                 scope='second_sketch_decoder', decode_sketch=True)

    def decode_infer(self, inputs, state):
        # state['enc']: [b * beam, q_1, e]  ,   state['dec']: [b * beam, q', e]
        # q' = previous decode output length
        # during infer, following graph are constructed using beam search
        target_sequence = inputs['target']  # [b * beam, q']
        vocab_size = len(self.params.vocabulary['target'])
        # trunct word idx, change those greater than vocab_size to zero
        shape = target_sequence.shape
        target_sequence = tf_trunct(target_sequence, vocab_size)
        target_sequence.set_shape(shape)
        target_length = inputs['target_length']  # [b * beam]

        def transformer_concated_decoder_internal(inputs, memory, bias, mem_bias, params, state=None, scope=None,
                                                  reuse=False):
            return transformer_decoder(inputs, memory, bias, mem_bias, params, state, scope, reuse)

        return self._decode_func(target_sequence, target_length, self.target_embeddings, self.decoder_weights,
                                 self.final_enc_attn_bias, 'infer', state, self.vocab_size, use_copy=True,
                                 expand_source_ids_oo=self.concat_src_ids_oo,
                                 max_out_oovs=self.max_out_oovs,
                                 src_mask=self.concat_src_mask,
                                 decoder_fn=transformer_concated_decoder_internal,
                                 scope='final_decoder')

    def calculate_final_logits(self, decoder_output, all_att_weights, vocab_probs, source_ids_oo, max_out_oovs,
                               src_mask,
                               vocab_size, tgt_seq_len):
        # vocab_probs: [b * q_2, v], copy_probs: [b * q_2, q_1 + q_3]
        avg_att_weights = all_att_weights[-1]  # [b, q_2, q_1 + q_3]
        # mask
        # avg_att_weights = avg_att_weights * tf.expand_dims(self.concat_src_true_mask, axis=1)
        copy_probs = tf.reshape(avg_att_weights, [-1, tf.shape(src_mask)[1]])  # [b * q_2, q_1 + q_3]
        # calculate copy gate
        p_gen = tf.nn.sigmoid(linear(decoder_output, 1, scope='p_gen'))  # [b * q_2, 1]

        # gate
        vocab_probs = p_gen * vocab_probs  # [b * q_2, v]
        copy_probs = (1 - p_gen) * copy_probs  # [b * q_2, q_1 + q_3]

        extended_vocab_size = tf.add(tf.constant(vocab_size), max_out_oovs)  # []
        b = tf.shape(vocab_probs)[0]  # b * q_2
        extra_zeros = tf.zeros(shape=tf.stack([b, max_out_oovs], axis=0))  # [b * q_2, v']
        vocab_prob_extended = tf.concat(axis=1, values=[vocab_probs, extra_zeros])  # [b * q_2, v + v']
        batch_nums = tf.range(0, limit=tf.shape(vocab_probs)[0])  # [b * q_2]  (0, 1, 2, ...)
        batch_nums = tf.expand_dims(batch_nums, 1)  # [b * q_2, 1]
        attn_len = tf.shape(copy_probs)[1]  # q_1 + q_3
        batch_nums = tf.tile(batch_nums, [1, attn_len])  # [b * q_2, q_1 + q_3]
        # tile source ids oo, [b, q_1 + q_3] => [b * q_2, q_1 + q_3]
        tiled_source_ids_oo = tf.tile(tf.expand_dims(source_ids_oo, 1), [1, tgt_seq_len, 1])
        tiled_source_ids_oo = tf.reshape(tiled_source_ids_oo, [-1, tf.shape(tiled_source_ids_oo)[2]])
        indices = tf.stack((batch_nums, tiled_source_ids_oo), axis=2)  # [b * q_2, q_1 + q_3, 2]
        shape = tf.stack([tf.shape(vocab_probs)[0], extended_vocab_size], axis=0)  # [2]
        attn_prob_projected = tf.scatter_nd(indices, copy_probs, shape)  # [b * q_2, v + v']
        logits = vocab_prob_extended + attn_prob_projected  # [b * q_2, v + v']
        return logits

    def make_feed_dict(self, batch):
        self.mode = 'train'
        self.params = copy.copy(self.parameters)
        feed_dict = {self.src_seq: batch.source_ids,
                     self.tgt_seq: batch.target_ids,
                     self.tgt_label: batch.target_ids_oo,
                     self.src_len: batch.source_len,
                     self.tgt_len: batch.target_len,
                     self.source_ids_oo: batch.source_ids_oo,
                     self.max_out_oovs: batch.max_oov_num,
                     self.pos_anno: batch.pos_anno,
                     self.first_sketch_seq: batch.sub_query_ids,
                     self.first_sketch_len: batch.sub_query_len,
                     self.first_sketch_label: batch.sub_query_ids,
                     self.second_sketch_seq: batch.sketch_ids,
                     self.second_sketch_len: batch.sketch_len,
                     self.second_sketch_label: batch.sketch_ids}
        return feed_dict

    def make_stage1_infer_feed_dict(self, batch):
        self.mode = 'infer'
        self.params = copy.copy(self.parameters)
        self.params.residual_dropout = 0.0
        self.params.attention_dropout = 0.0
        self.params.relu_dropout = 0.0
        self.params.label_smoothing = 0.0
        feed_dict = {self.src_seq: batch.source_ids,
                     self.src_len: batch.source_len,
                     self.source_ids_oo: batch.source_ids_oo,
                     self.max_out_oovs: batch.max_oov_num,
                     self.pos_anno: batch.pos_anno}
        return feed_dict

    def make_stage2_infer_feed_dict(self, batch):
        self.mode = 'infer'
        self.params = copy.copy(self.parameters)
        self.params.residual_dropout = 0.0
        self.params.attention_dropout = 0.0
        self.params.relu_dropout = 0.0
        self.params.label_smoothing = 0.0
        feed_dict = {self.src_seq: batch.source_ids,
                     self.src_len: batch.source_len,
                     self.source_ids_oo: batch.source_ids_oo,
                     self.max_out_oovs: batch.max_oov_num,
                     self.pos_anno: batch.pos_anno,
                     self.first_sketch_seq: batch.sketch_ids,
                     self.first_sketch_len: batch.sketch_len}
        return feed_dict

    def make_infer_feed_dict(self, batch):
        self.mode = 'infer'
        self.params = copy.copy(self.parameters)
        self.params.residual_dropout = 0.0
        self.params.attention_dropout = 0.0
        self.params.relu_dropout = 0.0
        self.params.label_smoothing = 0.0
        feed_dict = {self.src_seq: batch.source_ids,
                     self.src_len: batch.source_len,
                     self.source_ids_oo: batch.source_ids_oo,
                     self.max_out_oovs: batch.max_oov_num,
                     self.pos_anno: batch.pos_anno,
                     self.first_sketch_seq: batch.sketch_ids,
                     self.first_sketch_len: batch.sketch_len,
                     self.second_sketch_seq: batch.second_sketch_ids,
                     self.second_sketch_len: batch.second_sketch_len}
        return feed_dict

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            bos="<BOS>",
            eos="<EOS>",
            unk="<UNK>",
            eosId=0,
            unkId=1,
            bosId=2,
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.1,
            residual_dropout=0.1,
            relu_dropout=0.1,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            layer_preprocess="layer_norm",
            # extra hyper-parameters
            use_copy=True,  # use copy mechanism in decoder
            bidirectional=False,  # source -> target & target -> source
            bidirection_sub_query=False,  # source -> sub_query1 # sub_query2 & source -> sub_query2 # sub_query1
            use_pos=True,  # use POS annotation
            ner_emb_size=30,  # the dimension of ner feature
            use_pretrained_embedding=True,  # use pretrained word embedding
            embed_dim=300,  # pretrained word embedding dimension
            multi_step_inference=True,  # use multi step inference
            hierarchical_inference=True
        )

        return params

    @staticmethod
    def concat_padded_seq(encoder_output, sketch_encoder_output):
        concated = tf.concat([encoder_output, sketch_encoder_output], axis=1)  # [b, q_1 + q_3, e]
        # concated_list = tf.unstack(concated)  # [q_1 + q_3, e] list
        # zero = tf.constant(0, dtype=tf.int32)
        # indices = [tf.concat([tf.where(tf.not_equal(ex, zero)), tf.where(tf.equal(ex, zero))], 0) for ex in
        #            concated_list]
        # indices = tf.stack(indices)  # [b, q_1 + q_3]
        # bbb = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(4), 1), (1, 13)), -1)
        # indices = tf.concat([bbb, indices], -1)
        return concated
