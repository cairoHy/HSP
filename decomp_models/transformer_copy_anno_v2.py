import copy

from decomp_models.model import *
from decomp_models.model_utils.attention import *
from decomp_models.model_utils.embedding import get_embedding_may_pretrain
from decomp_models.model_utils.layer import *
from decomp_models.model_utils.module import transformer_encoder, transformer_decoder
from utils.optimize import smooth_cross_entropy
from utils.train_utils import tf_trunct


# noinspection PyAttributeOutsideInit
class TransformerCopyAnnoV2(Model):
    """
    Transformer + CopyV2 + pos_annotation + trainable_word_embedding
    ######################### shape notation def
    # b ---> batch_size
    # q ---> word(token) number
        .........q_1 for source
        .........q_2 for target
    # e ---> embedding dim or encoded feature dim
    # v ---> position of the word in vocabulary
    # beam ---> beam_size
    """

    def __init__(self, params, scope, mode='train', initializer=None):
        self.initializer = initializer
        self.mode = mode
        super(TransformerCopyAnnoV2, self).__init__(params=params, scope=scope)

    def build_graph(self):
        self._setup_hyper_params()
        with tf.variable_scope(self.scope, initializer=self.initializer):
            self._setup_placeholder()
            self._embed()
            self._encode()
            self._decode_train()

    def _setup_hyper_params(self):
        self.params = self.parameters
        self.feature = None
        self.hidden_size = self.params.hidden_size
        self.embed_dim = self.params.embed_dim
        self.vocab = self.params.vocabulary['source']
        self.vocab_size = len(self.vocab)

    def _setup_placeholder(self):
        self.src_seq = tf.placeholder(tf.int32, [None, None], name='src_seq')  # (b, q_1)
        self.tgt_seq = tf.placeholder(tf.int32, [None, None], name='tgt_seq')  # (b, q_2)

        self.src_len = tf.placeholder(tf.int32, [None], name='src_len')  # [b]
        self.tgt_len = tf.placeholder(tf.int32, [None], name='tgt_len')  # [b]
        # copy related placeholder
        self.tgt_label = tf.placeholder(tf.int32, [None, None], name='tgt_label')  # (b, q_2)
        self.max_out_oovs = tf.placeholder(tf.int32, [], name='max_out_oovs')  # []
        self.source_ids_oo = tf.placeholder(tf.int32, [None, None], name='source_ids_oo')  # [b, q_1]
        self.src_mask = tf.sequence_mask(self.src_len,
                                         maxlen=tf.shape(self.src_seq)[1],
                                         dtype=tf.float32)  # [b, q_1]
        self.tgt_mask = tf.sequence_mask(self.tgt_len,
                                         maxlen=tf.shape(self.tgt_seq)[1],
                                         dtype=tf.float32)  # [b, q_2]
        self.tiled_len = tf.shape(self.tgt_seq)[1]
        # annotation
        self.pos_anno = tf.placeholder(tf.int32, [None, None], name='pos_anno')  # [b, q_1]

    def _embed(self):
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = get_embedding_may_pretrain(self.vocab_size, self.embed_dim, self.params,
                                                              'word_embeddings', True)
        with tf.variable_scope('pos_embedding'):
            initializer = tf.random_uniform_initializer(-1, 1)
            self.pos_embeddings = tf.get_variable(name='pos_embeddings', shape=(2, self.params.ner_emb_size),
                                                  initializer=initializer, trainable=True)
        # weight matrix of decoder
        self.decoder_weights = self.word_embeddings  # [v, e]

    def _encode(self):
        # [b, q_1, e]
        self.src_embed = tf.nn.embedding_lookup(self.word_embeddings, self.src_seq) * (self.embed_dim ** 0.5)
        self.pos_embed = tf.nn.embedding_lookup(self.pos_embeddings, self.pos_anno) * (self.params.ner_emb_size ** 0.5)
        if self.params.use_pos:
            self.src_embed = tf.concat([self.src_embed, self.pos_embed], -1)  # [b, q_1, e_1 + e_2]
        self.src_embed = self.src_embed * tf.expand_dims(self.src_mask, -1)
        bias_shape = self.embed_dim if not self.params.use_pos else self.embed_dim + self.params.ner_emb_size
        bias = tf.get_variable("src_language_bias", [bias_shape])
        self.src_embed = tf.nn.bias_add(self.src_embed, bias)
        self.encoder_input = add_timing_signal(self.src_embed)
        self.enc_attn_bias = attention_bias(self.src_mask, "masking")
        if self.params.residual_dropout > 0:
            self.encoder_input = tf.nn.dropout(self.encoder_input, 1 - self.params.residual_dropout)
        self.encoder_output = transformer_encoder(self.encoder_input, self.enc_attn_bias, self.params)  # [b, q_1, e]

    def _decode_train(self):
        """During train, calculate loss of different time-steps in one mini-batch at the same time"""
        decode_params = copy.copy(self.params)
        decode_params.hidden_size = self.embed_dim
        # [b, q_2, e]
        self.tgt_embed = tf.nn.embedding_lookup(self.word_embeddings, self.tgt_seq) * (self.embed_dim ** 0.5)
        self.masked_tgt_embed = self.tgt_embed * tf.expand_dims(self.tgt_mask, -1)
        self.dec_attn_bias = attention_bias(tf.shape(self.masked_tgt_embed)[1], "causal")
        self.decoder_input = tf.pad(self.masked_tgt_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]  # Shift left
        self.decoder_input = add_timing_signal(self.decoder_input)
        if self.params.residual_dropout > 0:
            self.decoder_input = tf.nn.dropout(self.decoder_input, 1.0 - self.params.residual_dropout)
        self.all_att_weights, self.decoder_output = transformer_decoder(self.decoder_input, self.encoder_output,
                                                                        self.dec_attn_bias, self.enc_attn_bias,
                                                                        decode_params)
        # [b, q_2, e] => [b * q_2, v]
        self.decoder_output = tf.reshape(self.decoder_output, [-1, decode_params.hidden_size])
        self.vocab_logits = tf.matmul(self.decoder_output, self.decoder_weights, False, True)  # [b * q_2, v]
        self.vocab_probs = tf.nn.softmax(self.vocab_logits)  # [b * q_2, v]
        vocab_size = len(self.params.vocabulary['target'])
        self.logits = self.calculate_final_logits(self.decoder_output, self.all_att_weights, self.vocab_probs,
                                                  self.source_ids_oo, self.max_out_oovs, self.src_mask, vocab_size,
                                                  self.tiled_len)  # [b * q_2, v + v']
        self._compute_loss()

    def _compute_loss(self):
        # label smoothing
        self.ce = smooth_cross_entropy(
            self.logits,
            self.tgt_label,
            self.params.label_smoothing)

        self.ce = tf.reshape(self.ce, tf.shape(self.tgt_seq))  # [batch, q_2]

        self.loss = tf.reduce_sum(self.ce * self.tgt_mask) / tf.reduce_sum(self.tgt_mask)

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
        decode_params = copy.copy(self.params)
        decode_params.hidden_size = self.embed_dim
        tgt_mask = tf.sequence_mask(target_length,
                                    maxlen=tf.shape(target_sequence)[1],
                                    dtype=tf.float32)  # [b * beam, q']
        # [b * beam, q', e]
        self.tgt_embed = tf.nn.embedding_lookup(self.word_embeddings, target_sequence) * (self.embed_dim ** 0.5)
        self.masked_tgt_embed = self.tgt_embed * tf.expand_dims(tgt_mask, -1)
        self.dec_attn_bias = attention_bias(tf.shape(self.masked_tgt_embed)[1], "causal")
        # Shift left
        self.decoder_input = tf.pad(self.masked_tgt_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        self.decoder_input = add_timing_signal(self.decoder_input)  # [b * beam, q', e]
        if self.params.residual_dropout > 0:
            self.decoder_input = tf.nn.dropout(self.decoder_input, 1.0 - self.params.residual_dropout)
        self.infer_decoder_input = self.decoder_input[:, -1:, :]
        self.infer_dec_attn_bias = self.dec_attn_bias[:, :, -1:, :]

        self.all_att_weights, self.decoder_infer_output, self.decoder_state = transformer_decoder(
            self.infer_decoder_input,
            state['encoder'],
            self.infer_dec_attn_bias,
            self.enc_attn_bias,
            decode_params,
            state=state['decoder'])
        self.decoder_infer_output = self.decoder_infer_output[:, -1, :]  # [b * beam, e]
        # [b * beam, v]
        self.infer_vocab_logits = tf.matmul(self.decoder_infer_output, self.decoder_weights, False, True)
        self.infer_vocab_probs = tf.nn.softmax(self.infer_vocab_logits)
        vocab_size = len(self.params.vocabulary['target'])
        logits = self.calculate_final_logits(self.decoder_infer_output, self.all_att_weights, self.infer_vocab_probs,
                                             self.source_ids_oo, self.max_out_oovs, self.src_mask, vocab_size,
                                             1)  # we have tiled source_id_oo before feed, so last argument is set to 1
        self.log_prob = tf.log(logits)
        # set_shape for decoder state to keep loop shape invariant
        return self.log_prob, {'encoder': state['encoder'], 'decoder': self.decoder_state}

    @staticmethod
    def calculate_final_logits(decoder_output, all_att_weights, vocab_probs, source_ids_oo, max_out_oovs, src_mask,
                               vocab_size, tgt_seq_len):
        # select last layer weights
        avg_att_weights = all_att_weights[-1]  # [b, q_2, q_1]
        copy_probs = tf.reshape(avg_att_weights, [-1, tf.shape(src_mask)[1]])  # [b * q_2, q_1]
        # calculate copy gate
        p_gen = tf.nn.sigmoid(linear(decoder_output, 1))  # [b * q_2, 1]

        # gate
        vocab_probs = p_gen * vocab_probs  # [b * q_2, v]
        copy_probs = (1 - p_gen) * copy_probs  # [b * q_2, q_1]

        extended_vocab_size = tf.add(tf.constant(vocab_size), max_out_oovs)  # []
        b = tf.shape(vocab_probs)[0]  # b * q_2
        extra_zeros = tf.zeros(shape=tf.stack([b, max_out_oovs], axis=0))  # [b * q_2, v']
        vocab_prob_extended = tf.concat(axis=1, values=[vocab_probs, extra_zeros])  # [b * q_2, v + v']
        batch_nums = tf.range(0, limit=tf.shape(vocab_probs)[0])  # [b * q_2]  (0, 1, 2, ...)
        batch_nums = tf.expand_dims(batch_nums, 1)  # [b * q_2, 1]
        attn_len = tf.shape(copy_probs)[1]  # q_1
        batch_nums = tf.tile(batch_nums, [1, attn_len])  # [b * q_2, q_1]
        # tile source ids oo, [b, q_1] => [b * q_2, q_1]
        tiled_source_ids_oo = tf.tile(tf.expand_dims(source_ids_oo, 1), [1, tgt_seq_len, 1])
        tiled_source_ids_oo = tf.reshape(tiled_source_ids_oo, [-1, tf.shape(tiled_source_ids_oo)[2]])
        indices = tf.stack((batch_nums, tiled_source_ids_oo), axis=2)  # [b * q_2, q_1, 2]
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
                     self.pos_anno: batch.pos_anno}
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
                     self.pos_anno: batch.pos_anno}
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
        )

        return params
