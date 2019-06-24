from decomp_models.model_utils.attention import compute_copy_weights
from decomp_models.model_utils.layer import *


def lstm_encoder(inputs, length, params, scope=None, reuse=False):
    with tf.variable_scope(scope, default_name="encoder",
                           values=[inputs, length], reuse=reuse):
        cell_enc = []
        cell_enc_bw = []

        for _ in range(params.num_encoder_layers):
            if params.rnn_cell == "LSTM":
                cell_e = tf.nn.rnn_cell.LSTMCell(params.enc_hidden_size)
                cell_e_bw = tf.nn.rnn_cell.LSTMCell(params.enc_hidden_size)
            elif params.rnn_cell == "GRU":
                cell_e = tf.nn.rnn_cell.GRUCell(params.enc_hidden_size)
                cell_e_bw = tf.nn.rnn_cell.GRUCell(params.enc_hidden_size)
            else:
                raise ValueError("%s not supported" % params.rnn_cell)

            cell_e = tf.nn.rnn_cell.DropoutWrapper(
                cell_e,
                output_keep_prob=1.0 - params.residual_dropout,
                variational_recurrent=params.use_variational_dropout,
                input_size=params.hidden_size,
                dtype=tf.float32
            )
            cell_e_bw = tf.nn.rnn_cell.DropoutWrapper(
                cell_e_bw,
                output_keep_prob=1.0 - params.residual_dropout,
                variational_recurrent=params.use_variational_dropout,
                input_size=params.hidden_size,
                dtype=tf.float32
            )

            if params.use_residual:
                cell_e = tf.nn.rnn_cell.ResidualWrapper(cell_e)
                cell_e_bw = tf.nn.rnn_cell.ResidualWrapper(cell_e_bw)

            cell_enc.append(cell_e)
            cell_enc_bw.append(cell_e_bw)

        cell_enc = tf.nn.rnn_cell.MultiRNNCell(cell_enc)
        cell_enc_bw = tf.nn.rnn_cell.MultiRNNCell(cell_enc_bw)

        with tf.variable_scope("encoder"):
            final_output, final_state = tf.nn.bidirectional_dynamic_rnn(cell_enc, cell_enc_bw, inputs,
                                                                        length,
                                                                        dtype=tf.float32)
        final_output = tf.concat([final_output[0], final_output[1]], axis=-1)
        return final_output, final_state


def lstm_decoder(inputs, memory, bias, mem_bias, params, state=None, scope=None, reuse=False):
    with tf.variable_scope(scope, default_name="decoder",
                           values=[inputs, memory, bias, mem_bias], reuse=reuse):
        next_state = {}

        cell_dec = []

        print(inputs.shape)
        print(memory.shape)
        print(bias.shape)
        print(mem_bias.shape)

        hidden_size = params.hidden_size if not params.use_pos else params.hidden_size + params.ner_emb_size

        for _ in range(params.num_decoder_layers):
            if params.rnn_cell == "LSTM":
                cell_d = tf.nn.rnn_cell.LSTMCell(hidden_size)
            elif params.rnn_cell == "GRU":
                cell_d = tf.nn.rnn_cell.GRUCell(hidden_size)
            else:
                raise ValueError("%s not supported" % params.rnn_cell)

            cell_d = tf.nn.rnn_cell.DropoutWrapper(
                cell_d,
                output_keep_prob=1.0 - params.residual_dropout,
                variational_recurrent=params.use_variational_dropout,
                input_size=hidden_size,
                dtype=tf.float32
            )

            if params.use_residual:
                cell_d = tf.nn.rnn_cell.ResidualWrapper(cell_d)

            cell_dec.append(cell_d)

        cell_dec = tf.nn.rnn_cell.MultiRNNCell(cell_dec)

        with tf.variable_scope("decoder"):
            outputs, _ = tf.nn.dynamic_rnn(cell_dec, inputs,
                                           bias,
                                           initial_state=memory)

        att_weights = compute_copy_weights(outputs, memory, mem_bias, hidden_size, params.attention_dropout)

        if state is not None:
            return att_weights, outputs, next_state

        return att_weights, outputs
