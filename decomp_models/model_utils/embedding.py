import tensorflow as tf


def get_embedding_may_pretrain(vocab_size, embedding_dim, params, name, trainable=False):
    if params.use_pretrained_embedding:
        embeddings = tf.get_variable(name,
                                     shape=(vocab_size, embedding_dim),
                                     initializer=tf.initializers.constant(params.init_word_matrix ),
                                     trainable=trainable)
    else:
        initializer = tf.random_normal_initializer(0.0, embedding_dim ** -0.5)
        embeddings = tf.get_variable(name,
                                     shape=(vocab_size, embedding_dim),
                                     initializer=initializer,
                                     trainable=True)
    return embeddings


def get_embedding(vocab_size, emb_dim, name):
    initializer = tf.random_normal_initializer(0.0, emb_dim ** -0.5)
    embeddings = tf.get_variable(name,
                                 shape=(vocab_size, emb_dim),
                                 initializer=initializer,
                                 trainable=True)
    return embeddings
