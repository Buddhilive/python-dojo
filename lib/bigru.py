import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Concatenate, Dense, LayerNormalization, Dropout
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

class MultiHeadAttentionLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = Dense(units=d_model)
        self.key_dense = Dense(units=d_model)
        self.value_dense = Dense(units=d_model)

        self.dense = Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = tf.matmul(query, key, transpose_b=True)
        scaled_attention /= tf.math.sqrt(tf.cast(self.depth, tf.float32))
        if mask is not None:
            scaled_attention += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(attention_weights, value)

        # combine heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output

class BiGRUWithAttention(keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_heads, num_layers, max_seq_len, rate=0.1):
        super().__init__()

        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_seq_len)
        self.dropout = Dropout(rate)

        self.bidirectional_grus = []
        for i in range(num_layers):
            self.bidirectional_grus.append(Bidirectional(GRU(hidden_size, return_sequences=True)))

        self.attention_layers = []
        for i in range(num_layers):
            self.attention_layers.append(MultiHeadAttentionLayer(d_model=hidden_size, num_heads=num_heads))

        self.dense = Dense(units=vocab_size, activation="softmax")

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout(x)
        for i in range(len(self.bidirectional_grus)):
            x = self.bidirectional_grus[i](x)
            attention_output = self.attention_layers[i]({
                'query': x,
                'key': x,
                'value': x
            })
           
class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, n_heads, head_dim, mask_right=False):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.mask_right = mask_right
        self.wq = keras.layers.Dense(n_heads * head_dim, use_bias=False)
        self.wk = keras.layers.Dense(n_heads * head_dim, use_bias=False)
        self.wv = keras.layers.Dense(n_heads * head_dim, use_bias=False)
        self.dense = keras.layers.Dense(head_dim * n_heads, use_bias=False)

    def call(self, inputs, training=None):
        q, k, v, mask = inputs['q'], inputs['k'], inputs['v'], inputs['mask']

        # Linear projections
        qw = self.wq(q)  # [batch_size, seq_len, n_heads * head_dim]
        kw = self.wk(k)  # [batch_size, seq_len, n_heads * head_dim]
        vw = self.wv(v)  # [batch_size, seq_len, n_heads * head_dim]

        # Reshape to [batch_size, seq_len, n_heads, head_dim]
        qw = tf.reshape(qw, [tf.shape(qw)[0], -1, self.n_heads, self.head_dim])
        kw = tf.reshape(kw, [tf.shape(kw)[0], -1, self.n_heads, self.head_dim])
        vw = tf.reshape(vw, [tf.shape(vw)[0], -1, self.n_heads, self.head_dim])

        # Transpose to [batch_size, n_heads, seq_len, head_dim]
        qw = tf.transpose(qw, [0, 2, 1, 3])
        kw = tf.transpose(kw, [0, 2, 1, 3])
        vw = tf.transpose(vw, [0, 2, 1, 3])

        # Compute scaled dot-product attention
        att = tf.matmul(qw, kw, transpose_b=True)
        att = att / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

        if mask is not None:
            mask = tf.expand_dims(mask, axis=1)  # [batch_size, 1, seq_len_q, seq_len_k]
            if self.mask_right:
                mask = tf.linalg.band_part(mask, num_lower=-1, num_upper=0)
            att = att * mask + (1 - mask) * tf.float32.min

        att = tf.nn.softmax(att, axis=-1)
        att = self.dropout1(att, training=training)
        output = tf.matmul(att, vw)

        # Transpose to [batch_size, seq_len, n_heads, head_dim]
        output = tf.transpose(output, [0, 2, 1, 3])

        # Reshape to [batch_size, seq_len, n_heads * head_dim]
        concat_output = tf.reshape(output, [tf.shape(output)[0], -1, self.n_heads * self.head_dim])
        output = self.dense(concat_output)
        output = self.dropout2(output, training=training)
        return output

class BiGRUWithAttention(keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, n_heads, head_dim, max_sequence_len):
        super(BiGRUWithAttention, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = keras.layers.Embedding(self.vocab_size, embedding_dim, mask_zero=True)
        self.bi_gr
