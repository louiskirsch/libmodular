import tensorflow as tf


def gather_each(params, indices):
    count = tf.cast(tf.shape(indices)[0], indices.dtype)
    each = tf.range(count, dtype=indices.dtype)
    indices = tf.stack([indices, each], axis=1)
    return tf.gather_nd(params, indices)
