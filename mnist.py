import datetime

import tensorflow as tf
import numpy as np
import libmodular as modular
import observations

from libmodular.modular import create_m_step_summaries


def generator(arrays, batch_size):
    """Generate batches, one with respect to each array's first axis."""
    starts = [0] * len(arrays)  # pointers to where we are in iteration
    while True:
        batches = []
        for i, array in enumerate(arrays):
            start = starts[i]
            stop = start + batch_size
            diff = stop - array.shape[0]
            if diff <= 0:
                batch = array[start:stop]
                starts[i] += batch_size
            else:
                batch = np.concatenate((array[start:], array[:diff]))
                starts[i] = diff
            batches.append(batch)
        yield batches


def run():
    (x_train, y_train), (x_test, y_test) = observations.mnist('~/data/MNIST')
    dataset_size = x_train.shape[0]

    inputs = tf.placeholder(tf.float32, [None, 28 * 28], 'inputs')
    labels = tf.placeholder(tf.int32, [None], 'labels')
    data_indices = tf.placeholder(tf.int32, [None], 'data_indices')

    def network(context: modular.ModularContext):
        modules = modular.create_dense_modules(inputs, module_count=10, units=32)
        hidden = modular.modular_layer(inputs, modules, parallel_count=1, context=context)
        hidden = tf.nn.relu(hidden)

        modules = modular.create_dense_modules(hidden, module_count=8, units=10)
        logits = modular.modular_layer(hidden, modules, parallel_count=1, context=context)

        target = modular.modularize_target(labels, context)
        loglikelihood = tf.distributions.Categorical(logits).log_prob(target)

        predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, target), tf.float32))

        selection_entropy = context.selection_entropy()
        batch_selection_entropy = context.batch_selection_entropy()

        return loglikelihood, logits, accuracy, selection_entropy, batch_selection_entropy

    template = tf.make_template('network', network)
    optimizer = tf.train.AdamOptimizer()
    e_step, m_step, eval = modular.modularize(template, optimizer, dataset_size,
                                              data_indices, sample_size=10)
    ll, logits, accuracy, s_entropy, bs_entropy = eval

    tf.summary.scalar('loglikelihood', tf.reduce_mean(ll))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('entropy/exp_selection', tf.exp(s_entropy))
    tf.summary.scalar('entropy/exp_batch_selection', tf.exp(bs_entropy))

    with tf.Session() as sess:
        time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        writer = tf.summary.FileWriter(f'logs/train_{time}')
        test_writer = tf.summary.FileWriter(f'logs/test_{time}')
        general_summaries = tf.summary.merge_all()
        m_step_summaries = tf.summary.merge([create_m_step_summaries(), general_summaries])
        sess.run(tf.global_variables_initializer())

        batches = generator([x_train, y_train, np.arange(dataset_size)], 32)
        for i, (batch_x, batch_y, indices) in enumerate(batches):
            feed_dict = {
                inputs: batch_x,
                labels: batch_y,
                data_indices: indices
            }
            step = e_step if i % 10 == 0 else m_step
            summaries = m_step_summaries if step == m_step else general_summaries
            _, summary_data = sess.run([step, summaries], feed_dict)
            writer.add_summary(summary_data, global_step=i)
            if i % 100 == 0:
                test_feed_dict = {inputs: x_test, labels: y_test}
                summary_data = sess.run(general_summaries, test_feed_dict)
                test_writer.add_summary(summary_data, global_step=i)
        writer.close()
        test_writer.close()


if __name__ == '__main__':
    run()
