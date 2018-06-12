import datetime

import tensorflow as tf
import numpy as np
import libmodular as modular
import observations

from libmodular.modular import create_m_step_summaries, M_STEP_SUMMARIES


def make_handle(sess, dataset):
    iterator = dataset.make_initializable_iterator()
    handle, _ = sess.run([iterator.string_handle(), iterator.initializer])
    return handle


# noinspection PyProtectedMember
def run():
    # Load dataset
    (x_train, y_train), (x_test, y_test) = observations.cifar10('~/data/cifar10')
    y_test = y_test.astype(np.uint8)  # Fix test_data dtype
    dataset_size = x_train.shape[0]

    # Train dataset
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))._enumerate().repeat().shuffle(50000).batch(128)
    # Test dataset
    dummy_data_indices = tf.zeros([x_test.shape[0]], dtype=tf.int64)
    test = tf.data.Dataset.from_tensors((dummy_data_indices, (x_test, y_test))).repeat()

    # Handle to switch between datasets
    handle = tf.placeholder(tf.string, [])
    itr = tf.data.Iterator.from_string_handle(handle, train.output_types, train.output_shapes)
    data_indices, (inputs, labels) = itr.get_next()

    # Preprocessing
    inputs = tf.cast(inputs, tf.float32) / 255.0
    inputs = tf.transpose(inputs, perm=(0, 2, 3, 1))
    labels = tf.cast(labels, tf.int32)

    def network(context: modular.ModularContext):
        # 4 modular CNN layers
        activation = inputs
        for _ in range(4):
            input_channels = activation.shape[-1]
            filter_shape = [3, 3, input_channels, 8]
            modules = modular.create_conv_modules(filter_shape, module_count=5, strides=[1, 1, 1, 1])
            hidden = modular.modular_layer(activation, modules, parallel_count=1, context=context)
            pooled = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            activation = tf.nn.relu(pooled)

        flattened = tf.layers.flatten(activation)
        logits = tf.layers.dense(flattened, units=10)

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
        train_dict = {handle: make_handle(sess, train)}
        test_dict = {handle: make_handle(sess, test)}

        for i in range(10000):
            # Switch between E-step and M-step
            step = e_step if i % 10 == 0 else m_step

            # Sometimes generate summaries
            if i % 99 == 0:
                summaries = m_step_summaries if step == m_step else general_summaries
                _, summary_data = sess.run([step, summaries], train_dict)
                writer.add_summary(summary_data, global_step=i)
                summary_data = sess.run(general_summaries, test_dict)
                test_writer.add_summary(summary_data, global_step=i)
            else:
                sess.run(step, train_dict)

        writer.close()
        test_writer.close()


if __name__ == '__main__':
    run()
