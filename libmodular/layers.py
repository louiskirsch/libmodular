import tensorflow as tf
from tensorflow.contrib import distributions as tfd

from libmodular.modular import ModulePool, ModularContext, ModularMode, ModularLayerAttributes
from libmodular.modular import run_modules, e_step, m_step, evaluation


def create_dense_modules(inputs_or_shape, module_count: int, units: int = None, activation=None):
    with tf.variable_scope(None, 'dense_modules'):
        if hasattr(inputs_or_shape, 'shape') and units is not None:
            weights_shape = [module_count, inputs_or_shape.shape[-1].value, units]
        else:
            weights_shape = [module_count] + inputs_or_shape
            units = inputs_or_shape[-1]
        weights = tf.get_variable('weights', weights_shape)
        biases_shape = [module_count, units]
        biases = tf.get_variable('biases', biases_shape, initializer=tf.zeros_initializer())

        def module_fnc(x, a):
            out = tf.matmul(x, weights[a]) + biases[a]
            if activation is not None:
                out = activation(out)
            return out

        return ModulePool(module_count, module_fnc, output_shape=[units])


def modular_layer(inputs, modules: ModulePool, parallel_count: int, context: ModularContext):
    with tf.variable_scope(None, 'modular_layer'):
        inputs = context.begin_modular(inputs)

        logits = tf.layers.dense(inputs, modules.module_count * parallel_count)
        logits = tf.reshape(logits, [-1, parallel_count, modules.module_count])
        ctrl = tfd.Categorical(logits)

        initializer = tf.random_uniform_initializer(maxval=modules.module_count, dtype=tf.int32)
        shape = [context.dataset_size, parallel_count]
        best_selection_persistent = tf.get_variable('best_selection', shape, tf.int32, initializer)

        if context.mode == ModularMode.E_STEP:
            best_selection = tf.gather(best_selection_persistent, context.data_indices)[tf.newaxis]
            sampled_selection = tf.reshape(ctrl.sample(), [context.sample_size, -1, parallel_count])
            selection = tf.concat([best_selection, sampled_selection[1:]], axis=0)
            selection = tf.reshape(selection, [-1, parallel_count])
        elif context.mode == ModularMode.M_STEP:
            selection = tf.gather(best_selection_persistent, context.data_indices)
        elif context.mode == ModularMode.EVALUATION:
            selection = ctrl.mode()
        else:
            raise ValueError('Invalid modular mode')

        attrs = ModularLayerAttributes(selection, best_selection_persistent, ctrl)
        context.layers.append(attrs)

        return run_modules(inputs, selection, modules.module_fnc, modules.output_shape)


def modularize_target(target, context: ModularContext):
    if context.mode == ModularMode.E_STEP:
        rank = target.shape.ndims
        return tf.tile(target, [context.sample_size] + [1] * (rank - 1))
    return target


def modularize(template, optimizer, dataset_size, data_indices, sample_size):
    e = e_step(template, sample_size, dataset_size, data_indices)
    m = m_step(template, optimizer, dataset_size, data_indices)
    ev = evaluation(template)
    return e, m, ev
