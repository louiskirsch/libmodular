from collections import namedtuple
from enum import Enum
from typing import List
import libmodular.tensor_utils as tensor_utils

import tensorflow as tf

ModularMode = Enum('ModularMode', 'E_STEP M_STEP EVALUATION')
ModularLayerAttributes = namedtuple('ModularLayerAttributes', ['selection', 'best_selection', 'controller'])
ModulePool = namedtuple('ModulePool', ['module_count', 'module_fnc', 'output_shape'])


class ModularContext:

    def __init__(self, mode: ModularMode, data_indices=None, dataset_size: int = None, sample_size: int = 1):
        self.mode = mode
        self.data_indices = data_indices
        self.dataset_size = dataset_size
        self.sample_size = sample_size
        self.e_step_samples = False
        self.layers: List[ModularLayerAttributes] = []

    def begin_modular(self, inputs):
        if self.mode == ModularMode.E_STEP and not self.e_step_samples:
            self.e_step_samples = True
            rank = inputs.shape.ndims
            return tf.tile(inputs, [self.sample_size] + [1] * (rank - 1))
        return inputs

    def selection_entropy(self):
        return tf.reduce_mean([tf.reduce_mean(layer.controller.entropy()) for layer in self.layers])

    def batch_selection_entropy(self):
        def layer_entropy(layer):
            probs = tf.reduce_mean(layer.controller.probs, axis=0)
            return -tf.reduce_sum(probs * tf.log(probs + 1e-30), axis=-1)
        return tf.reduce_mean([layer_entropy(layer) for layer in self.layers])

    def selection_logprob(self):
        x = [tf.reduce_sum(attrs.controller.log_prob(attrs.selection), axis=-1) for attrs in self.layers]
        return tf.reduce_sum(x, axis=0)

    def update_best_selection(self, best_selection_indices):
        def update(layer):
            selection = tf.reshape(layer.selection, [self.sample_size, -1] + layer.selection.shape[1:].as_list())
            new_best_selection = tensor_utils.gather_each(selection, best_selection_indices)
            return tf.scatter_update(layer.best_selection, self.data_indices, new_best_selection)
        return tf.group(*(update(layer) for layer in self.layers))


def run_modules(inputs, selection, module_fnc, output_shape):
    batch_size = tf.shape(inputs)[0]
    output_shape = [batch_size] + output_shape
    used_modules, _ = tf.unique(tf.reshape(selection, (-1,)))

    def compute_module(accum, module):
        mask = tf.equal(module, selection)
        reduced_mask = tf.reduce_any(mask, axis=-1)
        indices = tf.where(reduced_mask)
        affected_inp = tf.boolean_mask(inputs, reduced_mask)
        output = module_fnc(affected_inp, module)
        return accum + tf.scatter_nd(indices, output, tf.cast(output_shape, tf.int64))

    output = tf.scan(compute_module, used_modules, initializer=tf.zeros(output_shape))[-1]
    return output


def e_step(template, sample_size, dataset_size, data_indices):
    context = ModularContext(ModularMode.E_STEP, data_indices, dataset_size, sample_size)

    loglikelihood = template(context)[0]
    selection_logprob = context.selection_logprob()

    shape = [sample_size, -1] + loglikelihood.shape[1:].as_list()
    logprob = tf.reshape(loglikelihood + selection_logprob, shape)
    best_selection_indices = tf.stop_gradient(tf.argmax(logprob, axis=0))

    return context.update_best_selection(best_selection_indices)


def m_step(template, optimizer, dataset_size, data_indices):
    context = ModularContext(ModularMode.M_STEP, data_indices, dataset_size)
    loglikelihood = template(context)[0]
    selection_logprob = context.selection_logprob()

    ctrl_objective = -tf.reduce_mean(selection_logprob)
    module_objective = -tf.reduce_mean(loglikelihood)

    return optimizer.minimize(ctrl_objective + module_objective)


def evaluation(template):
    context = ModularContext(ModularMode.EVALUATION)
    return template(context)


