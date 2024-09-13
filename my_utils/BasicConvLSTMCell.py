import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import logging
logging.disable(logging.WARNING)
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
import tf_slim as slim


class ConvRNNCell(object):
    def __call__(self, inputs, state, scope=None):
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        shape = self.shape
        num_features = self.num_features
        zeros = tf.compat.v1.zeros([batch_size, shape[0], shape[1], num_features * 2])
        return zeros


class BasicConvLSTMCell(ConvRNNCell):
    def __init__(self, shape, filter_size, num_features, forget_bias=1.0, input_size=None, state_is_tuple=False, activation=tf.compat.v1.nn.tanh):
        if input_size is not None:
            logging.warning("%s: The input_size parameter is deprecated.", self)
        self.shape = shape
        self.filter_size = filter_size
        self.num_features = num_features
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation

    @property
    def state_size(self):
        return (tf.compat.v1.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope='convLSTM'):
        """Long short-term memory cell (LSTM)."""
        with tf.compat.v1.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # print("Input Shape: \b")
            # print(inputs.shape)
            # print()
            # print("State Shape: \b")
            # print(state.shape)
            # print()
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.compat.v1.split(state, 2, 3)
            concat = _conv_linear([inputs, h], self.filter_size, self.num_features * 4, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.compat.v1.split(concat, 4, 3)

            new_c = (c * tf.compat.v1.nn.sigmoid(f + self._forget_bias) + tf.compat.v1.nn.sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c) * tf.compat.v1.nn.sigmoid(o)

            if self._state_is_tuple:
                new_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.compat.v1.concat([new_c, new_h], 3)
            return new_h, new_state


def _conv_linear(args, filter_size, num_features, bias, bias_start=0.0, scope=None):
    dtype = [a.dtype for a in args][0]
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME', activation_fn=None, scope=scope,
                        weights_initializer=tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=1.0e-3),
                        biases_initializer=bias and tf.compat.v1.constant_initializer(bias_start, dtype=dtype)):
        if len(args) == 1:
            res = slim.conv2d(args[0], num_features, [filter_size[0], filter_size[1]], scope='LSTM_conv')
        else:
            res = slim.conv2d(tf.compat.v1.concat(args, 3), num_features, [filter_size[0], filter_size[1]], scope='LSTM_conv')
        return res