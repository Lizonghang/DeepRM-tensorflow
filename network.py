import tensorflow as tf
import numpy as np
tf.set_random_seed(1)
np.random.seed(1)


class Network:

    def __init__(self,
                 input_height,  # 2D n_features[0]
                 input_width,  # 2D n_features[1]
                 n_actions,
                 lr=0.001,
                 reward_decay=0.9,
                 epsilon=1e-9):
        self.input_height = input_height
        self.input_width = input_width
        self.n_actions = n_actions
        self.lr = lr
        self.reward_decay = reward_decay
        self.epsilon = epsilon

        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='params')

    def _build_net(self):
        self.states = tf.placeholder(tf.float32, [None, 1, self.input_height, self.input_width])
        self.actions = tf.placeholder(tf.int32, [None, ])
        self.rewards = tf.placeholder(tf.float32, [None, ])

        self.hidden = tf.layers.dense(
            inputs=self.states,
            units=20,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
            bias_initializer=tf.constant_initializer(0)
        )

        self.output = tf.layers.dense(
            inputs=self.hidden,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
            bias_initializer=tf.constant_initializer(0)
        )

        self.act_prob = tf.nn.softmax(self.output)[0][0][0]

        self.neg_log_prob = tf.reduce_sum(-tf.log(self.act_prob) * tf.one_hot(self.actions, self.n_actions), axis=1)
        self.loss = tf.reduce_mean(self.neg_log_prob * self.rewards)

        self.optimizer = tf.train.RMSPropOptimizer(self.lr, self.reward_decay, epsilon=self.epsilon)

    def choose_action(self, state):
        act_prob = self.sess.run(self.act_prob, feed_dict={self.states: state[None, None, :, :]})
        action = np.random.choice(range(act_prob.shape[-1]), p=act_prob)
        return action, act_prob

    def get_params(self):
        return self.net_params

    def get_gradient(self, states, actions, rewards):
        grads_all = self.optimizer.compute_gradients(self.loss)
        grad_vars = [v for (g, v) in grads_all if g is not None]
        grads_and_vars = self.optimizer.compute_gradients(self.loss, grad_vars)
        return grads_and_vars

    def apply_gradient(self, grads_and_vars):
        self.optimizer.apply_gradients(grads_and_vars)

    def set_params(self, new_net_params):
        self.sess.run([tf.assign(t, e) for t, e in zip(self.net_params, new_net_params)])
