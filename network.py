import tensorflow as tf
import numpy as np


class Network:

    def __init__(self,
                 input_height,  # 2D n_features[0]
                 input_width,  # 2D n_features[1]
                 n_actions,
                 lr=0.1,
                 reward_decay=1.0,
                 decay=0.9,
                 epsilon=1e-9):
        self.input_height = input_height
        self.input_width = input_width
        self.n_actions = n_actions
        self.lr = lr
        self.reward_decay = reward_decay
        self.decay = decay
        self.epsilon = epsilon

        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def _build_net(self):
        self.states = tf.placeholder(tf.float32, [None, self.input_height, self.input_width], name='states')
        self.actions = tf.placeholder(tf.int32, [None, ], name='actions')
        self.rewards = tf.placeholder(tf.float32, [None, ], name='rewards')

        self.l_dense1 = tf.layers.dense(
            inputs=self.states,
            units=20,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02),
            bias_initializer=tf.constant_initializer(0),
            name='dense1'
        )

        # drop = tf.nn.dropout(self.l_dense1, keep_prob=0.9)

        self.l_dense2 = tf.layers.dense(
            inputs=self.l_dense1,
            units=40,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.22),
            bias_initializer=tf.constant_initializer(0),
            name='dense2'
        )

        # drop = tf.nn.dropout(self.l_dense2, keep_prob=0.8)

        self.l_dense3 = tf.layers.dense(
            inputs=self.l_dense2,
            units=20,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.16),
            bias_initializer=tf.constant_initializer(0),
            name='dense3'
        )

        # drop = tf.nn.dropout(self.l_dense3, keep_prob=0.95)

        self.output = tf.layers.dense(
            inputs=self.l_dense3,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.22),
            bias_initializer=tf.constant_initializer(0),
            name='dense4'
        )

        self.act_prob = tf.nn.softmax(self.output)[0][0]

        self.neg_log_prob = tf.reduce_sum(-tf.log(self.act_prob) * tf.one_hot(self.actions, self.n_actions), axis=1)
        self.loss = tf.reduce_mean(self.neg_log_prob * self.rewards)

        self.optimizer = tf.train.RMSPropOptimizer(self.lr, self.decay, epsilon=self.epsilon)
        # self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = self.optimizer.minimize(self.loss)

    def choose_action(self, state):
        act_prob = self.sess.run(self.act_prob, feed_dict={self.states: state[None, :, :]})
        action = np.random.choice(range(act_prob.shape[-1]), p=act_prob)
        return action

    def learn(self, states, actions, rewards):
        self.sess.run(self.train_op, feed_dict={
            self.states: states,
            self.actions: actions,
            self.rewards: self._discount_and_norm_rewards(rewards, self.reward_decay)
        })

    def _discount_and_norm_rewards(self, rewards, gamma):
        # discount episode rewards
        discounted_experience_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted_experience_rewards[t] = running_add

        # normalize episode rewards
        discounted_experience_rewards -= np.mean(discounted_experience_rewards)
        discounted_experience_rewards /= np.std(discounted_experience_rewards)
        return discounted_experience_rewards
