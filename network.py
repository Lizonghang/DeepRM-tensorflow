import tensorflow as tf
import numpy as np


class Network:

    def __init__(self,
                 input_height,  # 2D n_features[0]
                 input_width,  # 2D n_features[1]
                 n_actions,
                 lr=0.01,
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

        self.net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def _build_net(self):
        self.states = tf.placeholder(tf.float32, [None, self.input_height, self.input_width], name='states')
        self.actions = tf.placeholder(tf.int32, [None, ], name='actions')
        self.rewards = tf.placeholder(tf.float32, [None, ], name='rewards')

        state_image = tf.reshape(self.states, [-1, self.input_height, self.input_width, 1])

        with tf.name_scope('conv1') as scope:
            kernel = self._variable_with_weight_decay('kernel1', shape=[3, 3, 1, 32], stddev=0.05, wd=0.0)
            conv = tf.nn.conv2d(state_image, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases1', [32], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        with tf.name_scope('conv2') as scope:
            kernel = self._variable_with_weight_decay('kernel2', shape=[3, 3, 32, 64], stddev=0.05, wd=0.0)
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases2', [64], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope)

        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        with tf.name_scope('local1') as scope:
            local1 = tf.layers.dense(
                inputs=pool2,
                units=32,
                activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(0.05),
                bias_initializer=tf.constant_initializer(0.1),
                name=scope
            )

        with tf.name_scope('drop') as scope:
            drop = tf.layers.dropout(
                inputs=local1,
                rate=0.3,
                name=scope
            )

        with tf.name_scope('local2') as scope:
            local2 = tf.layers.dense(
                inputs=drop,
                units=self.n_actions,
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(0.05),
                bias_initializer=tf.constant_initializer(0.1),
                name=scope
            )

        self.act_prob = tf.nn.softmax(local2)[0][0][0]

        self.neg_log_prob = tf.reduce_sum(-tf.log(self.act_prob) * tf.one_hot(self.actions, self.n_actions), axis=1)
        self.loss = tf.reduce_mean(self.neg_log_prob * self.rewards)

        # self.optimizer = tf.train.RMSPropOptimizer(self.lr, self.reward_decay, epsilon=self.epsilon)
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = self.optimizer.minimize(self.loss)

    def choose_action(self, state):
        act_prob = self.sess.run(self.act_prob, feed_dict={self.states: state[None, :, :]})
        action = np.random.choice(len(act_prob), p=act_prob)
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

    def _variable_on_cpu(self, name, shape, initializer):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        var = self._variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
        if wd:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var
