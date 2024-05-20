#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:47:37 2024

@author: francesco
"""

import tensorflow as tf
import tensorflow_probability as tfp
import time

#from https://github.com/PacktPublishing/Hands-On-Generative-AI-with-Python-and-TensorFlow-2/blob/master/Chapter_4/models/rbm.py
class RBM(tf.keras.layers.Layer):
    def __init__(self, number_hidden_units=10, number_visible_units=None,
                 learning_rate=0.1, cd_steps=1):
        super().__init__()
        self.number_hidden_units = number_hidden_units
        self.number_visible_units = number_visible_units
        self.learning_rate = learning_rate
        self.cd_steps = cd_steps

    def build(self, input_shape):

        if not self.number_visible_units:
            self.number_visible_units = input_shape[-1]

        self.hb = self.add_weight(shape=(self.number_hidden_units, ),
                                  initializer='random_normal',
                                  trainable=True)
        self.vb = self.add_weight(shape=(self.number_visible_units, ),
                                  initializer='random_normal',
                                  trainable=True)
        self.w_rec = self.add_weight(shape=(self.number_visible_units,
                                     self.number_hidden_units),
                                     initializer='random_normal',
                                     trainable=True)
        self.w_gen = self.add_weight(shape=(self.number_hidden_units,
                                     self.number_visible_units),
                                     initializer='random_normal',
                                     trainable=True)

    def free_energy(self, x):
        return -tf.tensordot(x, self.vb, 1) - tf.reduce_sum(tf.math.log(1 + tf.math.exp(tf.add(tf.matmul(x, self.w_rec), self.hb))), 1)

    def forward(self, x):
        return tf.sigmoid(tf.add(tf.matmul(x, self.w_rec), self.hb))

    def sample_h(self, x):
        u_sample = tfp.distributions.Uniform().sample((x.shape[1], self.hb.shape[-1]))
        return tf.cast(self.forward(x) > u_sample, tf.float32)

    def reverse(self, x):
        return tf.sigmoid(tf.add(tf.matmul(x, self.w_gen), self.vb))

    def sample_v(self, x):
        u_sample = tfp.distributions.Uniform().sample((x.shape[1],
                                                       self.vb.shape[-1]))
        return tf.cast(self.reverse(x) > u_sample, tf.float32)

    def reverse_gibbs(self, x):
        return self.sample_h(self.sample_v(x))

    def forward_gibbs(self, x):
        return self.sample_v(self.sample_h(x))

    def cd_update(self, x):
        with tf.GradientTape(watch_accessed_variables=False) as g:

            h_sample = self.sample_h(x)
            for _ in range(self.cd_steps):
                v_sample = tf.constant(self.sample_v(h_sample))
                h_sample = self.sample_h(v_sample)

            g.watch(self.w_rec)
            g.watch(self.hb)
            g.watch(self.vb)
            cost = tf.reduce_mean(self.free_energy(x)) - tf.reduce_mean(self.free_energy(v_sample))

        w_grad, hb_grad, vb_grad = g.gradient(cost,
                                              [self.w_rec, self.hb, self.vb])

        self.w_rec.assign_sub(self.learning_rate * w_grad)
        self.w_gen = tf.Variable(tf.transpose(self.w_rec))  # force tieing
        self.hb.assign_sub(self.learning_rate * hb_grad)
        self.vb.assign_sub(self.learning_rate * vb_grad)

        return self.reconstruction_cost(x).numpy()

    def reconstruction_cost(self, x):
        return tf.reduce_mean(
            tf.reduce_sum(
                tf.math.add(
                            tf.math.multiply(x,
                                             tf.math.log(
                                                 self.reverse(
                                                     self.forward(x)))),
                            tf.math.multiply(tf.math.subtract(1, x),
                                             tf.math.log(
                                  tf.math.subtract(1, self.reverse(
                                                      self.forward(x)))))
                            ), 1),)
        
    def train_rbm(self, data, map_fn, num_epochs=100, tolerance=1e-3, batch_size=32, shuffle_buffer=1024):
    
        last_cost = None
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            cost = 0.0
            count = 0.0
            for datapoints in data.map(map_fn).shuffle(shuffle_buffer).batch(batch_size):
                cost += self.cd_update(datapoints)
                count += 1.0
            cost /= count
            print("epoch: {}, cost: {}".format(epoch, cost))
            if last_cost and abs(last_cost-cost) <= tolerance:
                break
            last_cost = cost
            print('-------- epoch: %.2f seconds --------' % (time.time() - epoch_start_time))
        
        #return rbm