#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:39:38 2024

@author: francesco
"""

import tensorflow as tf

def flatten_image(x):
    return (tf.divide(tf.dtypes.cast(tf.reshape(x["image"], (1,28*28)), tf.float32), 256.0))
def flatten_image_label(x):
    return (tf.divide(tf.dtypes.cast(tf.reshape(x["image"], (1,28*28)), tf.float32), 256.0) , x["label"])