# import numpy as np
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# X_1 = tf.placeholder(tf.float32, name = "X_1")
# X_2 = tf.placeholder(tf.float32, name = "X_2")

# multiply = tf.multiply(X_1, X_2, name = "multiply")

# import numpy as np
# x_input = np.random.sample((1,2))
# print(x_input)

# # using a placeholder
# x = tf.placeholder(tf.float32, shape=[1,2], name = 'X')


import tensorflow as tf
tf.config.list_physical_devices('GPU')