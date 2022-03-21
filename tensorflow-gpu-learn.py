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

# import tensorflow lib and choose GPU
import tensorflow as tf
tf.config.list_physical_devices('GPU')

# example of batch gradient descent
from sklearn.datasets import make_circles
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot


# generate dataset
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=1000, batch_size=len(trainX), verbose=0)
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss learning curves
pyplot.subplot(211)
pyplot.title('Cross-Entropy Loss', pad=-40)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy learning curves
pyplot.subplot(212)
pyplot.title('Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')

import sys
# save plot to file
filename = sys.argv[0].split('/')[-1]
pyplot.savefig(filename + '_plot.png')
pyplot.close()


# import matplotlib
# matplotlib.use('Agg')

# import matplotlib.pyplot as plt
# import numpy as np

# x = np.random.randn(60)
# y = np.random.randn(60)

# plt.scatter(x, y, s=20)

# out_png = 'meow.png'
# plt.savefig(out_png, dpi=150)