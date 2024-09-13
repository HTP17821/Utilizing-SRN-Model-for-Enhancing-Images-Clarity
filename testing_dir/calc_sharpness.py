import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import math

tf.random.set_seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)


def lap_img(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])

    laplacianImage = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return laplacianImage


input_path = './results'
gt_path = './sharp'
images = os.listdir(input_path)
x_train = []
y_train = []
for image in images:
    x_train.append(cv2.imread(os.path.join(input_path, image), flags=cv2.IMREAD_GRAYSCALE).astype(np.float32))
    y_train.append(cv2.imread(os.path.join(gt_path, image), flags=cv2.IMREAD_GRAYSCALE).astype(np.float32))

for i in range(len(y_train)):
    y = y_train[i]
    x = x_train[i]
    y_train[i] = (y - x) / 255.0
for i in range(len(x_train)):
    image = x_train[i]
    x_train[i] = lap_img(image) / 255.0
x_train = np.array(x_train)
y_train = np.array(y_train)

checkpoint_path = "./checkpoints"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
batch_size = 1
epoch = 300
n_batches = len(x_train) / batch_size
n_batches = math.ceil(n_batches)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False, verbose=0, save_freq=5*n_batches)

model = Sequential()
model.add(Dense(units=1, input_shape=(720, 1280, 1)))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), loss='mse')
print(model.summary())
# model.save_weights(checkpoint_path.format(epoch=0))
history = model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, callbacks=[cp_callback], validation_split=0.1)
model.save_weights(checkpoint_path)
weight = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]
print('Weight = ' + str(weight))
print('Bias = ' + str(bias))
