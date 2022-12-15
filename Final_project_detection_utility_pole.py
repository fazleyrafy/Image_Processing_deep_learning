import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import imghdr    ## determines the type of image

dataset_dir = '/home/jarvis/programming/neural_net_final/5G-Utility-Pole-Planner/Data'
os.listdir(dataset_dir)
# os.listdir(os.path.join(dataset_dir, 'poles'))

img = cv2.imread(os.path.join(dataset_dir, 'poles', 'pole.329.jpg'))
print(img.shape)
# plt.imshow(img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

for image_class in os.listdir(dataset_dir):
    if image_class == 'poles' or image_class=='nonpoles':
        count = 0
        for image in os.listdir(os.path.join(dataset_dir, image_class)):
            image_path = os.path.join(dataset_dir, image_class, image)
            try:
                img = cv.imread(image_path)
            except Exception as e: 
#                 print('Issue with image {}'.format(image_path))
                count = count+1
        print(f"number of errors in {image_class} is : {count}")

data = tf.keras.utils.image_dataset_from_directory(dataset_dir)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
batch[1]

fig, ax = plt.subplots(ncols=4, figsize = (20,20))
for idx, img in enumerate(batch[0][12:16]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)
print(train_size + val_size + test_size)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

## importing necessary modules
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Concatenate, Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
import random
import os
import cv2

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline

## define identity block

def identity_block(input_tensor, kernel_size, filters, stage, block):
    
    nb_filter1, nb_filter2, nb_filter3 = filters
#     if K.image_dim_ordering() == 'tf':
    if K.image_data_format() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1)(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1)(x)
    x = BatchNormalization(axis=bn_axis)(x)

    x = Concatenate([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x

temp = 'res' + str(3) + 'a' + '_branch'
# Convolution2D??
img_input = Input(shape=(256,256,3))

x = ZeroPadding2D((3, 3))(img_input)    # 262,262,3
x = Convolution2D(64, 7, 7, name='conv1')(x)
x = BatchNormalization(axis=3, name='bn_conv1')(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)
# x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
nb_filter1, nb_filter2, nb_filter3 = [64,64,256]
if K.image_data_format() == 'tf':
    bn_axis = 3
else:
    bn_axis = 1
conv_name_base = 'res' + str(2) + 'a' + '_branch'
bn_name_base = 'bn' + str(2) + 'a' + '_branch'
x = Convolution2D(nb_filter1, 1, 1, (1,1))(x)
x.shape

## define convolution block

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    nb_filter1, nb_filter2, nb_filter3 = filters
#     if K.image_dim_ordering() == 'tf':
    if K.image_data_format() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, strides)(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1)(x)
    x = BatchNormalization(axis=bn_axis)(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, strides)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis)(shortcut)

    x = Concatenate([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x

img_input = Input(shape=(256,256,3))

x = ZeroPadding2D((3, 3))(img_input)
x = Convolution2D(64, 7, 7, name='conv1')(x)
x = BatchNormalization(axis=3, name='bn_conv1')(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

base_model = Model(img_input, x)

## importing the required weight of resnet-50

TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/\
v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        TF_WEIGHTS_PATH_NO_TOP,
                        cache_subdir='models',
                        md5_hash='a268eb855778b3df3c7506639542a6af')
base_model.load_weights(weights_path)

##applying cnn model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# removing unexpected format images

from pathlib import Path
import imghdr

data_dir = "/home/jarvis/programming/neural_net_final/5G-Utility-Pole-Planner/Data/nonpoles/"
image_extensions = [".png", ".jpg"]  # add there all your images file extensions

img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
for filepath in Path(data_dir).rglob("*"):
    if filepath.suffix.lower() in image_extensions:
        img_type = imghdr.what(filepath)
        if img_type is None:
            print(f"{filepath} is not an image")
        elif img_type not in img_type_accepted_by_tf:
            print(f"{filepath} is a {img_type}, not accepted by TensorFlow")

## Defining the model

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.summary()

hist = model.fit(train, epochs=20, validation_data=val)

hist.history

## plotting the performance

## plot performance
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper right")
plt.savefig('loss_fucntion.jpg', dpi=300)
plt.show()

## metrics
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

from sklearn import metrics
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f"precision: {pre.result().numpy()}, recall: {re.result().numpy()}, accuracy: {acc.result().numpy()}")



