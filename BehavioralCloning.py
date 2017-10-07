import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import scipy.misc
import math
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import errno
import json
import os


def path(source_path):
    filename = source_path.split('\\')[-1]
    current_path = './data/IMG/' + filename
    return current_path


def random_flip(image, steering_angle, flipping_prob=0.5):
    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle


def crop_img(image):
    top = 70
    bottom = image.shape[0] - 20
    return image[top:bottom, :]


def blur(img):
    gb = cv2.GaussianBlur(img, (5,5), 20.0)
    return cv2.addWeighted(img, 2, gb, -1, 0)

def resize(image, new_size):
    return scipy.misc.imresize(image, new_size)


def process_img(image, steering_angle):
    image = crop_img(image)
    image = blur(image)
    image, steering_angle = random_flip(image, steering_angle, flipping_prob=0.5)
    image = resize(image, new_size=(64, 128))
    return image, steering_angle


def select_img(line):
    indicator_img = np.random.randint(0, 3)
    correction = 0.2
    if indicator_img == 0:  # center
        img = cv2.imread(path(line[0]))
        angle = float(line[3])
    elif indicator_img == 1:  # left
        img = cv2.imread(path(line[1]))
        angle = float(line[3]) + correction
    else:  # right
        img = cv2.imread(path(line[2]))
        angle = float(line[3]) - correction
    return img, angle


def visualize(lines):
    # three camera view
    plt.figure(figsize=(12,4))
    imtitle1 = ['left', 'center', 'right']
    p = [path(lines[0][1]), path(lines[0][0]), path(lines[0][2])]
    for i in range(3):
        img = mpimg.imread(p[i])
        plt.subplot(1, 3, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(imtitle1[i])
    plt.show()

    # image process
    img = mpimg.imread(p[1])
    crop = crop_img(img)
    bl = blur(crop)
    flip = np.fliplr(bl)
    resz = resize(flip, (64,128))
    imgs = [img, crop, bl, flip, resz]

    plt.figure(figsize=(16, 6))
    imtitle2 = ['Raw', 'Crop', 'Sharpen', 'Flip', 'Resize']
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.title(imtitle2[i])
    plt.show()


def load_data(lines, batch_size):
    while True:
        x_batch = []
        y_batch = []
        idx = 0
        for line in lines:
            image, angle = select_img(line)
            image_process, angle_process = process_img(image, angle)
            x_batch.append(image_process)
            y_batch.append(angle_process)
            idx += 1
            if idx == batch_size:
                idx = 0
                yield np.array(x_batch), np.array(y_batch)
                x_batch = []
                y_batch = []


def silent_delete(file):
    """
    This method delete the given file from the file system if it is available
    Source: http://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist
    :param file:
        File to be deleted
    :return:
        None
    """
    try:
        os.remove(file)

    except OSError as error:
        if error.errno != errno.ENOENT:
            raise

def save_model(model, model_name='model.json', weights_name='model.h5'):
    """
    Save the model into the hard disk
    :param model:
        Keras model to be saved
    :param model_name:
        The name of the model file
    :param weights_name:
        The name of the weight file
    :return:
        None
    """
    silent_delete(model_name)
    silent_delete(weights_name)

    json_string = model.to_json()
    with open(model_name, 'w') as outfile:
        json.dump(json_string, outfile)

    model.save_weights(weights_name)



lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        lines.append(row)
num_data = len(lines)
lines = shuffle(lines)
print('num_data = {}'.format(num_data))


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64, 128, 3)))
# model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

# Next, five fully connected layers
model.add(Dense(1164))
model.add(Activation(activation="relu"))

model.add(Dense(100))
model.add(Activation(activation="relu"))

model.add(Dense(50))
model.add(Activation(activation="relu"))

model.add(Dense(10))
model.add(Activation(activation="relu"))

model.add(Dense(1))

model.summary()

batch_size = 128
learning_rate = 1e-4

model.compile(optimizer=Adam(learning_rate), loss="mse")

training_num = math.ceil(num_data * 0.8/batch_size)*batch_size
validation_num = math.ceil(num_data * 0.2/batch_size)*batch_size
result = model.fit_generator(generator=load_data(lines[:training_num], batch_size),
                             samples_per_epoch=training_num,
                             nb_epoch=2,
                             validation_data=load_data(lines[-validation_num:], batch_size),
                             nb_val_samples=validation_num,
                             verbose=1)

#
# images = []
# measurements = []
# for line in lines:
#     source_path = line[0]
#     filename = source_path.split('\\')[-1]
#     current_path = './data/IMG/' + filename
#     image = cv2.imread(current_path)
#     images.append(resize(image, new_size=(64,128)))
#     measurement = float(line[3])
#     measurements.append(measurement)
#
# X_train = np.array(images)
# y_train = np.array(measurements)
#
# model = Sequential()
# model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(64,128,3)))
# # model.add(Cropping2D(cropping=((70,25),(0,0))))
# model.add(Convolution2D(6,5,5,activation="relu"))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5,5,activation="relu"))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')
# model.summary()
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True,
#           nb_epoch=2)

# ## print the keys contained in the history object
# print(result.history.keys())
# model.save('model.h5', overwrite=True)
save_model(model)
print("Model Saved.")

# ### plot the training and validation loss for each epoch
# plt.figure()
# plt.plot(result.epoch, result.history['loss'])
# plt.plot(result.epoch, result.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.ylim([0, 0.1])
# plt.show()

