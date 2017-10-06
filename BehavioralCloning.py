import csv
import cv2
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import scipy.misc


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


def resize(image, new_size):
    return scipy.misc.imresize(image, new_size)


def process_img(image, steering_angle):
    image = crop_img(image)
    image, steering_angle = random_flip(image, steering_angle, flipping_prob=0.5)
    image = resize(image, new_size=(64, 128))
    return image, steering_angle


def select_img(line):
    indicator_img = np.random.randint(0, 3)
    correction = 0.2
    if indicator_img == 0: # center
        img = cv2.imread(path(line[0]))
        angle = float(line[3])
    elif indicator_img == 1: # left
        img = cv2.imread(path(line[1]))
        angle = float(line[3]) + correction
    else: # right
        img = cv2.imread(path(line[2]))
        angle = float(line[3]) - correction
    return img, angle


def load_data(lines, batch_size=64):
    while True:
        X_batch = []
        y_batch = []
        batch = 0
        if (batch + batch_size) < len(lines):
            for line in lines[batch:batch+batch_size]:
                image, angle = select_img(line)
                image_process, angle_process = process_img(image, angle)
                X_batch.append(image_process)
                y_batch.append(angle_process)

            batch += batch_size
        yield np.array(X_batch), np.array(y_batch)

data_saved = False
if data_saved:
    print('Loading data...')
    with open('./data_pickle.p', mode="rb") as f:
        data_pickle = pickle.load(f)
    num_data = len(data_pickle["y_data"])
    train_gen = np.array(data_pickle["x_data"][: int(0.8*num_data)],
                         data_pickle["y_data"][: int(0.8*num_data)])
    valid_gen = np.array(data_pickle["x_data"][int(0.8*num_data):],
                         data_pickle["y_data"][int(0.8*num_data):])
else:
    lines = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            lines.append(row)
    num_data = len(lines)
    print('num_data = {}'.format(num_data))
    batch_size = 64
    generator = load_data(lines, batch_size)
    data = []
    #
    for i in range(int(num_data / batch_size)):
        data.append(next(generator))

    new = []
    for idx in range(64):
        new[0][idx] = data[0][0][idx]
        new[1][idx] = data[0][1][idx]




    # train_gen = dataset[:, :int(0.8*len(dataset[1]))]
    # valid_gen = dataset[:, int(0.8*len(dataset[1])):]

    # save data into pickle file
    # data_pickle = {}
    # data_pickle['x_data'] = X_batch
    # data_pickle['y_data'] = y_batch
    # pickle.dump(data_pickle, open("./data_pickle.p", "wb"))
    # print('Data saved')

#
# model = Sequential()
# model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
#
# model.add(Convolution2D(6,5,5,activation="relu"))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5,5,activation="relu"))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))
#
# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True,
#           nb_epoch=3)
#
#
#
# model.save('model.h5')
# print("Model Training Completed.")
#
# history = model.fit_generator(train_gen,
#                               samples_per_epoch=number_of_samples_per_epoch,
#                               nb_epoch=number_of_epochs,
#                               validation_data=validation_gen,
#                               nb_val_samples=number_of_validation_samples,
#                               verbose=1)
#
#
#
# ### print the keys contained in the history object
# print(history.history.keys())
#
# ### plot the training and validation loss for each epoch
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
