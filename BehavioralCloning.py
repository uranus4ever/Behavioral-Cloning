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
    bottom = 20
    return image[top:bottom, :]

def resize(image, new_size):
    return scipy.misc.imresize(image, new_size)

def process_img(image, steering_angle):
    image = crop_img(image)
    image, steering_angle = random_flip(image, steering_angle, flipping_prob=0.5)
    image = resize(image, new_size=(64, 64))
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
    else indicator_img == 2: # right
        img = cv2.imread(path(line[2]))
        angle = float(line[3]) - correction
    return  img, angle

def LoadData(DataSaved = True):
    if DataSaved:
        print('Loading data...')
        with open('./data_pickle.p', mode="rb") as f:
            data_pickle = pickle.load(f)
        X_train, y_train = data_pickle["X_train"], data_pickle["y_train"]
        print('Data loaded.')

    else:
        lines = []
        with open('./data/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                lines.append(row)
        print('line = {}'.format(len(lines)))

        car_images = []
        steering_angles = []
        # create adjusted steering measurements for the side camera images
        correction = 0.2  # this is a parameter to tune
        print('processing images...')
        for line in lines:
            image, angle = select_img(line)
            image_process = process_img(image)
            car_images.extend(image_process)
            steering_angles.extend(angle)

        X_train = np.array(car_images)
        y_train = np.array(steering_angles)
        print('Total Training Images = {}'.format(len(X_train)))

        # save data into pickle file
        data_pickle = {}
        data_pickle['X_train'] = X_train
        data_pickle['y_train'] = y_train
        pickle.dump(data_pickle, open("./data_pickle.p", "wb"))
        print('Data saved')

    return X_train, y_train

def generator(batch_size=64):

    while True:
        X_batch = []
        y_batch = []
        images = get_next_image_files(batch_size)
        for img_file, angle in images:
            raw_image = plt.imread(IMG_PATH + img_file)
            raw_angle = angle
            new_image, new_angle = generate_new_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)

        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

        yield np.array(X_batch), np.array(y_batch)


X_train, y_train = LoadData(DataSaved=True)
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))

model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True,
          nb_epoch=3)



model.save('model.h5')
print("Model Training Completed.")

history_object = model.fit_generator(X_train, y_train, samples_per_epoch =
    len(train_samples), validation_data = validation_generator,
    nb_val_samples = len(validation_samples),
    nb_epoch=5, verbose=1)



### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
