import csv
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

# Get images' paths from the csv file
lines = []
with open("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print (lines)
print ("how many lines: ", len(lines))
images = []
measurements = []
c = 0.20 # Correction constant for right and left images

# Get images 
for line in lines[1:]:
    
    # Get center images
    center_path = line[0]
    image_path = center_path.split('/')[-1]
    read_path = "./data/IMG/" + image_path
    image = cv2.imread(read_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
    # Get left images
    left_path = line[1]
    image_path = left_path.split('/')[-1]
    read_path = "./data/IMG/" + image_path
    image = cv2.imread(read_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement + c)
    
    # Get right images
    right_path = line[2]
    image_path = right_path.split('/')[-1]
    read_path = "./data/IMG/" + image_path
    image = cv2.imread(read_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement - c)

# Define the generator    
def generator(samples, batch_size=32):
    """
    Generate the data (list of pairs of images and measurements) for later process
    """
    num_samples = len(samples)
    while 1:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            imgs=[]
            steer=[]
            for image, measurement in batch_samples:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                imgs.append(image)
                steer.append(measurement)
                # Flipping
                imgs.append(cv2.flip(image,1))
                steer.append(measurement*-1.0)
            inputs = np.array(imgs)
            outputs = np.array(steer)
            yield sklearn.utils.shuffle(inputs,outputs)
            
# Split data
from sklearn.model_selection import train_test_split

samples = list(zip(images, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('Train samples:{}'.format(len(train_samples)))
print('Validation samples:{}'.format(len(validation_samples)))

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
import sklearn

# Model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)
model.save('model3.h5')

# Training history visualization
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])
model.save('model3.h5')
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
