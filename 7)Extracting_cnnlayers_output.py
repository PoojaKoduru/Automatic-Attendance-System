from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Model
import tensorflow

#create a deep learning model and add required layers
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
model.add(Convolution2D(64, (7, 7),strides=(2, 2), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), strides=(2, 2), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
print(model.summary())


# Let us print the first hidden layer as the layer of interest.
layer = model.layers  # Conv layers at 1, 4, 6, 8, 10, 12, 14
filters, biases = model.layers[1].get_weights()
print(layer[1].name, filters.shape)
fig1 = plt.figure(figsize=(8, 12))
columns = 8
rows = 8
n_filters = columns * rows # to display 64 filters as a matrix
for i in range(1, n_filters + 1):
    f = filters[:, :, :, i - 1]
    fig1 = plt.subplot(rows, columns, i)
    fig1.set_xticks([])  # Turn off axis
    fig1.set_yticks([])
    plt.imshow(f[:, :, 0], cmap='gray')  # Show only the filters from 0th channel (R)
    # ix += 1
plt.show()

#### Now plot filter outputs
# Define a new truncated model to only include the conv layers of interest
# conv_layer_index = [1, 4, 6, 8, 10, 12, 14]
conv_layer_index = [1, 6, 12,18]  # TO define a shorter model
#creating a model with image as input and required convolutional layers as output
outputs = [model.layers[i].output for i in conv_layer_index]
model_short = Model(inputs=model.inputs, outputs=outputs)
print(model_short.summary())

# Input shape to the model is 224 x 224. SO resize input image to this shape.
from keras.preprocessing.image import load_img, img_to_array
img = load_img('obama.jpg', target_size=(224, 224))  # VGG user 224 as input
# convert the image to an array
img = img_to_array(img)
# expand dimensions to match the shape of model input
img = np.expand_dims(img, axis=0)
# Generate feature output by predicting on the input image
feature_output = model_short.predict(img)

#plot 8*8 matrix of 64 filters output for each convolutional layer
columns = 8
rows = 8
for ftr in feature_output:
    # pos = 1
    fig = plt.figure(figsize=(12, 12))
    for i in range(1, columns * rows + 1):
        fig = plt.subplot(rows, columns, i)
        fig.set_xticks([])  # Turn off axis
        fig.set_yticks([])
        plt.imshow(ftr[0, :, :, i - 1], cmap='gray')
        # pos += 1
    plt.show()
