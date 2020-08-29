# https://www.tensorflow.org/tutorials/keras/classification

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

'''
Load the dataset
'''
# import the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# names of the labels, that are integers ranging from 0 to 9
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''
Explore the data
'''
print('Data information:')
# in the training set there are 60,000 images with 28x28 pixels and there are 60,000 labels
print('train_images.shape =', train_images.shape)
print('train_labels size =', len(train_labels))
# in the test set there are 10,000 images with 28x28 pixels and there are 10,000 labels
print('test_images.shape =', test_images.shape)
print('test_labels size =', len(test_labels))

'''
Preprocess the data
'''
# inspecting the first image in the training set, pixels values range from 0 to 255
print('\nPlot of the first train image...')
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# scale the values to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# verify that the data is in the correct format
# display the first 25 images from the training set and there respective class name
print('\nPlot of the first 25 train images...')
plt.figure(figsize=(10, 10))
for index in range(25):
    plt.subplot(5, 5, index + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[index], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[index]])
plt.show()

'''
Build the model
'''
'''Set up the layers'''
# first - transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a
# # one-dimensional array (of 28 * 28 = 784 pixels)
# second - has 128 nodes that are fully connected
# third - has 10 nodes that are fully connected, that represent the 10 different classes
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

'''Compile the model'''
# optimizer - how the model is updated based on the data it sees and its loss function
# loss function - measures how accurate the model is during training. The goal is to minimize this function to "steer"
# # the model in the right direction
# metrics - used to monitor the training and testing steps
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

'''
Train the model
'''
'''Feed the model'''
print('\nTraining...')
model.fit(train_images, train_labels, epochs=10)

'''Evaluate accuracy'''
print('\nTesting...')
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy:', test_acc)

'''Make predictions'''
# attach a softmax layer to convert the logits to probabilities, which are easier to interpret
# logits - https://developers.google.com/machine-learning/glossary#logits
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print('\nFirst image prediction:')
print('> probabilities:', predictions[0])
_predicted_label = int(np.argmax(predictions[0]))
print('> label with highest confidence value: %i (%s)' % (_predicted_label, class_names[_predicted_label]))
print('> test label: %i (%s)' % (test_labels[0], class_names[test_labels[0]]))
print('> prediction is %sCORRECT' % ("" if test_labels[0] == _predicted_label else "IN"))


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = int(np.argmax(predictions_array))
    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            class_names[predicted_label],
            100 * np.max(predictions_array),
            class_names[true_label]),
        color='blue' if predicted_label == true_label else 'red')


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    this_plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')


'''Verify the predictions'''
num_rows, num_cols = 5, 3
num_images = num_rows * num_cols
print('\nPlot of the prediction of the first %i test images...' % num_images)
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for index in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * index + 1)
    plot_image(index, predictions[index], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * index + 2)
    plot_value_array(index, predictions[index], test_labels)
plt.tight_layout()
plt.show()

'''
Use the train model
'''
print('\nPredict the 12th image of the test collection...')
img = test_images[12]
# Add the image to a batch where it's the only member, because tf.keras models are optimized to make predictions on a
# # batch, or collection, of examples at once
img = (np.expand_dims(img, 0))
predictions_single = probability_model.predict(img)
print('Predictions array:', predictions_single)
print('Plot of the prediction of the 12th image...')
plot_value_array(12, predictions_single[0], test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()
