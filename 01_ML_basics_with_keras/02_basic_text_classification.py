# https://www.tensorflow.org/tutorials/keras/text_classification

import os
import re
import shutil
import string

import matplotlib.pyplot as plt
import tensorflow as tf

'''
Download and explore the dataset
'''
print('Downloading the dataset...')
dataset = tf.keras.utils.get_file(
    'aclImdb_v1.tar.gz', 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', untar=True
)
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
print('\nDataset dir location:', os.path.abspath(dataset_dir))
print('List dataset dir:', os.listdir(dataset_dir))

train_dir = os.path.join(dataset_dir, 'train')
print('\nTrain dir location:', os.path.abspath(train_dir))
print('List train dir:', os.listdir(train_dir))

sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print('\nContent of the file %s:\n%s' % (os.path.abspath(sample_file), f.read()))

'''
Load the dataset
'''
# remove extra folders, because the utility text_dataset_from_directory expects a directory structure as follows:
# main_directory/
# ...class_a/
# ......a_text_1.txt
# ......a_text_2.txt
# ...class_b/
# ......b_text_1.txt
# ......b_text_2.txt
shutil.rmtree(os.path.join(train_dir, 'unsup'))

#
batch_size = 32
seed = 42
#

# only use 80% of the train set, the 20% left is for validation purposes
print('\nTrain set...')
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    os.path.abspath(train_dir),
    subset='training',
    validation_split=0.2,
    batch_size=batch_size,
    seed=seed
)

# iterate over the dataset and print out a few examples
number_of_examples = 3
print('\nIterate over %i examples from the train set' % number_of_examples)
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(number_of_examples):
        print('Example %i:' % (i + 1))
        print('> Review', text_batch.numpy()[i])
        print('> Label', label_batch.numpy()[i])

print('\nLabels values:')
print('Label 0 corresponds to', raw_train_ds.class_names[0])
print('Label 1 corresponds to', raw_train_ds.class_names[1])

print('\nValidation set...')
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    os.path.abspath(train_dir),
    subset='validation',
    validation_split=0.2,
    batch_size=batch_size,
    seed=seed
)

print('\nTest set...')
test_dir = os.path.join(dataset_dir, 'test')
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    os.path.abspath(test_dir),
    batch_size=batch_size
)

'''
Prepare the dataset for training
'''
# the class below is helpful to standardize, tokenize, and vectorize the data
# Standardization - preprocessing the text, typically to remove punctuation or HTML elements to simplify the dataset
# Tokenization - splitting strings into tokens (for example, splitting a sentence into individual words, by splitting
#   on whitespace)
# Vectorization - converting tokens into numbers so they can be fed into a neural network
tf.keras.layers.experimental.preprocessing.TextVectorization()


# custom standardization function to remove the HTML tag "<br />" because the default function only converts the text to
#   lowercase and strips the punctuation
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


max_features = 10000
sequence_length = 250

vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    standardize=custom_standardization,
    # to create unique integer indices for each token
    output_mode='int',
    max_tokens=max_features,
    output_sequence_length=sequence_length
)

# make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
# calling adapt will cause the model to build an index of strings to integers
vectorize_layer.adapt(train_text)


def vectorize_review(review):
    # transforms the tensor with the text to a tensor with an array with the text
    array_with_text = tf.expand_dims(review, -1)
    return vectorize_layer(array_with_text)


# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review = text_batch[0]
print('\nFirst review vectorized...')
print('Review:', first_review)
vectorized_review = vectorize_review(first_review)
print('Vectorized review:', vectorized_review)

print('\nLookup the token (string) that each integer corresponds to:')
tokens = vectorized_review[0]
for index in range(3):
    token = tokens[index]
    print('\t>', token, '=', vectorize_layer.get_vocabulary()[token])
print('Vocabulary size:', len(vectorize_layer.get_vocabulary()))


# apply the TextVectorization layer to the train, validation, and test dataset
def vectorize_review_plus_label(review, label):
    return vectorize_review(review), label


train_ds = raw_train_ds.map(vectorize_review_plus_label)
val_ds = raw_val_ds.map(vectorize_review_plus_label)
test_ds = raw_test_ds.map(vectorize_review_plus_label)

'''
Configure the dataset for performance
'''
# .cache() - keeps data in memory after it's loaded off disk. This will ensure the dataset does not become a bottleneck
#     while training your model. If your dataset is too large to fit into memory, you can also use this method to create
#     a performant on-disk cache, which is more efficient to read than many small files.
# .prefetch() - overlaps data preprocessing and model execution while training
# https://www.tensorflow.org/guide/data_performance
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

'''
Create the model
'''
# The layers are stacked sequentially to build the classifier:
# 1. The first layer is an Embedding layer. This layer takes the integer-encoded reviews and looks up an embedding
#     vector for each word-index. These vectors are learned as the model trains. The vectors add a dimension to the
#     output array. The resulting dimensions are: (batch, sequence, embedding). To learn more about embeddings, see
#     the word embedding tutorial (https://www.tensorflow.org/tutorials/text/word_embeddings).
# 2. Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the
#     sequence dimension. This allows the model to handle input of variable length, in the simplest way possible.
# 3. This fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.
# 4. The last layer is densely connected with a single output node.
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_features + 1, 16),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])
print('\nModel summary:')
model.summary()

'''
Loss function and optimizer
'''
# loss function tf.keras.losses.BinaryCrossentropy since this is a binary classification problem and the model outputs
#     a probability (a single-unit layer with a sigmoid activation)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0)
)

'''
Train the model
'''
print('\nTraining...')
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

'''
Evaluate the model
'''
print('\nEvaluating...')
loss, accuracy = model.evaluate(test_ds)
print('Results:')
print('\t> Loss:', loss)
print('\t> Accuracy:', accuracy)

'''
Create a plot of accuracy and loss over time
'''
# model.fit() returns a History object that contains a dictionary with everything that happened during training
history_dict = history.history
# there are four entries: one for each monitored metric (loss and accuracy) during training and validation
print('\nHistory entries;', history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

print('\nPlot with training and validation loss...')
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print('\nPlot with training and validation accuracy...')
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

'''
Export the model
'''
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    tf.keras.layers.Activation('sigmoid')
])

export_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# test with raw_test_ds, which yields raw strings
print('\nEvaluate the test dataset that yields raw strings...')
loss, accuracy = export_model.evaluate(raw_test_ds)
print('Accuracy:', accuracy)
