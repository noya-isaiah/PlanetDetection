#imports
import pathlib
import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

interactive_mode = False

data_dir = pathlib.Path('./Datasets/Planets_Moons_Data/Planets and Moons') 
image_files = list(data_dir.glob('**/*.jpg'))

#picturs examples
for i in range(10):
  sample_img_path = str(random.choice(image_files))
  planet_name = sample_img_path.split('/')[-1]
  sample_img = cv.imread(sample_img_path)
  if (interactive_mode == True) :
    plt.imshow(cv.cvtColor(sample_img, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(label = str(planet_name))
    plt.show()

def data_load(image_files):
    X = []
    Y = []

    random.shuffle(image_files)

    for file in image_files:
        img = cv.imread(str(file))
        planet_name = file.parent.name

        X.append(img)
        Y.append(str(planet_name))

    X = np.array(X)

    return X, Y

loaded_data= data_load(image_files)

def split_data(X, Y, split_percentage=[70, 20]):
    train_size = int(split_percentage[0] / 100 * len(X))
    val_size = int(split_percentage[1] / 100 * len(X))
    test_size = len(X) - train_size - val_size

    train_X, train_Y = X[:train_size], Y[:train_size]
    valid_X, valid_Y = X[train_size:train_size + val_size], Y[train_size:train_size + val_size]
    test_X, test_Y = X[train_size + val_size:], Y[train_size + val_size:]

    return (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y)

(train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = split_data(loaded_data[0], loaded_data[1])

print(len(train_X), "train")
print(len(valid_X), "valid")
print(len(test_X), "test")

print(train_X.shape)

train_images = train_X / 255.0
test_images = test_X / 255.0
valid_images = valid_X / 255.0

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_Y)
test_labels_encoded = label_encoder.transform(test_Y)
valid_labels_encoded = label_encoder.transform(valid_Y)

print(label_encoder.classes_)

model = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=(144, 256, 3)),
    MaxPooling2D(pool_size=2),
    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(11, activation='softmax')
])

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

history = model.fit(train_images, train_labels_encoded, validation_data=(valid_images, valid_labels_encoded), epochs=4)

loss = history.history['loss']
accuracy = history.history['sparse_categorical_accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_sparse_categorical_accuracy']

if (interactive_mode == True) :
    plt.plot(loss, 'g');
    plt.plot(val_loss, 'r');
    plt.title('Training loss',fontsize=20);

    plt.plot(accuracy, 'b');
    plt.plot(val_accuracy, 'r');
    plt.title('Training accuracy',fontsize=20);

y_pred = model.predict(test_images)
predicted_labels = tf.argmax(y_pred, axis=1)
true_labels = test_labels_encoded
predicted_labels = predicted_labels.numpy()

confusion_matrix = tf.math.confusion_matrix(true_labels, predicted_labels)
confusion_matrix_np = confusion_matrix.numpy()


if (interactive_mode == True) :
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_np)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

print(classification_report(true_labels, predicted_labels))

model.save('../models/planet_detecion_model.keras', save_format='tf')

