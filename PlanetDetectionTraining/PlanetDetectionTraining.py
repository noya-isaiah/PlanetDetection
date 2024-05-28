#imports
from calendar import EPOCH
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


data_dir = pathlib.Path('./Datasets/Planets_Moons_Data/Planets and Moons') #Data path
image_paths = list(data_dir.glob('**/*.jpg'))#Making a list of all images paths

#show pictures examples
for i in range(10):
  random_path = random.choice(image_paths)#Choosing random image path
  sample_img_path = str(random_path)
  sample_img = cv.imread(sample_img_path)#Load image in bgr
  sample_img = cv.cvtColor(sample_img, cv.COLOR_BGR2RGB)#Convert to rgb
  plt.imshow(sample_img)
  plt.axis("off")
  plt.title(label = random_path.parent.name)
  plt.show()

#Create numpy arrays of images and their labels
def data_load(image_paths):
    X = []
    Y = []

    random.shuffle(image_paths)#shuffle paths

    #running on all the images
    for path in image_paths:
        img = cv.imread(str(path))#load image in gbr
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)#convert into RGB
        planet_name = path.parent.name#label
        X.append(img)
        Y.append(planet_name)

    #convert X and Y into numpy array
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

#Divides the data into train, validation and test sets
def split_data(X, Y, split_percentage=[70, 20]):
    
    #calculate train, validation and test sets length
    train_size = int(split_percentage[0] / 100 * len(X))
    val_size = int(split_percentage[1] / 100 * len(X))
    test_size = len(X) - train_size - val_size

    #split the data according to the length calculation
    train_X, train_Y = X[:train_size], Y[:train_size]
    valid_X, valid_Y = X[train_size:train_size + val_size], Y[train_size:train_size + val_size]
    test_X, test_Y = X[train_size + val_size:], Y[train_size + val_size:]

    return (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y)

loaded_data= data_load(image_paths)#loading data

(train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = split_data(loaded_data[0], loaded_data[1])#getting train, validation and test sets

print("train length:", len(train_X))
print("validation length:", len(valid_X))
print("test length:", len(test_X))

#Normalize data
train_images = train_X / 255.0
test_images = test_X / 255.0
valid_images = valid_X / 255.0

#encoding labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_Y)
test_labels_encoded = label_encoder.transform(test_Y)
valid_labels_encoded = label_encoder.transform(valid_Y)
print(label_encoder.classes_)


#Creating model
model = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=(144, 256, 3)),#input layer
    MaxPooling2D(pool_size=2),
    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(11, activation='softmax')#output layer with 11 classes
])

#Plotting the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


#Configures the model for training with Adam optimizer and sparse categorical loss and metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

#Training model
history = model.fit(train_images, train_labels_encoded, validation_data=(valid_images, valid_labels_encoded), epochs=4)

#Loss and accuracy history
loss = history.history['loss']
accuracy = history.history['sparse_categorical_accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_sparse_categorical_accuracy']


#Plotting history
plt.plot(loss, 'r');
plt.plot(val_loss, 'g');
plt.title('Training loss',fontsize=20);
plt.plot(accuracy, 'b');
plt.plot(val_accuracy, 'y');
plt.title('Training accuracy',fontsize=20);
plt.legend(['loss', 'validation loss', 'accuracy', 'validation accuracy'])
plt.xlabel("epoch")


y_pred = model.predict(test_images)#Test predictions
predicted_labels = tf.argmax(y_pred, axis=1)#Taking the most probable class
predicted_labels = y_pred.numpy()

#create confusion matrix
confusion_matrix = tf.math.confusion_matrix(test_labels_encoded, predicted_labels)
confusion_matrix_np = confusion_matrix.numpy()


#plotting confusion matrix
confusion_matrix_dispaly = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_np)
confusion_matrix_dispaly.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

#printing classification report
print(classification_report(test_labels_encoded, predicted_labels))

#saving model
model.save('../models/planet_detecion_model.keras', save_format='tf')

