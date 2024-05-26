import numpy as np
import sys
import keras
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QFileDialog, QPushButton, QLabel
from PyQt5.QtGui import QPixmap, QImage

class Ui_MainWindow(QDialog):
    
    def setupUI(self, MainWindow):
        MainWindow.setObjectName("Noya's Planets Detection")
        MainWindow.resize(611, 505)
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)
        
        # Add a label with the text "Upload Planet"
        self.titleLabel = QLabel("Upload Planet", self.centralwidget)
        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.titleLabel.setGeometry(0, 0, 611, 50)
        self.titleLabel.setObjectName("titleLabel")

        # Customize the font of the label
        font = self.titleLabel.font()
        font.setFamily("Calibri Light")
        font.setPointSize(24)
        font.setBold(True)
        self.titleLabel.setFont(font)

        # Add a QLabel for displaying the uploaded photo
        self.photoLabel = QLabel(self.centralwidget)
        self.photoLabel.setGeometry(150, 100, 300, 300)
        self.photoLabel.setObjectName("photoLabel")
        self.photoLabel.setScaledContents(True)  # Ensure the image scales to fit the label
        
        # Add a button for browsing files
        self.browseButton = QPushButton("Browse File", self.centralwidget)
        self.browseButton.setGeometry(250, 420, 100, 30)
        self.browseButton.setObjectName("browseButton")
        self.browseButton.clicked.connect(self.browseFile)

        # Add a button for prediction
        self.predictButton = QPushButton("Predict", self.centralwidget)
        self.predictButton.setGeometry(250, 460, 100, 30)
        self.predictButton.setObjectName("predictButton")
        self.predictButton.setVisible(False)  # Initially hidden
        self.predictButton.clicked.connect(self.predictPhoto)
        
        # Add a label for the prediction results
        self.resultLabel = QLabel("", self.centralwidget)
        self.resultLabel.setAlignment(Qt.AlignCenter)  # Align text to center
        self.resultLabel.setGeometry(0, 0, 611, 50)
        self.resultLabel.setObjectName("resultLabel")
        self.resultLabel.setVisible(False)

        # Customize the font of the label
        font = self.resultLabel.font()
        font.setFamily("Calibri Light")
        font.setPointSize(24)
        font.setBold(True)
        self.resultLabel.setFont(font)

    def browseFile(self):
        # Open a file dialog and get the selected file path
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", 
                                                  "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if fileName:
            # Display the selected photo
            pixmap = QPixmap(fileName)
            self.photoLabel.setPixmap(pixmap)
            # Show the predict button
            self.predictButton.setVisible(True)

    def predictPhoto(self):
        self.titleLabel.setVisible(False)
        self.photoLabel.setVisible(False)
        self.browseButton.setVisible(False)
        self.predictButton.setVisible(False)

        pixmap = self.photoLabel.pixmap()
        image = pixmap.toImage()
        input_data = self.preprocessImage(image)
            
        prediction = model.predict(input_data)
        predicted_class = self.decodePrediction(prediction)

        self.resultLabel.setText(f"Your planet is: {predicted_class}")
        self.resultLabel.setVisible(True)

    def preprocessImage(self, image):
        image = image.scaled(256, 144)
    
        # Convert QImage to numpy array
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)
        arr = arr[:, :, :3]
    
        # Normalize pixel values to range [0, 1]
        arr = arr / 255.0
    
        # Add batch dimension
        arr = np.expand_dims(arr, axis=0)
    
        return arr

    def decodePrediction(self, prediction):
        class_index = np.argmax(prediction, axis=1)[0]
        class_names = ['Earth', 'Jupiter', 'MakeMake', 'Mars', 'Mercury', 'Moon', 'Neptune', 'Pluto', 'Saturn', 'Uranus', 'Venus']
        return class_names[class_index]

model = keras.models.load_model('../models/planet_detecion_model.keras')
app = QApplication(sys.argv)
MainWindow = QMainWindow()
ui = Ui_MainWindow()
ui.setupUI(MainWindow)
MainWindow.show()
sys.exit(app.exec_())

