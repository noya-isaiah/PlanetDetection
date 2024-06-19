#imports
import numpy as np
import sys
import keras
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QFileDialog, QPushButton, QLabel
from PyQt5.QtGui import QPixmap
from PIL import Image

#Planet_Detection_UI subclass of QDialog
class Planet_Detection_UI(QDialog):
    
    #Creates the main window, buttons and the labels
    def setupUI(self, MainWindow):
        
        #Creats window and resize it
        MainWindow.setObjectName("Noya's Planets Detection")
        MainWindow.resize(611, 505)
        

        #Add a label with the text "Upload Planet"
        self.titleLabel = QLabel("Upload Planet", MainWindow)
        self.titleLabel.setGeometry(155,0, 500, 50)
        #Customize the font of the label
        font = self.titleLabel.font()
        font.setFamily("Calibri Light")
        font.setPointSize(24)
        font.setBold(True)
        self.titleLabel.setFont(font)
        

        #Add a QLabel for displaying the uploaded photo
        self.photoLabel = QLabel(MainWindow)
        self.photoLabel.setGeometry(150, 100, 300, 300)
        self.photoLabel.setScaledContents(True)#Fitting the image to the size of the label
        
        #Add a button for browsing files
        self.browseButton = QPushButton("BrowseFile", MainWindow)
        self.browseButton.setGeometry(250, 420, 100, 30)
        self.browseButton.clicked.connect(self.browseFile)

        #Add a button for prediction
        self.predictButton = QPushButton("Predict", MainWindow)
        self.predictButton.setGeometry(250, 460, 100, 30)
        self.predictButton.setVisible(False)#Initially hidden
        self.predictButton.clicked.connect(self.predictPhoto)
        
        #Add a label for the prediction results
        self.resultLabel = QLabel("", MainWindow)
        self.resultLabel.setGeometry(100,0, 500, 50)
        self.resultLabel.setVisible(False)
        #Customize the font of the label
        font = self.resultLabel.font()
        font.setFamily("Calibri Light")
        font.setPointSize(24)
        font.setBold(True)
        self.resultLabel.setFont(font)


    #Selecting photo from browse file and makes the predict button visible 
    def browseFile(self):
        #Open a file dialog and get the selected file path
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()","", 
                                                  "Image Files (*.jpg)")
        #Display the selected photo
        pixmap = QPixmap(fileName)
        self.photoLabel.setPixmap(pixmap)
        
        #Opening image
        self.image = Image.open(fileName)

        #Show the predict button
        self.predictButton.setVisible(True)

    #Praparing photo for prediction
    def preprocessImage(self, image): 
        image = image.resize((256, 144))#Resizing photo
        arr = np.array(image)#Convert into numpy array
    
        #Normalization
        arr = arr / 255.0
    
        #Adding dimension
        arr = np.expand_dims(arr, axis=0)
    
        return arr

    #Converting predictions result from numbers to verabel class
    def ResultsMapping(self, prediction):
        class_index = np.argmax(prediction, axis=1)[0]#Taking the most probable class
        class_names = ['Earth', 'Jupiter', 'MakeMake', 'Mars', 'Mercury', 
                       'Moon', 'Neptune', 'Pluto', 'Saturn', 'Uranus', 'Venus']
        return str(class_names[class_index])
    

    def predictPhoto(self):
        #Cleaning the window
        self.titleLabel.setVisible(False)
        self.photoLabel.setVisible(False)
        self.browseButton.setVisible(False)
        self.predictButton.setVisible(False)

        #Preaparing photo for prediction
        input_data = self.preprocessImage(self.image)
        
        #Prediction
        prediction = model.predict(input_data)
        
        #Mapping the prediction result for planet name
        predicted_class = self.ResultsMapping(prediction)

        #Showing prediction in the window
        self.resultLabel.setText("Your planet is: "+ predicted_class)
        self.resultLabel.setVisible(True)

#Main
model = keras.models.load_model('../models/planet_detecion_model.keras')#Loading model
app = QApplication(sys.argv)#Creates QApplication which is controlling the application's event loop 
MainWindow = QMainWindow()#Creates interface window
ui = Planet_Detection_UI()#Creates planet detection ui
ui.setupUI(MainWindow)#Realization main window
MainWindow.show()#Showing window on screen
sys.exit(app.exec_())#Starts the event loop and when the main window is closed it returns the exit status