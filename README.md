
# Malaria infected cell classification

Malaria, a life-threatening infectious disease caused by Plasmodium parasites, is transmitted to humans through the bites of infected female Anopheles mosquitoes. Due to the severity and high infection rate, malaria poses a significant public health challenge globally, with 
an estimated 247 million cases and 84 endemic countries. Malaria is predominantly in sub-Saharan Africa and South Asia; Thailand has been historically endemic to malaria, with 
significant transmission rates in some regions.

The crucial step in diagnosing and treating malaria infection is identifying the Plasmodium spp. within red blood cells. The gold standard methods to identify and quantify are thick and thin film, an observation of stained-peripheral blood samples under a light microscope. This task necessitates skilled individuals with experience and expertise in identifying specific morphological features of parasite-infected cells. However, the manual identification of infected cells can be time-consuming and prone to error, particularly in resource-limited settings where trained personnel may be lacking. Thus, the advancement of computerized approaches for 
identifying infected cells is essential for enhancing the accuracy and efficiency of malaria 
diagnosis.

In recent years, the use of deep learning techniques (DL) has gained considerable attention 
in medical imaging analysis. These learning models utilize pre-trained models to fine-tune or 
extract a specific feature on large datasets to perform specified tasks such as object detection and image classification. Many investigations have been completed and tried to identify malaria 
infections with different techniques such as deep neural network (DNN), machine learning (ML),
deep learning (DL), convolutional neural network (CNN) and other models. Many approaches
show promising findings with high accuracy of prediction.

This study aims to use the Resnet-50 model, one of the high-accuracy convolutional 
neural networks reported by Nayak et al, to pre-trained malaria-infected cells dataset and 
classify input images whether they are infected or non-infected cells.
## Acknowledgements

 - [Malaria cell images dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
 - [Resnet-50](https://datagen.tech/guides/computer-vision/resnet-50/)
 

## Import dataset

Import images dataset from kaggel

```bash
# Import kaggel and temporary mount google drive

! pip install kaggle
from google.colab import drive
drive.mount('/content/drive')m run deploy
```
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```
Download and unzip dataset
```bash
!kaggle datasets download -d iarunava/cell-images-for-detecting-malaria
!unzip cell-images-for-detecting-malaria.zip
```
## Data explor

Import essential library
```bash
from PIL import Image
from glob import glob
import torchvision
from pathlib import Path
from collections import Counter
import os
import shutil
```
```bash
data = '/content/cell_images'

print('Total number of classes: ', os.listdir(data))
```
There are 2 classes in these dataset including ['Parasitized', 'Uninfected'] 

We can sampling some of the data by
```bash
P_paths = glob("cell_images/Parasitized/*.png")
P_img = Image.open(P_paths[10])
Image.open(P_paths[10])

Un_paths = glob("cell_images/Uninfected/*.png")
Un_img = Image.open(Un_paths[10])
Image.open(Un_paths[10])
```
```bash
# Recheck the size of data
print( 'Parasitized Size: ', P_img.size, "\n" 'Uninfected Size: ', Un_img.size)
```
Parasitized Size:  (130, 139) 

Uninfected Size:  (124, 124)

From this we can see that the size of dataset are not consistant and need to be resize to fit the model.

## Resize
The tarket size of the images will be (224x224)

```bash
# Create directory to Parasitized and Uninfected files

dir_path_P = '/content/cell_images/Parasitized'
dir_path_UnP = '/content/cell_images/Uninfected'

# Define the images size   
target_size = (224, 224)
```
Images in each folder will be resize and save on the original images.

```bash
# RESIZE DATA IN PARASITIZED FOLDER

for filename in os.listdir(dir_path_P):

  # Check if the file is an image
  if filename.endswith('.jpg') or filename.endswith('.png'):

    # Open the image
    img_path_P = os.path.join(dir_path_P, filename)
    img = Image.open(img_path_P)

    # Resize 
    img = img.resize(target_size)

    # Replace the original image
    img.save(img_path_P)
```
We can sampling some of the data to ensure that the images are resized.

```bash
Try_P_paths = glob('/content/cell_images/Parasitized/*.png')
Try_P = Image.open(Try_P_paths[10])
Image.open(Try_P_paths[10])
```
```bash
print( 'Parasitized Resize: ', Try_P.size)
```
The result "Parasitized Resize:  (224, 224)" has ensure that our dataset is already resized to the target size 

## Split data
 
The dataset was spitted into 3 seperate folders, test, validate and test with 2 sub folders including parasitized and uninfected.
```bash
# Import library

import shutil
import random
```
```bash
# Define the direction 

original_dir = '/content/cell_images'
train_dir = '/content/train'
val_dir = '/content/validation'
test_dir = '/content/test'

# Define the subdirectories in the train, validation, and test sets

train_subdirs = ['Parasitized', 'Uninfected']
val_subdirs = ['Parasitized', 'Uninfected']
test_subdirs = ['Parasitized', 'Uninfected']

# Split ratio

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
```
```bash
# Loop over each subdirectory in the original data directory
for subdir in os.listdir(original_dir):
    subdir_path = os.path.join(original_dir, subdir)
    
    # Skip any non-directory files
    if not os.path.isdir(subdir_path):
        continue
    
    # Create the corresponding subdirectories in the train, validation, and test sets
    train_subdir_path = os.path.join(train_dir, subdir)
    os.makedirs(train_subdir_path, exist_ok=True)
    
    val_subdir_path = os.path.join(val_dir, subdir)
    os.makedirs(val_subdir_path, exist_ok=True)
    
    test_subdir_path = os.path.join(test_dir, subdir)
    os.makedirs(test_subdir_path, exist_ok=True)
    
    # Get the list of all image files in the subdirectory
    images = os.listdir(subdir_path)
    
    # Shuffle the list of images
    random.shuffle(images)
    
    # Split the images into train, validation, and test sets
    num_images = len(images)
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)
    
    train_images = images[:num_train]
    val_images = images[num_train:num_train+num_val]
    test_images = images[num_train+num_val:]
    
    # Copy the images into the corresponding subdirectories in the train, validation, and test sets
    for image in train_images:
        src_path = os.path.join(subdir_path, image)
        dst_path = os.path.join(train_subdir_path, image)
        shutil.copy(src_path, dst_path)
    
    for image in val_images:
        src_path = os.path.join(subdir_path, image)
        dst_path = os.path.join(val_subdir_path, image)
        shutil.copy(src_path, dst_path)
    
    for image in test_images:
        src_path = os.path.join(subdir_path, image)
        dst_path = os.path.join(test_subdir_path, image)
        shutil.copy(src_path, dst_path)
```
## Model traning

```bash
# Import libraries for ResNet50

import matplotlib.pyplot as plt
import numpy as np
import PIL as image_lib
import tensorflow as tflow
from tensorflow.keras.layers import Flatten
from keras.layers.core import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
```
Data augmentation
```bash
# Data augmentation

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

```
Import the pretrained model from keras
```bash
# Import ResNet-50

resnet_model = Sequential()

pretrained_model = tflow.keras.applications.ResNet50(
    include_top=False,
    pooling='avg',classes=2,
    weights='imagenet')

for each_layer in pretrained_model.layers:

        each_layer.trainable=False

resnet_model.add(pretrained_model)

resnet_model.add(Flatten())

resnet_model.add(Dense(512, activation='relu'))

resnet_model.add(Dense(2, activation='softmax'))
```
Train the model with 10 epochs 
```bash
# Train model

optm = Adam(learning_rate=0.0001)

resnet_model.compile(
    optimizer= optm,
    loss ='categorical_crossentropy',
    metrics =['accuracy'])

```
```bash
epochs=10

history = resnet_model.fit(
  train_data,
  validation_data=val_data,
  epochs=epochs
)
```
Evaluate the model accuracy

```bash
# Evaluate the model

plt.plot(range(epochs), history.history['accuracy'], label="Training Accuracy")

plt.plot(range(epochs), history.history['val_accuracy'], label="Validation Accuracy")

# Plot model accuracy

plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.xlim([1, 10])
plt.ylim([0, 1])
plt.legend(['train', 'validation'])
```
From the result, the validation data shows the accuracy around 60%. 

## Test the model with test data
Load trained model
```bash
# Load model

model_path = '/content/gdrive/MyDrive/Resnet_model.pt'
load = torch.load(model_path)
```
Load the processed data form directory
```bash
# Load images from directories and create batches of images

test_data = test_datagen.flow_from_directory(
    test_dir,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
```
Predict the data with trained model
```bash
# predict

test_pred = load.predict(test_data) 
```
Plot the confusion matriz
```bash
from sklearn.metrics import confusion_matrix
import seaborn as sns

# convert prediction to class label
test_pred_label = np.argmax(test_pred, axis=1)

# Get the true labels from the test data generator
true_labels = test_data.classes

# confusion matrix
cm = confusion_matrix(true_labels, test_pred_label)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', 
            xticklabels=test_subdirs, yticklabels=test_subdirs)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
```

## Implement the medle with Gradio

Import Gradio libraries
```bash
!pip install gradio
!pip install git+https://github.com/huggingface/transformers
```

```bash
import gradio as gr
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
```
Load the model
```bash
# Load model

model_path = '/content/gdrive/MyDrive/Resnet_model.pt'
load = torch.load(model_path)
```
Define function to process input data from gradio interface
```bash
import numpy as np

def classify_cell(img):
    img = image.array_to_img(img)
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = tf.expand_dims(x, axis=0)
    pred = load.predict(x)
    predicted_class_idx = np.argmax(pred[0])
    predicted_class_prob = pred[0][predicted_class_idx]
    return {class_name[predicted_class_idx]: float(predicted_class_prob)}

```
Gradio interface
```bash
# Define the Gradio interface
inputs = gr.inputs.Image(shape=(224, 224))
outputs = gr.outputs.Label(num_top_classes=2)


interface = gr.Interface(fn=classify_cell, inputs=inputs, outputs=outputs, 
                         title='Malaria Classifier', 
                         description='Classify whether the cell is infected with Plasmodium or not')

# Launch the interface
interface.launch(debug="True")
```
