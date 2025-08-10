import numpy as np
import matplotlib.pyplot as plt
import pandas
from PIL import Image
import os 
from tensorflow import keras
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import pickle
from sklearn.metrics import accuracy_score, classification_report

'''
Important resources for this portion of code include the following:
https://www.analyticsvidhya.com/blog/2021/12/traffic-signs-recognition-using-cnn-and-keras-in-python/
https://towardsdatascience.com/recognizing-traffic-signs-with-over-98-accuracy-using-deep-learning-86737aedc2ab
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?resource=download
'''
def image_loading():
  all_images = []
  class_labels = []
  train = pandas.read_csv('Train.csv')
  gt_labels = train["ClassId"].values
  images = train["Path"].values
  cur_path = os.getcwd()
  for image, label in zip(images, gt_labels): 
    path = cur_path + '/'+ image
    print(path)
    new_image = Image.open(path) 
    new_image = new_image.resize((30,30)) 
    new_image = np.array(new_image) 
    all_images.append(new_image) 
    class_labels.append(label) 
    
  all_images = np.array(all_images)
  class_labels = np.array(class_labels)
  return (all_images,class_labels)

def data_creation(images, class_labels):
  x_train, x_val, y_train, y_val = train_test_split(images, class_labels, test_size=0.2)
  #print("shape sizing ",x_train.shape, x_val.shape, y_train.shape, y_val.shape)
  #one hot encoding
  y_train = to_categorical(y_train, 43)
  y_val = to_categorical(y_val, 43)
  return x_train, x_val, y_train, y_val

def build_model(x_train):
  #creation of model
  model = Sequential()
  model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=x_train.shape[1:]))
  model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Dropout(rate=0.25))
  model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
  model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Dropout(rate=0.25))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(rate=0.5))
  model.add(Dense(43, activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

def build_model_ds(x_train):
  #creation of model
  model = Sequential()
  model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=x_train.shape[1:]))
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(256, activation=None))
  model.add(Dense(256, activation=None))
  model.add(Dropout(rate=0.5))
  model.add(Dense(43, activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

def train(model, epoch_num, x_train, y_train, batch_size, x_val, y_val):
  history = model.fit(x_train, y_train, batch_size=32, epochs=epoch_num, validation_data=(x_val, y_val))
  '''
  with open('./trainHistoryDict', 'wb') as file_pi:
      pickle.dump(history.history, file_pi)
  '''
  return history.history, model


def plot_train_val(history):
  plt.figure(0)
  plt.plot(history['accuracy'], label='training accuracy')
  plt.plot(history['val_accuracy'], label='val accuracy')
  plt.title('Accuracy')
  plt.xlabel('epochs')
  plt.ylabel('accuracy')
  plt.legend()
  plt.show()
  plt.figure(1)
  plt.plot(history['loss'], label='training loss')
  plt.plot(history['val_loss'], label='val loss')
  plt.title('Loss')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend()
  plt.show()

def test_model(model):
  #testing accuracy on test dataset
  y_test = pandas.read_csv('Test.csv')
  labels = y_test["ClassId"].values
  imgs = y_test["Path"].values
  test_images=[]
  for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    test_images.append(np.array(image))
  X_test=np.array(test_images)
  predict_x = model.predict(X_test)
  pred=np.argmax(predict_x,axis=1)
  #Accuracy with the test data
  print(accuracy_score(labels, pred))
  print(classification_report(labels,pred))
  model.save("traffic_classifier.h5")

if __name__=="__main__":
  images, class_labels = image_loading()
  x_train, x_val, y_train, y_val = data_creation(images, class_labels)
  model = build_model(x_train)

  #training
  epoch_num = 15
  batch_size = 32
  history, model = train(model, epoch_num, x_train, y_train, batch_size, x_val, y_val)
  model.save("my_model3.h5")

  '''
  with open('./trainHistoryDict', "rb") as file_pi:
      history = pickle.load(file_pi)
    
  model = keras.models.load_model('my_model.h5')
  '''
  plot_train_val(history)
  test_model(model)