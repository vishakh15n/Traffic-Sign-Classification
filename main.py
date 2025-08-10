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
import cv2

classes = { 1:'Speed limit (20km/h)',
           2:'Speed limit (30km/h)',
           3:'Speed limit (50km/h)',
           4:'Speed limit (60km/h)',
           5:'Speed limit (70km/h)',
           6:'Speed limit (80km/h)',
           7:'End of speed limit (80km/h)',
           8:'Speed limit (100km/h)',
           9:'Speed limit (120km/h)',
           10:'No passing',
           11:'No passing veh over 3.5 tons',
           12:'Right-of-way at intersection',
           13:'Priority road',
           14:'Yield',
           15:'Stop',
           16:'No vehicles',
           17:'Veh > 3.5 tons prohibited',
           18:'No entry',
           19:'General caution',
           20:'Dangerous curve left',
           21:'Dangerous curve right',
           22:'Double curve',
           23:'Bumpy road',
           24:'Slippery road',
           25:'Road narrows on the right',
           26:'Road work',
           27:'Traffic signals',
           28:'Pedestrians',
           29:'Children crossing',
           30:'Bicycles crossing',
           31:'Beware of ice/snow',
           32:'Wild animals crossing',
           33:'End speed + passing limits',
           34:'Turn right ahead',
           35:'Turn left ahead',
           36:'Ahead only',
           37:'Go straight or right',
           38:'Go straight or left',
           39:'Keep right',
           40:'Keep left',
           41:'Roundabout mandatory',
           42:'End of no passing',
           43:'End no passing vehicle with a weight greater than 3.5 tons' }

if __name__=="__main__":
    recognition_model = keras.models.load_model('my_model.h5')
    vid = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        #print(type(frame))
        test_images=[]
        image = cv2.resize(frame,(30,30))
        #print(image)
        test_images.append(image)
        #print(np.array(image).shape)

        X_test=np.array(test_images)
        predict_x = recognition_model.predict(X_test)
        pred=np.argmax(predict_x,axis=1)
        pred_val = predict_x[0][pred-1]
        if(pred_val > 0.5):
            text = classes[pred]
            cv2.putText(frame, 
                    text, 
                    (50, 50), 
                    font, 1, 
                    (0, 255, 255), 
                    2, 
                    cv2.LINE_4)
  
        cv2.imshow('frame', frame)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()