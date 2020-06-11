from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

# Cascade Classifier path
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
# Trained model path
classifier = load_model(r'Emotion_little_vgg .h5')


# Name of the expressions
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
# Starting the camera
cap = cv2.VideoCapture(0)
while True:
#     Grab a single frame of the video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
#   Draw rectangle and put the text  
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w, y+h), (255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray])!=0: # If there is a face
            roi = roi_gray.astype('float')/255.0 # Dividing by 255 is to reduce the pixel size
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
#             make prediction on the roi(face) , then lookup the class
            preds = classifier.predict(roi)[0] # Predict the model
            label = class_labels[preds.argmax()]
            label_position = (x,y)
#             Puttinng the emotion text in face
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
#     Show the frame
    cv2.imshow('Emotion Detector', frame)
#     Pressing the q for closing the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
            