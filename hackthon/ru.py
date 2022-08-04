import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar.xml')

people = ['Ankit','carry','elvish']

# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
# To capture video from webcam.   
cap = cv.VideoCapture(0)  
c=cv.VideoCapture("videoplayback.mp4")
  
while True:  
    # Read the frame  
    _, img = cap.read()  
    rt,video=c.read()
  
    # Convert to grayscale  
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
  
    # Detect the faces  
    faces = haar_cascade.detectMultiScale(gray, 1.1, 4)  
    x,y,w,h=0,0,0,0
    # Draw the rectangle around each face  
    for (x,y,w,h) in faces:
        
        faces_roi = gray[y:y+h,x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')

        cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    
    #crop= cv.rectangle(video, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    
    cv.startWindowThread()
    cv.namedWindow('image')
    cv.imshow('image',img)
    if rt==True: 
            
        img=video[x:x+w,y:y+h]
        winname = "imag"
        cv.namedWindow(winname)        # Create a named window
        cv.moveWindow(winname, x,y)  # Move it to (40,30)
        cv.imshow(winname, img) 
    #print(x,y)  
        #cv.imshow('Vide',crop)  
    #cv.imshow('Video', img)  
    #cv.namedWindow('crop',cv.WND_PROP_FULLSCREEN)
    #cv.setWindowProperty('crop',cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
        
       
  
    # Display  
    #cv.imshow('Video', img)  
   # cv.imshow('s',crop)
  
    # Stop if escape key is pressed  
    k = cv.waitKey(30) & 0xff  
    if k==27:  
        break  
          
# Release the VideoCapture object  
cap.release() 