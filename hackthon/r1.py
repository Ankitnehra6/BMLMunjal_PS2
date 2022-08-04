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
d=cv.VideoCapture("y.mp4")
  
  
while True:  
    # Read the frame  
    _, img = cap.read()  
    rt,video=c.read()
  
    # Convert to grayscale  
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
  
    # Detect the faces  
    faces = haar_cascade.detectMultiScale(gray, 1.1, 4)  
  
    # Draw the rectangle around each face  
    for (x,y,w,h) in faces:
        
        faces_roi = gray[y:y+h,x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
       
        print(f'Label = {people[label]} with a confidence of {confidence}')
        
        if people[label]=="carry"  or people[label]=="elvish" or people[label]=="Ankit":
            cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        elif people[label]=="ab":
             cv.putText(img, str("default"), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
            
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    
        #if people[label]=="Ankit" or people[label]=="elvish":
           # rt,video=d.read()
       # else:
            #rt,video=c.read()
            
            
        
        cv.rectangle(video, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        #cv.rectangle(i, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        #cv.resize(video,(x:x+w,y:y+h))
        if rt==True: 
            g=video[x:x+w+20,y:y+h+20]
        #print(x,y)
        #cv.imshow('h',g)  
        #ad=cv.addWeighted(img,alpha,g,1-alpha,0)
        cv.imshow('Video',img ) 
  
      
        
     
        
        #cv.imshow("Vide",crop)
        #cv.imshow('s',crop)
        #v=cv.merge(img,crop)
        #cv.imshow("new",v)
        
       
  
    # Display  
    #
   # cv.imshow('s',crop)
  
    # Stop if escape key is pressed  
    k = cv.waitKey(30) & 0xff  
    if k==27:  
        break  
          
# Release the VideoCapture object  
cap.release() 