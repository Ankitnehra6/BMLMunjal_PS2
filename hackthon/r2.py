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
fourcc=cv.VideoWriter_fourcc(*'XVID')

  
while True:  
    # Read the frame  
    _, img = cap.read()  
    #rt,video=c.read()
  
    # Convert to grayscale  
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
  
    # Detect the faces  
    faces = haar_cascade.detectMultiScale(gray, 1.1, 4)  
    x,y,w,h=0,0,0,0
    bh,bw,bc=img.shape
    # Draw the rectangle around each face  
    for (x,y,w,h) in faces:
        
        faces_roi = gray[y:y+h,x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')

        cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        #crop= cv.rectangle(video, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    #out=cv.VideoWriter('output.avi',fourcc,5,(x+w,y+h))
   
   
        #c.set(x,x+w+20)
        #c.set(y,y+h+20)
        rt,video=c.read()
        h,w,oc=video.shape
        #if rt==True:
           # b=cv.resize(video,(x+w+20,y+h+20),x,y,interpolation=cv.INTER_CUBIC)
        w=800    #out.write(b)
        Is=w/bw
        nbh,nbw=int(bh*Is),int(bw*Is)
        nbg=cv.resize(img,(nbw,nbh))
        o=.7
        square=video[x:x+h,y:y+w]
        ai=cv.addWeighted(video,0.6,square,.4,0)
        cv.imshow("h",ai)
        
    #if rt==True: 
            
       # crop=video[x:x+w+20,y:y+h+20]
    #print(x,y)  
       # cv.imshow('Vide',b)  
   # cv.imshow('Video', img)  
   # cv.namedWindow('crop',cv.WND_PROP_FULLSCREEN)
    #cv.setWindowProperty('crop',cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
        
       
  
    # Display  
    #cv.imshow('Video', img)  
   # cv.imshow('s',crop)
  
    # Stop if escape key is pressed  
    k = cv.waitKey(30) & 0xff  
    if k==27:  
        break  
          
# Release the VideoCapture object  
out.release()
cap.release() 