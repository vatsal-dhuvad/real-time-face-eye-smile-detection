import os
import cv2

face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')) # load the face classifier from the local project folder
eye_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')) # load the eye classifier from the local project folder
smile_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_smile.xml'))# load the smile classifier from the local project folder

webcam = cv2.VideoCapture(0) # 0 for default camera, 1 for external camera

while True:
    _, img = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Scan frame for faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # 1.3 = scale factor, 5 = minimum neighbors
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) # rectangle parameters: image, top-left corner, bottom-right corner, color (BGR), thickness
        cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Crop the face area to search only inside it
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22) # Scan inside the face to locate the eyes

        for (ex, ey, ew, eh) in eyes: 
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)# Draw Blue rectangle over eye
            cv2.putText(roi_color, "Eye", (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22) # Scan inside the face to locate the smile
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2) # Draw Red rectangle over smile expression
            cv2.putText(roi_color, "Smile", (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imshow("Multi-Detection System", img) # Show the live video 
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # q key to exit the loop
        break

webcam.release()
cv2.destroyAllWindows()