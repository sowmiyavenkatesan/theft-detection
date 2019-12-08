import cv2,time
import pandas
from datetime import datetime
first_frame=None
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Start","End"])#to store wen obj movement occurs
video=cv2.VideoCapture(0)
flag=0
while True:
    check,frame=video.read()
    status=0#initially status is 0
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#convert to gray scale 
    gray=cv2.GaussianBlur(gray,(21,21),0)#convert to blur
    
    if first_frame is None:
        first_frame=gray#store the first frame
        continue
    delta_frame=cv2.absdiff(first_frame,gray)#calculates the diff
    thresh_delta=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]#provides threshold wen val <30 it converts to black else to white
    thresh_delta=cv2.dilate(thresh_delta,None,iterations=0)#provides threshold wen val <30 it converts to black else to white
    (_,cnts,_)=cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in cnts:
        if cv2.contourArea(contour)<1000:#removes noise
            continue
        status=1#changes the status wen obj is detected
        if(flag==0):
            
            s=str(datetime.now())
            flag=0.1
        if(flag==0.1):
            firsts_frame=frame
            flag=1
        if(flag==1):
            s1=str(datetime.now())
        
        (x,y,w,h)=cv2.boundingRect(contour)#creates rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    status_list.append(status)
    status_list=status_list[-2:]
    if(status_list[-1]==1 and status_list[-2]==0):#records time wen change occurs
        times.append(datetime.now())
    if(status_list[-1]==0 and status_list[-2]==1):
        times.append(datetime.now())
    cv2.imshow('frame',frame)
    cv2.imshow('capturing',gray)
    cv2.imshow('delta',delta_frame)
    cv2.imshow('thresh',thresh_delta)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("detection captured during time:",s,"to",s1)
cv2.imshow("detected obj",firsts_frame)
video.release()
cv2.destroyAllWindows
