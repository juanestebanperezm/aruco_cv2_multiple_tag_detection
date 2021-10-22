import numpy as np
import cv2
import cv2.aruco as aruco
import sys
import shutil

#Function to detect tags
def findArucoMarkers(img, markerSize = 5, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs) 
    return [bboxs,ids]

cap = cv2.VideoCapture('tags.mp4')

while True:
    
    orig_stdout=sys.stdout
    f=open('c2.txt','w')
    sys.stdout=f
            
    success, img = cap.read()
    arucofound = findArucoMarkers(img)
    if  len(arucofound[0])!=0:
        for bbox, id in zip(arucofound[0], arucofound[1]):
            #Capture the bbox cordinates and write them into a txt to file
            print(bbox[0][0][0],bbox[0][1][0],bbox[0][2][0],bbox[0][3][0],bbox[0][0][1],bbox[0][1][1],bbox[0][2][1],bbox[0][3][1])
        sys.stdout=orig_stdout
    
    #Write the BBOX coordinates
    file_=open('c2.txt','r')
    l=file_.readlines()
    if len([i for i in l])>=4:
        shutil.copyfile('c2.txt','c2_copy.txt')
        file_.close()

    #Read the txt coordinates copy file and extract 
    z=open('c2_copy.txt','r')
    new_file=z.readlines()
    c1= [i.strip() for i in new_file][0:1]
    c2= [i.strip() for i in new_file][1:2]
    c3= [i.strip() for i in new_file][2:3]
    c4= [i.strip() for i in new_file][3:4]
    c1=[float(i) for i in [i.split() for i in c1][0]]
    c2=[float(i) for i in [i.split() for i in c2][0]]
    c3=[float(i) for i in [i.split() for i in c3][0]]
    c4=[float(i) for i in [i.split() for i in c4][0]]
   
    cv2.circle(img,(int(sum(c1[0:4])/4),int(sum(c1[4:])/4)),7,(255,0,0),2)
    cv2.circle(img,(int(sum(c2[0:4])/4),int(sum(c2[4:])/4)),7,(255,0,0),2)
    cv2.circle(img,(int(sum(c3[0:4])/4),int(sum(c3[4:])/4)),7,(255,0,0),2)
    cv2.circle(img,(int(sum(c4[0:4])/4),int(sum(c4[4:])/4)),7,(255,0,0),2)

    #Array to apply the transformation, the parameters are the previous coordinates that we copy into the c2.txt

    pt1=np.float32([[int(sum(c1[0:4])/4),int(sum(c1[4:])/4)],[int(sum(c2[0:4])/4),int(sum(c2[4:])/4)],[int(sum(c3[0:4])/4),int(sum(c3[4:])/4)],[int(sum(c4[0:4])/4),int(sum(c4[4:])/4)]])
    pt2=np.float32([ [0,0],[480,0],[0,300],[480,300] ])
    tran=cv2.getPerspectiveTransform(pt1,pt2)
    d=cv2.warpPerspective(img,tran,(480,300))
    
    #Show the new video with apply transformation perspective
    cv2.imshow('img',d)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()

