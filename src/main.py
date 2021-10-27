import numpy as np
import cv2
from cv2 import aruco
import yaml
import pandas as pd


file_='calibration_matrix.yaml'
#Function to detect tags
def findArucoMarkers(img, markerSize = str('5'), totalMarkers=str('250'), draw=True):
    #x=yaml.safe_load(open("src/calibration_matrix.yaml",'r'))
    gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    if markerSize=='APRILTAG':
        key = getattr(aruco, f'DICT_{markerSize}_{totalMarkers}')
    else:
        key=getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    corners, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    #,cameraMatrix=np.array([[646.7676471528549, 0.0, 318.93014949990015], [0.0, 647.3042988053412, 244.93412329482263], [0.0, 0.0, 1.0]]) ,distCoeff=np.array([[0.04953697129350694, -0.4206829889465348, 0.0017941220198914778, -0.006434768303329634, 1.6399056290328016]])
    if draw:
        aruco.drawDetectedMarkers(img_, corners) 
    return [corners,ids]

cap = cv2.VideoCapture('D:\\programacion\\aruco_cv2\\src\\tags.mp4')

while True:

    success, img_ = cap.read()
    callFunction=findArucoMarkers(img_)
    corners=callFunction[0]
    ids=callFunction[1]

    cor = {
        'refPt1': '',
        'refPt2': '',
        'refPt3': '',
        'refPt4': '',
    }

    try:
        index1 = np.squeeze(np.where(ids==1))
        x1 = [np.squeeze(corners[index1[0]])[0], 
            np.squeeze(corners[index1[0]])[1],
            np.squeeze(corners[index1[0]])[2],
            np.squeeze(corners[index1[0]])[3]]
        
        cor['refPt1'] = '' if len(x1)<4 else sum(x1[0:4])/4
        
        index2 = np.squeeze(np.where(ids==2))
        x2=[
        np.squeeze(corners[index2[0]])[0], 
        np.squeeze(corners[index2[0]])[1],
        np.squeeze(corners[index2[0]])[2],
        np.squeeze(corners[index2[0]])[3]]
        
        cor['refPt2'] = '' if len(x2)<4 else sum(x2[0:4])/4

        index3 = np.squeeze(np.where(ids==3))
        x3 = [np.squeeze(corners[index3[0]])[0], 
            np.squeeze(corners[index3[0]])[1],
            np.squeeze(corners[index3[0]])[2],
            np.squeeze(corners[index3[0]])[3]]
        
        cor['refPt3'] = '' if len(x3)<4 else sum(x3[0:4])/4
        
        index4 = np.squeeze(np.where(ids==4))
        x4=[
        np.squeeze(corners[index4[0]])[0], 
        np.squeeze(corners[index4[0]])[1],
        np.squeeze(corners[index4[0]])[2],
        np.squeeze(corners[index4[0]])[3]]
        
        cor['refPt4'] = '' if len(x4)<4 else sum(x4[0:4])/4

    except:
        print('Points need to be in the scene...')

    print(cor)

    try:
        pt1=np.float32([cor['refPt1'], cor['refPt2'], cor['refPt3'], cor['refPt4']])
        pt2=np.float32([ [0,0],[480,0],[0,300],[480,300] ])
        tran=cv2.getPerspectiveTransform(pt1,pt2)
        d=cv2.warpPerspective(img,tran,(480,300))
        #Show the new video with apply transformation perspective
        cv2.imshow('img',img)
        cv2.imshow('trass',d)
    except:
        print('coordinates are empty')

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
