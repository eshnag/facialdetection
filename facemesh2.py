import os 
import numpy as np 
import pandas as pd 
import mediapipe as mp
import cv2
import math
import pickle
from tqdm import tqdm 

#mediapipe declarations
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

#global vars
max_num_faces=1
refine_landmarks=True
fps=29.97
min_detection_confidence=0.5
min_tracking_confidence=0.9

def overlay_single_image(landmark_array,img):
    for i in np.arange(len(landmark_array)):    
        cv2.circle(img, (int(landmark_array[i][0]), int(landmark_array[i][1])), 4, (255,0,0),5)
    cv2.imwrite('temp.png', img)
    return(img)

def obtain_landmark_points(results,width,height):
    landmark_list=[]
    for face_landmarks in results.multi_face_landmarks:
        face_landmark_current=face_landmarks.landmark
        for i in np.arange(len(face_landmark_current)):
            x_normalized=face_landmark_current[i].x
            y_normalized=face_landmark_current[i].y
            xx_point=min(math.floor(x_normalized * width), width - 1)
            yy_point=min(math.floor(y_normalized * height), height - 1)
            landmark_list.append([xx_point,yy_point])

    landmark_list=np.array(landmark_list)
    return(landmark_list)

def extract_facial_mesh(file_list):
    for i in tqdm(range(len(file_list))):
        file_path=file_list[i]
        #read using opencv 
        with mp_face_mesh.FaceMesh(max_num_faces=max_num_faces,refine_landmarks=refine_landmarks,min_detection_confidence=min_detection_confidence,min_tracking_confidence=min_tracking_confidence) as face_mesh:
        #print(file_path)
            image = cv2.imread(file_path)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            height, width, _ = image.shape
            landmark_c=obtain_landmark_points(results,width,height)
            print(landmark_c)
            img=overlay_single_image(landmark_c,image)
            filename="temp.png"
            img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename,img)
IMAGE_FILES = ["/Users/eshnagupta/opencv_tests/Normal1/Normal1_1.jpg"]
extract_facial_mesh(IMAGE_FILES)