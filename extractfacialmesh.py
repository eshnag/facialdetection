import os 
import numpy as np 
import pandas as pd 
import mediapipe as mp
import cv2
import math
import pickle
from tqdm import tqdm 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

class face_mesh_extract:
    def __init__(self,src_folder,dest_base_folder,img_base_folder,max_num_faces=1,refine_landmarks=True,fps=29.97,min_detection_confidence=0.5,min_tracking_confidence=0.9):

        self.src_folder=src_folder
        self.dest_base_folder=dest_base_folder
        self.img_base_folder=img_base_folder
        self.max_num_faces=max_num_faces
        self.refine_landmarks=refine_landmarks
        self.fps=fps
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence

        self.dest_sample_folder=os.path.join(self.dest_base_folder,self.src_folder.split("/")[-2])
        self.dest_sample_img_folder=os.path.join(self.img_base_folder,self.src_folder.split("/")[-2]+"_img")

        if not os.path.exists(self.dest_sample_folder):
            os.mkdir(self.dest_sample_folder)

        if not os.path.exists(self.dest_sample_img_folder):
            os.mkdir(self.dest_sample_img_folder)

        self.file_list=os.listdir(self.src_folder)

    def overlay_single_image(self,landmark_array,img):
        for i in np.arange(len(landmark_array)):    
            cv2.circle(img, (int(landmark_array[i][0]), int(landmark_array[i][1])), 1, (255,0,0),2)

        return(img)

    def obtain_landmark_points(self,results,width,height):

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

    def extract_facial_mesh(self):

        for i in tqdm(range(len(self.file_list))):

            #complete file path
            file_path=os.path.join(self.src_folder,self.file_list[i])

            #read using opencv 
            with mp_face_mesh.FaceMesh(max_num_faces=self.max_num_faces,refine_landmarks=self.refine_landmarks,min_detection_confidence=self.min_detection_confidence,min_tracking_confidence=self.min_tracking_confidence) as face_mesh:
                #print(file_path)
                image = cv2.imread(file_path)
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                height, width, _ = image.shape
                landmark_c=self.obtain_landmark_points(results,width,height)
                img=self.overlay_single_image(landmark_c,image)
                filename=os.path.join(self.dest_sample_img_folder,self.file_list[i])
                img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename,img)

            #save the landmark array
            landmark_file_path=os.path.join(self.dest_sample_folder,self.file_list[i].split(".")[0]+".npy")
            np.save(landmark_file_path,landmark_c)


#main 
if __name__=="__main__":
    src_folder="/data/paralysis/datasets/Toronto_NeuroFace/NeuroFace Open Access Data/Stroke/Frames"
    dest_base_folder="/data/paralysis/datasets/Toronto_NeuroFace/mesh_landmarks"
    image_base_folder="/data/paralysis/datasets/Toronto_NeuroFace/mesh_images"
    obj=face_mesh_extract(src_folder,dest_base_folder,image_base_folder)
    obj.extract_facial_mesh()



            