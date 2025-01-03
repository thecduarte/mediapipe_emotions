import cv2
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np

from PIL import Image

from natsort import natsorted

# Media
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'project_root:', project_root)
sys.path.append(project_root)

from general_helpers.general_helper import GeneralHelper
from model import GNN

mp_face_mesh = mp.solutions.face_mesh
FACEMESH_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION

class Tester:
    def __init__(self):
        self.general_helper = GeneralHelper()
        self.device = None
        self.model = None

        face_landmarker_model = os.path.join(project_root, 'face_landmarker_v2_with_blendshapes.task')
        self.det_result = None
        
        # Generate FaceLandmarker object.
        base_options = python.BaseOptions(model_asset_path=face_landmarker_model)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.facemesh_detector = vision.FaceLandmarker.create_from_options(options)

        self.sample_imgs_path = 'start_imgs'
        self.general_helper.dir_create(self.sample_imgs_path)

        self.img_files = natsorted(self.general_helper.recursive_get_file_list(dir_path=self.sample_imgs_path))
        # print(f'imgs_files: {self.img_files}')

        # self.img_files = [self.img_files[0]]
        self.img_files = self.img_files[0:100]
        print(f'imgs_files: {self.img_files}')

        self.load_model()
        self.predict(img_file_list=self.img_files)

    def load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GNN(input_dim=3, hidden_dim=128, label_dim=7).to(self.device)
        self.model.load_state_dict(torch.load('best_model.pth', map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def process_single_img(self, elem, debug=False):
        img_path = elem['path']
        print(f'current_path:', img_path)

        file_index = elem['basename'].split('.')[0]
        print(f'file_index: {file_index}')

        annotate_img_fp = os.path.join(f'{self.annotate_img_path}', f"{file_index}.jpg")
        print(f'current annotate img fp:', annotate_img_fp)

        annotate_json_fp = os.path.join(f'{self.annotate_json_path}', f"{file_index}.json")
        print(f'current annotate json fp:', annotate_json_fp)

        img = Image.open(img_path)
        img = np.array(img)

        # Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python#code_example
        # Run face landmarker using model.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        det_result = self.facemesh_detector.detect(mp_img)

        print(f'det_result: {det_result}')

        if det_result is not None:
            # print(f'det_result: {det_result}')

            det_num_faces = det_result.face_landmarks

            print(f'num_det_faces: {len(det_num_faces)}')

            # There must be at least one face in the current frame.
            if len(det_num_faces) != 1:
                return

            for landmarks in det_num_faces:
                # print('num_landmarks:', len(landmarks))

                # There must be a complete set of landmarks to each face.
                if len(landmarks) != 478:
                    return

            img = self.draw_mp_landmarks(img=img, det_result=det_result)

            # self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)

            if debug:
                self.view_single_img(img=img)

            self.save_facemesh_img(img=img, annotate_img_fp=annotate_img_fp)

            self.save_facemesh_json(det_result=det_result, annotate_json_fp=annotate_json_fp)

    def predict(self, img_file_list):
        for elem in img_file_list:
            img_path = elem['path']
            print(f'current_path:', img_path)

            img = Image.open(img_path)
            img = np.array(img)

            fn = elem['basename'].split('.')[0]
            exp_fp = os.path.join('imgs_annotate', 'npy', 'exp', f'{fn}_exp.npy')
            print(f'exp_fp: {exp_fp}')

            # Run face landmarker using model.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

            facemesh_data = self.facemesh_detector.detect(mp_img)
            # print(f'facemesh_data: {facemesh_data}')

            for face_idx, landmarks in enumerate(facemesh_data.face_landmarks):
                # print(f'face_idx: {face_idx}\nlandmarks: {landmarks}')

                nodes = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                print(f'nodes: {nodes}')
                print(f'nodes shape: {nodes.shape}')

                nodes = torch.tensor(nodes, dtype=torch.float)

                edge_index = np.array(list(FACEMESH_CONNECTIONS), dtype=np.int64).T
                print(f'edge_index: {edge_index}')
                print(f'edge_index shape: {edge_index.shape}')

                edge_index = torch.tensor(edge_index, dtype=torch.long)

                graph_data = Data(x=nodes, edge_index=edge_index)
                graph_data = graph_data.to(self.device)

                with torch.no_grad():
                    prediction = self.model(graph_data)
                    print(f'model prediction: {prediction}')

                    probabilities = F.softmax(prediction, dim=1)
                    print(f'probabilities: {probabilities}')

                    predicted_class = torch.argmax(prediction, dim=1).item()
                    print(f'predicted class: {predicted_class}')

                    actual_class = np.load(exp_fp)
                    print(f'actual_class: {actual_class}')

if __name__ == '__main__':
    tester = Tester()
    

    