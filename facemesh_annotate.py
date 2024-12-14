# General
import os
import numpy as np
import subprocess
import time

# Vision
import cv2
from PIL import Image

# Media
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Custom
from general_helpers.general_helper import GeneralHelper

class FaceMeshAnnotate:
    def draw_mp_landmarks(self):
        # Draw landmarks.
        for face_landmarks in self.det_result.face_landmarks:
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x,
                                                y=landmark.y,
                                                z=landmark.z) for
                landmark in
                face_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                image=self.current_frame,
                landmark_list=face_landmarks_proto,
                connections=solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=self.current_frame,
                landmark_list=face_landmarks_proto,
                connections=solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=self.current_frame,
                landmark_list=face_landmarks_proto,
                connections=solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())
   
    def process_img_frame(self):
        # Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python#code_example
        
        # Run face landmarker using model.
        
        image = self.current_frame

        rgb_image = image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        self.det_result = self.facemesh_detector.detect(mp_image)

        # print(f'det_result: {self.det_result}')

        if self.det_result:
            # print(f'det_result: {self.det_result}')

            self.draw_mp_landmarks()

            self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)


    def view_img_frame(self):
        print(f'view img frame!')
        # print(f'self.det_result:', self.det_result)
        # print(f'type self.det_result:', type(self.det_result))

        # If detection results exist
        if self.det_result:
            while True:
                # Display the frame using OpenCV

                cv2.imshow("Face Mesh Landmarks", self.current_frame)

                # Wait for a key press (press 'q' to quit)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting display...")
                    cv2.destroyAllWindows()
                    return

    def save_result(self, result, unused_output_image, timestamp_ms): 
        self.det_result = result

    def save_facemesh_img(self):
        print(f'IMG save!')
        cv2.imwrite(self.current_annotate_img_fp, self.current_frame)

    def save_facemesh_json(self):
        print(f'JSON save!')
        # View detection _results
        # print(self.det_result.__dict__)

        all_faces_data = []
        for face_landmarks in enumerate(self.det_result.face_landmarks):
            face_data_dict = {}

            face_data_dict[f'{face_landmarks[0]}'] = []

            for landmark in enumerate(face_landmarks[1]):
                landmark_data_dict = {
                    'index': landmark[0],
                    'position': [landmark[1].x, landmark[1].y, landmark[1].z],
                    'visibility': landmark[1].visibility,
                    'presence': landmark[1].presence
                }

                face_data_dict[f'{face_landmarks[0]}'].append(landmark_data_dict)

            all_faces_data.append(face_data_dict)

        # print(f'all_faces_dict:', all_faces_data)

        self.general_helper.write_json_file(data_dict=all_faces_data, data_file_path=self.current_annotate_json_fp)

    def __init__(self):
        self.general_helper = GeneralHelper()

        self.imgs_path = 'imgs'

        self.annotate_img_path = os.path.join(f'{self.imgs_path}_facemesh_annotate', 'img')

        if not os.path.exists(self.annotate_img_path):
            self.general_helper.dir_create(self.annotate_img_path)

        self.annotate_json_path = os.path.join(f'{self.imgs_path}_facemesh_annotate', 'json')

        if not os.path.exists(self.annotate_json_path):
            self.general_helper.dir_create(self.annotate_json_path)

        self.img_files = self.general_helper.recursive_get_file_list(dir_path='imgs')

        print(f'img_files: {self.img_files}')

        # Initialize MediaPipe FaceMesh
        # Designate face landmarker model
        # model = 'face_landmarker.task'
        model = 'face_landmarker_v2_with_blendshapes.task'
        self.det_result = None
        
        # Generate FaceLandmarker object.
        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.facemesh_detector = vision.FaceLandmarker.create_from_options(options)

        for elem in self.img_files:
            self.current_img_path = elem['path']
            print(f'current_path:', self.current_img_path)

            self.current_annotate_img_fp = os.path.join(f'{self.annotate_img_path}', f"{elem['basename'].split('.')[0]}.jpg")
            print(f'current annotate img fp:', self.current_annotate_img_fp)

            self.current_annotate_json_fp = os.path.join(f'{self.annotate_json_path}', f"{elem['basename'].split('.')[0]}.json")
            print(f'current annotate json fp:', self.current_annotate_json_fp)

            img = Image.open(self.current_img_path)
            img = np.array(img)

            # print(f'img np:', img)

            self.current_frame = img

            self.process_img_frame()

            self.save_facemesh_img()

            self.save_facemesh_json()

            # self.view_img_frame()


if __name__ == '__main__':
    facemesh = FaceMeshAnnotate()