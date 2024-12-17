# General
import multiprocessing
import os
import numpy as np
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

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
    def draw_mp_landmarks(self, img, det_result):
        # Draw landmarks.
        for face_landmarks in det_result.face_landmarks:
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x,
                                                y=landmark.y,
                                                z=landmark.z) for
                landmark in
                face_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                image=img,
                landmark_list=face_landmarks_proto,
                connections=solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=img,
                landmark_list=face_landmarks_proto,
                connections=solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=img,
                landmark_list=face_landmarks_proto,
                connections=solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())
            
        return img

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

    def view_single_img(self, img):
        while True:
            cv2.imshow('FaceMesh Annotate', img)
        
            key = cv2.waitKey(1)

            # keys 'q'
            if key == ord('q'):
                break

    def save_result(self, result, unused_output_image, timestamp_ms): 
        self.det_result = result

    def save_facemesh_img(self, img, annotate_img_fp):
        print(f'IMG save! @ {annotate_img_fp}')
        cv2.imwrite(annotate_img_fp, img)

    def save_facemesh_json(self, det_result, annotate_json_fp):
        print(f'JSON save! @ {annotate_json_fp}')
        # View detection _results
        # print(self.det_result.__dict__)

        all_faces_data = []
        for face_idx, landmarks in enumerate(det_result.face_landmarks):
            face_data_dict = {}

            face_data_dict[f'{face_idx}'] = []

            for landmark_idx, landmark in enumerate(landmarks):
                landmark_data_dict = {
                    'index': landmark_idx,
                    'position': [landmark.x, landmark.y, landmark.z],
                    'visibility': landmark.visibility,
                    'presence': landmark.presence
                }

                face_data_dict[f'{face_idx}'].append(landmark_data_dict)

            all_faces_data.append(face_data_dict)

        # print(f'all_faces_dict:', all_faces_data)

        self.general_helper.write_json_file(data_dict=all_faces_data, data_file_path=annotate_json_fp)

    def process_imgs_in_parallel(self):
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
            executor.map(self.process_single_img, self.img_files)

    def copy_files(self, src_dir, dst_dir, valid_files):
        if not os.path.exists(src_dir):
            print('src_dir does not exist!')
            return
        
        if not os.path.exists(dst_dir):
            print('dst_dir does not exist!')
            return
        
        for elem in valid_files:
            src_fp = os.path.join(src_dir, elem)
            dst_fp = os.path.join(dst_dir, elem)

            try:
                if os.path.isfile(src_fp):
                    shutil.copy(src_fp, dst_fp)
                    print(f"Copied file from '{src_fp} to '{dst_fp}'!")

            except Exception as e:
                print(f"Unable to copy file from '{src_fp}' to '{dst_fp}'! Exception: {e}")
                return

            # file_index = elem['basename'].split('.')[0]

            # aro_filename = f'{file_index}_aro.npy'
            # exp_filename = f'{file_index}_exp.npy'
            # lnd_filename = f'{file_index}_lnd.npy'

            # src_aro_fp = os.path.join(src_dir, aro_filename)
            # src_exp_fp = os.path.join(src_dir, exp_filename)
            # src_lnd_fp = os.path.join(src_dir, lnd_filename)

            # dst_aro_fp = os.path.join(dst_dir, aro_filename)
            # # self.general_helper.dir_create(dir_path=)

            # dst_exp_fp = os.path.join(dst_dir, exp_filename)
            # dst_lnd_fp = os.path.join(dst_dir, lnd_filename)

            # if os.path.isfile(src_aro_fp):
            #     shutil.copy(src_aro_fp, dst_aro_fp)
            #     print(f"Copied file from '{src_aro_fp} to '{dst_aro_fp}'!")

            # if os.path.isfile(src_exp_fp):
            #     shutil.copy(src_exp_fp, dst_exp_fp)
            #     print(f"Copied file from '{src_aro_fp} to '{dst_aro_fp}'!")
            
            # if os.path.isfile(src_lnd_fp):
            #     shutil.copy(src_lnd_fp, dst_lnd_fp)
            #     print(f"Copied file from '{src_aro_fp} to '{dst_aro_fp}'!")
           
        print(f'Copied all valid files from {src_dir} to {dst_dir}!')

    def __init__(self):
        self.general_helper = GeneralHelper()

        self.train_path = 'train'
        self.general_helper.dir_create(self.train_path)

        self.start_imgs_path = os.path.join(self.train_path, 'start_imgs')
        self.general_helper.dir_create(self.start_imgs_path)

        self.start_npy_path = os.path.join(self.train_path, 'start_npy')
        self.general_helper.dir_create(self.start_npy_path)

        self.imgs_annotation_path = os.path.join(self.train_path, 'imgs_annotate')
        self.general_helper.dir_remove(self.imgs_annotation_path)
        self.general_helper.dir_create(self.imgs_annotation_path)

        self.annotate_img_path = os.path.join(self.imgs_annotation_path, 'img')
        self.general_helper.dir_create(self.annotate_img_path)

        self.annotate_json_path = os.path.join(self.imgs_annotation_path, 'json')
        self.general_helper.dir_create(self.annotate_json_path)

        self.annotate_npy_path = os.path.join(self.imgs_annotation_path, 'npy')
        self.general_helper.dir_create(self.annotate_npy_path)

        self.annotate_aro_path = os.path.join(self.annotate_npy_path, 'aro')
        self.general_helper.dir_create(self.annotate_aro_path)

        self.annotate_exp_path = os.path.join(self.annotate_npy_path, 'exp')
        self.general_helper.dir_create(self.annotate_exp_path)

        self.annotate_lnd_path = os.path.join(self.annotate_npy_path, 'lnd')
        self.general_helper.dir_create(self.annotate_lnd_path)
        
        self.img_files = self.general_helper.recursive_get_file_list(dir_path=self.start_imgs_path)

        # self.general_helper.pprint_iterable(iterable=self.img_files)

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

        start_time = time.time()

        # Single Thread
        for elem in self.img_files:
            self.process_single_img(elem=elem)

        # Multiple Threads
        # self.process_imgs_in_parallel()

        print(f'num_processed_files: {len(self.img_files)}')

        self.valid_img_files = self.general_helper.recursive_get_file_list(dir_path=self.annotate_img_path)
        print(f'num valid_imgs_files: {len(self.valid_img_files)}')

        self.valid_files_idxs = [elem['basename'].split('.')[0] for elem in self.valid_img_files]
        print(f'self.valid_files_idxs: {self.valid_files_idxs}')

        npy_files = self.general_helper.recursive_get_file_list(dir_path=self.start_npy_path)
        print(f'npy_files: {npy_files}')

        ano_npy_files = [elem['basename'] for elem in npy_files if (elem['basename'].split('_')[0] in self.valid_files_idxs and elem['basename'].split('_')[1] == 'aro.npy')]
        print(f'ano_npy_files: {ano_npy_files}')

        exp_npy_files = [elem['basename'] for elem in npy_files if (elem['basename'].split('_')[0] in self.valid_files_idxs and elem['basename'].split('_')[1] == 'exp.npy')]
        print(f'exp_npy_files: {exp_npy_files}')

        lnd_npy_files = [elem['basename'] for elem in npy_files if (elem['basename'].split('_')[0] in self.valid_files_idxs and elem['basename'].split('_')[1] == 'lnd.npy')]
        print(f'lnd_npy_files: {lnd_npy_files}')

        self.copy_files(src_dir=self.start_npy_path, dst_dir=self.annotate_aro_path, valid_files=ano_npy_files)
        self.copy_files(src_dir=self.start_npy_path, dst_dir=self.annotate_exp_path, valid_files=exp_npy_files)
        self.copy_files(src_dir=self.start_npy_path, dst_dir=self.annotate_lnd_path, valid_files=lnd_npy_files)

        end_time = time.time() - start_time
        print(f'end_time: {end_time}')

if __name__ == '__main__':
    facemesh = FaceMeshAnnotate()