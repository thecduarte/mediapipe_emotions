# General
import numpy as np
import subprocess
import time

# Vision
import cv2

# Media
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceMesh:
    def select_device(self):
        output = subprocess.check_output(['v4l2-ctl', '--list-devices']).decode('utf-8')

        # DEBUG
        print(f'output: {output}')

        lines = output.strip().split('\n')

        lines = [line for line in lines if line]

        # DEBUG
        print(f'lines: {lines}')

        device_groups = {}

        current_group = None
        for line in lines:
            if line.startswith('\t'):
                line = line.strip('\t')

                if 'video' in line:
                    device_groups[current_group].append(line)

            else:
                line = line.strip(':')
                current_group = line
                device_groups[current_group] = []

            # DEBUG
            # print(f'current device groups: {device_groups}')
                

        enumerated_devices = list(enumerate(device_groups.items()))

        print('Available Devices:')
        for i, (k,v) in enumerated_devices:
            print(i, k, v)

        if len(enumerated_devices) == 0:
            print('No camera devices currently available!')
            return
        
        chosen_device = enumerated_devices[-1]
        
        # if not self.camera_type or self.camera_type == 'integrated':
        #     chosen_device = enumerated_devices[0]

        # # Preference to latest external device if connected. 
        # elif self.camera_type == 'external':
        #     chosen_device = enumerated_devices[-1]

        print(f'chosen_device: {chosen_device}')

        cam_index = int(chosen_device[1][1][0][-1])

        print(f'cam_index: {cam_index}')

        return cam_index
    
    def get_usb_img_frame(self):
        ret, frame = self.cap.read()
        # DEBUG
        # print(f'ret: {ret}')
        # print(f'frame: {frame}')

        if ret:
            # resized_frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_AREA)
            resized_frame = frame

            # DEBUG
            # print(f'resized_frame shape: {resized_frame.shape}')

            return resized_frame
        
    # Obtained from https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

        return annotated_image
    
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

        self.facemesh_detector.detect_async(mp_image,  time.time_ns() // 1_000_000)

        if self.det_result:
            print(f'det_result: {self.det_result}')

            self.draw_mp_landmarks()


    def view_img_frame(self):
        while True:
            self.current_frame = self.get_usb_img_frame()
            
            self.process_img_frame()

            cv2.imshow('usb_stream', self.current_frame)
        
            key = cv2.waitKey(1)

            # keys 'q'
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def save_result(self, result, unused_output_image, timestamp_ms): 
        self.det_result = result

    def __init__(self):
        # Ubuntu: Look for corresponding device using `v4l2-ctl --list-devices`. Leave at default 0 unless other camera desired.

        # Camera device select
        try:
            self.cam_index = self.select_device()
        except Exception as e:
            print(f'Exception: {e}')
            print('\nCheck if you have a camera plugged in and try again.')
            return
        
        # Initialize OpenCV Video Capture
        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)

        if not self.cap.isOpened():
            print(f'Cannot open cam index: {self.cam_index}')
            exit()

        self.cam_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # DEBUG
        print(f'starting cam fps: {self.cam_fps}')

        # Initialize MediaPipe FaceMesh
        # Designate face landmarker model
        # model = 'face_landmarker.task'
        model = 'face_landmarker_v2_with_blendshapes.task'
        self.det_result = None
        
        # Generate FaceLandmarker object.
        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self.save_result
        )
        self.facemesh_detector = vision.FaceLandmarker.create_from_options(options)

        self.view_img_frame()

if __name__ == '__main__':
    facemesh = FaceMesh()