import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'project_root:', project_root)
sys.path.append(project_root)

import torch
import numpy as np
import mediapipe as mp

from natsort import natsorted
from torch_geometric.data import Data

from general_helpers.general_helper import GeneralHelper

mp_face_mesh = mp.solutions.face_mesh
FACEMESH_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION
# print(f'FACEMESH_CONNECTIONS:', FACEMESH_CONNECTIONS)

class EmotionGNN:
    def facemesh_to_graph(self, facemesh_landmarks):
        # print(f'facemesh_landmarks: {facemesh_landmarks}')

        nodes = np.array([lm['position'] for lm in facemesh_landmarks])
        print(f'nodes: {nodes}')
        print(f'nodes shape: {nodes.shape}')

        nodes = torch.tensor(nodes, dtype=torch.float)

        edge_index = np.array(list(FACEMESH_CONNECTIONS), dtype=np.int64).T
        print(f'edge_index: {edge_index}')
        print(f'edge_index shape: {edge_index.shape}')

        edge_index = torch.tensor(edge_index, dtype=torch.long)

        return nodes, edge_index
    
    def construct_train_dataset(self, arousal_train_data, valence_train_data, expression_train_data, facemesh_train_data): 
        train_data = []
        for i in range(len(facemesh_train_data)):
            facemesh_elem = facemesh_train_data[i]
            arousal_elem = arousal_train_data[i]
            valence_elem = valence_train_data[i]
            expression_elem = expression_train_data[i]

            # print(f'iter num: {i}')
            # print(f"facemesh_elem: {facemesh_elem}")
            # print(f"arousal_elem: {arousal_elem}")
            # print(f"valence_elem: {valence_elem}")
            # print(f"expression_elem: {expression_elem}")

            if not (facemesh_elem['idx'] == arousal_elem['idx'] == valence_elem['idx'] == expression_elem['idx']):
                print(f'Data elem index mismatch please verify!')

                print(f'iter num: {i}')
                print(f"facemesh_elem idx: {facemesh_elem['idx']}")
                print(f"arousal_elem idx: {arousal_elem['idx']}")
                print(f"valence_elem idx: {valence_elem['idx']}")
                print(f"expression_elem idx: {expression_elem['idx']}")

                return
            
            # Assumption: FaceMesh data instance contains only one face.
            facemesh_data = facemesh_elem['data']
            facemesh_num_faces = len(facemesh_data)

            print(f'num faces: {facemesh_num_faces}')

            if facemesh_num_faces != 1:
                print(f'Facemesh data instance contains more than one face!')
                print(f'iter num: {i}')

            facemesh_landmarks = facemesh_data[0]['0']

            nodes, edge_index = self.facemesh_to_graph(facemesh_landmarks=facemesh_landmarks)

            arousal_label = float(arousal_elem['data'])
            valence_label = float(valence_elem['data'])
            expression_label = float(expression_elem['data'])

            print(f'arousal_label: {arousal_label}')
            print(f'valence_label: {valence_label}')
            print(f'expression_label: {expression_label}')

            graph_data = Data(x=nodes, edge_index=edge_index, y=torch.tensor([arousal_label, valence_label, expression_label], dtype=torch.float))
            print(f'graph_data: {graph_data}')

            train_data.append(graph_data)

            # print(f'facemesh_data: {facemesh_data}')
            
            # nodes, edge_index = self.facemesh_to_graph()

        return train_data

    def __init__(self):
        self.general_helper = GeneralHelper()

        self.train_dir = os.path.join(project_root, 'train')
        self.imgs_annotate_dir = os.path.join(self.train_dir, 'imgs_annotate')
        self.json_dir = os.path.join(self.imgs_annotate_dir, 'json')
        # print(f'self.json_dir:', self.json_dir)

        self.npy_dir = os.path.join(self.imgs_annotate_dir, 'npy')
        self.arousal_dir = os.path.join(self.npy_dir, 'aro')
        self.valence_dir = os.path.join(self.npy_dir, 'exp')
        self.lnd_dir = os.path.join(self.npy_dir, 'lnd')

        # print(f'self.arousal_dir: {self.arousal_dir}')

        # landmark_file_list = sorted(self.general_helper.recursive_get_file_list(dir_path=self.json_dir), key=lambda x:x['basename'])
        # arousal_file_list = sorted(self.general_helper.recursive_get_file_list(dir_path=self.arousal_dir), key=lambda x:x['basename'])
        # valence_file_list = sorted(self.general_helper.recursive_get_file_list(dir_path=self.valence_dir), key=lambda x:x['basename'])
        # expression_file_list = sorted(self.general_helper.recursive_get_file_list(dir_path=self.valence_dir), key=lambda x: x['basename'])

        landmark_file_list = natsorted(self.general_helper.recursive_get_file_list(dir_path=self.json_dir), key=lambda x:x['basename'])
        arousal_file_list = natsorted(self.general_helper.recursive_get_file_list(dir_path=self.arousal_dir), key=lambda x:x['basename'])
        valence_file_list = natsorted(self.general_helper.recursive_get_file_list(dir_path=self.valence_dir), key=lambda x:x['basename'])
        expression_file_list = natsorted(self.general_helper.recursive_get_file_list(dir_path=self.valence_dir), key=lambda x:x['basename'])
        
        # print(f'landmark_file_list:', landmark_file_list)
        # print(f'arousal_file_list:', arousal_file_list)
        # print(f'valence_file_list: {valence_file_list}')
        # print(f'land_file_list: {land_file_list}')

        print(f'len landmark_file_list: {len(landmark_file_list)}')
        print(f'len arousal_file_list: {len(arousal_file_list)}')
        print(f'len valence_file_list: {len(valence_file_list)}')
        print(f'len expression_file_list: {len(expression_file_list)}')

        arousal_train_data = []
        for elem in arousal_file_list:
            data = np.load(elem['path'])
            idx = elem['basename'].split('_')[0]
            arousal_train_data.append({'idx': idx, 'data': data})

        valence_train_data = []
        for elem in valence_file_list:
            data = np.load(elem['path'])
            idx = elem['basename'].split('_')[0]
            valence_train_data.append({'idx': idx, 'data': data})
        
        expression_train_data = []
        for elem in expression_file_list:
            data = np.load(elem['path'])
            idx = elem['basename'].split('_')[0]
            expression_train_data.append({'idx': idx, 'data': data})

        facemesh_train_data = []
        for elem in landmark_file_list:
            data = self.general_helper.read_json_file(data_file_path=elem['path'])
            idx = elem['basename'].split('.')[0]
            facemesh_train_data.append({'idx': idx, 'data': data})

        train_data = self.construct_train_dataset(arousal_train_data=arousal_train_data,
                                                  valence_train_data=valence_train_data,
                                                  expression_train_data=expression_train_data,
                                                  facemesh_train_data=facemesh_train_data)
        
        print(f'train_data: {train_data}')
        print(f'len train_data: {len(train_data)}')

        # for elem in landmark_file_list:
        #     path = elem['path']

        #     json_data = self.general_helper.read_json_file(data_file_path=path)
        #     # print(f'json_data:', json_data)

        

        # print(f'lnd_train_data: {lnd_train_data}')
        # print(f'lnd_train_dta')

if __name__ == '__main__':
    emotion_gnn = EmotionGNN()