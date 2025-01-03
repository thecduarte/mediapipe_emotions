import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'project_root:', project_root)
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mediapipe as mp

from collections import Counter
from natsort import natsorted
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.data import Dataset, random_split
from tqdm import tqdm

from general_helpers.general_helper import GeneralHelper
from model import GNN

mp_face_mesh = mp.solutions.face_mesh
FACEMESH_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION
# print(f'FACEMESH_CONNECTIONS:', FACEMESH_CONNECTIONS)

class EmotionFacemeshDataset(Dataset):
    def facemesh_to_graph(self, facemesh_landmarks):
        # print(f'facemesh_landmarks: {facemesh_landmarks}')

        nodes = np.array([lm['position'] for lm in facemesh_landmarks])
        # print(f'nodes: {nodes}')
        # print(f'nodes shape: {nodes.shape}')

        nodes = torch.tensor(nodes, dtype=torch.float)

        edge_index = np.array(list(FACEMESH_CONNECTIONS), dtype=np.int64).T
        # print(f'edge_index: {edge_index}')
        # print(f'edge_index shape: {edge_index.shape}')

        edge_index = torch.tensor(edge_index, dtype=torch.long)

        return nodes, edge_index
    
    def construct_train_dataset(self, arousal_train_data, valence_train_data, expression_train_data, facemesh_train_data):
        arousal_labels, valence_labels, expression_labels = [], [], []
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
            
            arousal_label = float(arousal_elem['data'])
            valence_label = float(valence_elem['data'])
            expression_label = float(expression_elem['data'])

            # print(f'arousal_label: {arousal_label}')
            # print(f'valence_label: {valence_label}')
            # print(f'expression_label: {expression_label}')

            arousal_labels.append(arousal_label)
            valence_labels.append(valence_label)
            expression_labels.append(valence_label)

        min_expression_label = min(expression_labels)
        max_expression_label = max(expression_labels)

        label_frequency = Counter(expression_labels)

        for label, count in label_frequency.items():
            print(f'Num label -> {label} appears {count} times.')

        print(f'expression label min -> {min_expression_label} max -> {max_expression_label}')

        train_data = []
        for i in tqdm(range(len(facemesh_train_data)), desc='Constructing Training Dataset...'):
            facemesh_elem = facemesh_train_data[i]
            arousal_label = arousal_labels[i]
            valence_elem = valence_labels[i]
            expression_elem = expression_labels[i]

            # print(f'expression_elem: {expression_elem}')
            
            # Assumption: FaceMesh data instance contains only one face.
            facemesh_data = facemesh_elem['data']
            facemesh_num_faces = len(facemesh_data)

            # print(f'num faces: {facemesh_num_faces}')

            if facemesh_num_faces != 1:
                print(f'Facemesh data instance contains more than one face!')
                print(f'iter num: {i}')

            facemesh_landmarks = facemesh_data[0]['0']

            nodes, edge_index = self.facemesh_to_graph(facemesh_landmarks=facemesh_landmarks)

            # label_tensor = torch.tensor(expression_label, dtype=torch.long)
            # label_tensor = torch.tensor([expression_label], dtype=torch.float)

            label_tensor = torch.tensor(expression_elem, dtype=torch.long)

            # graph_data = Data(x=nodes, edge_index=edge_index, y=torch.tensor([arousal_label, valence_label, expression_label], dtype=torch.float))
            graph_data = Data(x=nodes, edge_index=edge_index, y=label_tensor)
            # print(f'graph_data: {graph_data}')

            train_data.append(graph_data)

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

        self.train_data_fp = os.path.join(self.imgs_annotate_dir, 'data.pt')

        if not os.path.isfile(self.train_data_fp):

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
            for i in tqdm(range(len(arousal_file_list)), desc='Formatting Arousal Data...'):
                elem = arousal_file_list[i]
                data = np.load(elem['path'])
                idx = elem['basename'].split('_')[0]
                arousal_train_data.append({'idx': idx, 'data': data})

            valence_train_data = []
            for i in tqdm(range(len(valence_file_list)), desc='Formatting Valence Data...'):
                elem = valence_file_list[i]
                data = np.load(elem['path'])
                idx = elem['basename'].split('_')[0]
                valence_train_data.append({'idx': idx, 'data': data})
            
            expression_train_data = []
            for i in tqdm(range(len(expression_file_list)), desc='Formatting Expression Data...'):
                elem = expression_file_list[i]
                data = np.load(elem['path'])
                idx = elem['basename'].split('_')[0]
                expression_train_data.append({'idx': idx, 'data': data})

            facemesh_train_data = []
            for i in tqdm(range(len(landmark_file_list)), desc='Formatting Facemesh Data...'):
                elem = landmark_file_list[i]
                data = self.general_helper.read_json_file(data_file_path=elem['path'])
                idx = elem['basename'].split('.')[0]
                facemesh_train_data.append({'idx': idx, 'data': data})

            print(f'len arousal_train_data: {len(arousal_train_data)}')
            print(f'len valence_train_data: {len(valence_train_data)}')
            print(f'len expression_train_data: {len(expression_train_data)}')
            print(f'len facemesh_train_data: {len(facemesh_train_data)}')

            print(f'Constructing training dataset!')

            self.train_data = self.construct_train_dataset(arousal_train_data=arousal_train_data,
                                                    valence_train_data=valence_train_data,
                                                    expression_train_data=expression_train_data,
                                                    facemesh_train_data=facemesh_train_data)
            
            # print(f'train_data: {self.train_data}')

            torch.save(self.train_data, self.train_data_fp)

            print(f'Training data saved at -> {self.train_data_fp}')
        else:
            self.train_data = torch.load(self.train_data_fp)

            print(f'Training data loaded from -> {self.train_data_fp}')

        # Train data elem
        if len(self.train_data) > 0:
            train_data_elem = self.train_data[0]
            print(f'train_data_elem: {train_data_elem}')
            print(f'train_data_elem_x: {train_data_elem.x}')

            print(f'len train_data: {len(self.train_data)}')

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]

class Trainer:
    def __init__(self, device, model, loader, criterion, optimizer, val_loader=None):
        self.device = device
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.best_val_loss = float('inf')  # Initialize best validation loss

    def evaluate(self, loader):
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch.y)
                epoch_loss += loss.item()

        mean_loss = epoch_loss / len(loader)
        return mean_loss

    def train(self, num_epochs, checkpoint_path='best_model.pth'):
        train_loss_history = []
        val_loss_history = []

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch in tqdm(self.loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(batch)

                loss = self.criterion(outputs, batch.y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            mean_epoch_loss = epoch_loss / len(self.loader)
            train_loss_history.append(mean_epoch_loss)

            if self.val_loader:
                val_loss = self.evaluate(self.val_loader)
                val_loss_history.append(val_loss)
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {mean_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save the model if the validation loss has improved
                if val_loss < self.best_val_loss:
                    print(f"Validation loss improved. Saving model...")
                    self.best_val_loss = val_loss
                    self.save_model(checkpoint_path)
            else:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {mean_epoch_loss:.4f}")

        return {"train_loss": train_loss_history, "val_loss": val_loss_history}

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"Model loaded from {path}")

# Now let's run everything with the GNN model.

if __name__ == '__main__':
    emotion_facemesh_dataset = EmotionFacemeshDataset()  # Assume this returns your dataset
    
    # 80 - 20 train-val split
    train_size = int(0.8 * len(emotion_facemesh_dataset))
    print(f'train_size: {train_size}')
    val_size = len(emotion_facemesh_dataset) - train_size
    print(f'val_size: {val_size}')
    
    train_dataset, val_dataset = random_split(emotion_facemesh_dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Define Trainer parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(input_dim=3, hidden_dim=256, label_dim=8).to(device)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    # Initialize Trainer
    trainer = Trainer(
        device=device,
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        val_loader=val_loader
    )

    # Train the model
    history = trainer.train(num_epochs=10000, checkpoint_path='best_model.pth')