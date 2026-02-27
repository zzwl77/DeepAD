import os
import pandas as pd
import PIL.Image as Image
from torch.utils.data import Dataset
import torch
import h5py

class AD2_Dataloader(Dataset):
    def __init__(self, root, data_list, label, transform=None, hdf5_file=None):  
        self.root = root
        self.data = data_list 
        self.label = label
        self.transform = transform
        self.hdf5_file = hdf5_file or os.path.join(root, 'ad.hdf5')
        self.gazeheatmap = ["presaccade1_l.png", "presaccade1_r.png", "presaccade2_l.png", "presaccade2_r.png", 
                            "antisaccade1_l.png", "antisaccade1_r.png", "antisaccade2_l.png", "antisaccade2_r.png", 
                            "sensitivity1_l.png", "sensitivity1_r.png", "sensitivity2_l.png", "sensitivity2_r.png", 
                            "sensitivity3_l.png", "sensitivity3_r.png", "saliency1_l.png", "saliency1_r.png", 
                            "saliency2_l.png", "saliency2_r.png", "saliency3_l.png", "saliency3_r.png", 
                            "saliency4_l.png", "saliency4_r.png", "saliency5_l.png", "saliency5_r.png"]
        if not os.path.exists(self.hdf5_file):
            self.create_hdf5(self.hdf5_file, self.data)
        self.hdf5_data = h5py.File(self.hdf5_file, 'r')
        self.taskmaps = [torch.tensor(self.hdf5_data['taskmaps'][str(i)][()], dtype=torch.float32) for i in range(1, 13)]
    
    def create_hdf5(self, file_path, data):
        with h5py.File(file_path, 'w') as file:
            # Create datasets for images and labels
            gaze_images = file.create_group('gaze_images')
            taskmaps = file.create_group('taskmaps')

            # Save taskmaps
            for i in range(1, 13):
                img_path = os.path.join(self.root, 'taskmaps', f'{i}.png')
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                taskmaps.create_dataset(str(i), data=img.numpy())

            # Iterate over rows in the DataFrame to save gaze heatmap images
            for index, row in data.iterrows():
                gazeheat_path = os.path.join(self.root, row.iloc[0], 'gazeheat')
                for idx, gh in enumerate(self.gazeheatmap):
                    img_path = os.path.join(gazeheat_path, gh)
                    img = Image.open(img_path)
                    if self.transform:
                        img = self.transform(img)
                    gaze_images.create_dataset(f'{index}_{idx}', data=img.numpy())

            # Save other data
            file.create_dataset('image_paths', data=data.iloc[:, 0].astype('S'))  # 存储图像路径
            file.create_dataset('mmse', data=data.iloc[:, 1].to_numpy(), dtype='f')
            file.create_dataset('moca', data=data.iloc[:, 2].to_numpy(), dtype='f')
            file.create_dataset('other', data=data.iloc[:, 3:8].to_numpy(), dtype='f')
            file.create_dataset('age_edu', data=data.iloc[:, 8:10].to_numpy(), dtype='f')
            file.create_dataset('icls', data=data.iloc[:, 11].to_numpy(), dtype='f')
    
    def __getitem__(self, idx):
        mmse = torch.tensor(self.hdf5_data['mmse'][idx], dtype=torch.float32)
        moca = torch.tensor(self.hdf5_data['moca'][idx], dtype=torch.float32)
        other = torch.tensor(self.hdf5_data['other'][idx], dtype=torch.float32)
        age_edu = torch.tensor(self.hdf5_data['age_edu'][idx], dtype=torch.float32)
        icls = torch.tensor(self.hdf5_data['icls'][idx], dtype=torch.float32)

        if self.label == 'mmse':
            target = mmse
        elif self.label == 'moca':
            target = moca
        elif self.label == 'demo':
            target = mmse  # Demo 也使用 MMSE 作为目标
        elif self.label == 'cls':
            target = icls
        else:
            raise ValueError('Invalid label specified')
        imgs = [torch.tensor(self.hdf5_data['gaze_images'][f'{idx}_{i}'][()], dtype=torch.float32) for i in range(len(self.gazeheatmap))]
        image_path = self.hdf5_data['image_paths'][idx].astype('str')
        # print(gazeheat_path)
        return imgs, self.taskmaps, age_edu, target, image_path

    def __len__(self):
        return len(self.data)    

class ADNC2_Dataloader(Dataset):
    def __init__(self, root, data_list, label, transform=None, hdf5_file=None):  
        self.root = root
        self.data = data_list 
        self.label = label
        self.transform = transform
        self.hdf5_file = hdf5_file or os.path.join(root, 'data.hdf5')
        self.gazeheatmap = ["presaccade1_l.png", "presaccade1_r.png", "presaccade2_l.png", "presaccade2_r.png", 
                            "antisaccade1_l.png", "antisaccade1_r.png", "antisaccade2_l.png", "antisaccade2_r.png", 
                            "sensitivity1_l.png", "sensitivity1_r.png", "sensitivity2_l.png", "sensitivity2_r.png", 
                            "sensitivity3_l.png", "sensitivity3_r.png", "saliency1_l.png", "saliency1_r.png", 
                            "saliency2_l.png", "saliency2_r.png", "saliency3_l.png", "saliency3_r.png", 
                            "saliency4_l.png", "saliency4_r.png", "saliency5_l.png", "saliency5_r.png"]
        if not os.path.exists(self.hdf5_file):
            self.create_hdf5(self.hdf5_file, self.data)
        self.hdf5_data = h5py.File(self.hdf5_file, 'r')
        self.taskmaps = [torch.tensor(self.hdf5_data['taskmaps'][str(i)][()], dtype=torch.float32) for i in range(1, 13)]
    
    def create_hdf5(self, file_path, data):
        with h5py.File(file_path, 'w') as file:
            # Create datasets for images and labels
            gaze_images = file.create_group('gaze_images')
            taskmaps = file.create_group('taskmaps')

            # Save taskmaps
            for i in range(1, 13):
                img_path = os.path.join(self.root, 'taskmaps', f'{i}.png')
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                taskmaps.create_dataset(str(i), data=img.numpy())

            # Iterate over rows in the DataFrame to save gaze heatmap images
            for index, row in data.iterrows():
                gazeheat_path = os.path.join(self.root, row.iloc[0], 'gazeheat')
                for idx, gh in enumerate(self.gazeheatmap):
                    img_path = os.path.join(gazeheat_path, gh)
                    img = Image.open(img_path)
                    if self.transform:
                        img = self.transform(img)
                    gaze_images.create_dataset(f'{index}_{idx}', data=img.numpy())

            # Save other data
            file.create_dataset('image_paths', data=data.iloc[:, 0].astype('S'))  # 存储图像路径
            file.create_dataset('mmse', data=data.iloc[:, 1].to_numpy(), dtype='f')
            file.create_dataset('moca', data=data.iloc[:, 2].to_numpy(), dtype='f')
            file.create_dataset('other', data=data.iloc[:, 3:8].to_numpy(), dtype='f')
            file.create_dataset('age_edu', data=data.iloc[:, 8:10].to_numpy(), dtype='f')
            file.create_dataset('icls', data=data.iloc[:, 11].to_numpy(), dtype='f')
    
    def __getitem__(self, idx):
        mmse = torch.tensor(self.hdf5_data['mmse'][idx], dtype=torch.float32)
        moca = torch.tensor(self.hdf5_data['moca'][idx], dtype=torch.float32)
        other = torch.tensor(self.hdf5_data['other'][idx], dtype=torch.float32)
        age_edu = torch.tensor(self.hdf5_data['age_edu'][idx], dtype=torch.float32)
        icls = torch.tensor(self.hdf5_data['icls'][idx], dtype=torch.float32)

        if self.label == 'mmse':
            target = mmse
        elif self.label == 'moca':
            target = moca
        elif self.label == 'demo':
            target = mmse  # Demo 也使用 MMSE 作为目标
        elif self.label == 'cls':
            target = icls
        else:
            raise ValueError('Invalid label specified')
        weight = 2.0 if target < 24 else 1.0
        imgs = [torch.tensor(self.hdf5_data['gaze_images'][f'{idx}_{i}'][()], dtype=torch.float32) for i in range(len(self.gazeheatmap))]
        image_path = self.hdf5_data['image_paths'][idx].astype('str')
        # print(gazeheat_path)
        return imgs, self.taskmaps, age_edu, target, weight, image_path

    def __len__(self):
        return len(self.data)


class ADF_Dataloader(Dataset):
    def __init__(self, root, data_list, label, transform=None):  
        self.root = root
        self.data = data_list 
        self.label = label
        self.transform = transform
        self.taskmaps = self.load_taskmaps(os.path.join(root, 'taskmaps'))
        self.gazeheatmap = ["presaccade1_l.png", "presaccade1_r.png", "presaccade2_l.png", "presaccade2_r.png", 
                            "antisaccade1_l.png", "antisaccade1_r.png", "antisaccade2_l.png", "antisaccade2_r.png", 
                            "sensitivity1_l.png", "sensitivity1_r.png", "sensitivity2_l.png", "sensitivity2_r.png", 
                            "sensitivity3_l.png", "sensitivity3_r.png", "saliency1_l.png", "saliency1_r.png", 
                            "saliency2_l.png", "saliency2_r.png", "saliency3_l.png", "saliency3_r.png", 
                            "saliency4_l.png", "saliency4_r.png", "saliency5_l.png", "saliency5_r.png"]
        
    def load_taskmaps(self, taskmap_path):
        taskmap_images = []
        for i in range(1,13):  # 假设有 16 张 taskmap 图片
            img_path = os.path.join(taskmap_path, f'{i}.png')  # 根据实际情况修改文件名格式
            with Image.open(img_path) as img:
                if self.transform is not None:
                    img = self.transform(img)
                taskmap_images.append(img)
        return taskmap_images

    def __getitem__(self, idx):
        mmse = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)
        moca = torch.tensor(self.data.iloc[idx, 2], dtype=torch.float32)
        other = torch.tensor(self.data.iloc[idx, 3:8], dtype=torch.float32)
        age_edu = torch.tensor(self.data.iloc[idx, 8:10], dtype=torch.float32)
        icls = torch.tensor(self.data.iloc[idx, 11], dtype=torch.float32)
        # disease = torch.tensor(self.data.iloc[idx, 10], dtype=torch.float32)

        if self.label == 'mmse':
            target = mmse
        elif self.label == 'moca':
            target = moca
        else:
            raise ValueError('label must be mmse, moca, or fuse')
        # age_edu = torch.tensor(self.data.iloc[idx, 3:5], dtype=torch.float32) 
        # print(self.data.iloc[idx, 0])
        gazeheat_path = os.path.join(self.root, self.data.iloc[idx, 0], 'gazeheat')
        
        imgs = []
        for gh in self.gazeheatmap:
            img_path = os.path.join(gazeheat_path, gh)
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        # print(gazeheat_path)
        return imgs, self.taskmaps, age_edu, icls, target 

    def __len__(self):
        return len(self.data)
          

class Test_dataloader(Dataset):
    def __init__(self, root, data_list, label, transform=None, hdf5_file=None):  
        self.root = root
        self.data = data_list 
        self.label = label
        self.transform = transform
        self.hdf5_file = os.path.join(root, hdf5_file)
        self.gazeheatmap = ["presaccade1_l.png", "presaccade1_r.png", "presaccade2_l.png", "presaccade2_r.png", 
                            "antisaccade1_l.png", "antisaccade1_r.png", "antisaccade2_l.png", "antisaccade2_r.png", 
                            "sensitivity1_l.png", "sensitivity1_r.png", "sensitivity2_l.png", "sensitivity2_r.png", 
                            "sensitivity3_l.png", "sensitivity3_r.png", "saliency1_l.png", "saliency1_r.png", 
                            "saliency2_l.png", "saliency2_r.png", "saliency3_l.png", "saliency3_r.png", 
                            "saliency4_l.png", "saliency4_r.png", "saliency5_l.png", "saliency5_r.png"]
        if not os.path.exists(self.hdf5_file):
            self.create_hdf5(self.hdf5_file, self.data)
        self.hdf5_data = h5py.File(self.hdf5_file, 'r')
        self.taskmaps = [torch.tensor(self.hdf5_data['taskmaps'][str(i)][()], dtype=torch.float32) for i in range(1, 13)]
    
    def create_hdf5(self, file_path, data):
        with h5py.File(file_path, 'w') as file:
            # Create datasets for images and labels
            gaze_images = file.create_group('gaze_images')
            taskmaps = file.create_group('taskmaps')

            # Save taskmaps
            for i in range(1, 13):
                img_path = os.path.join(self.root, 'taskmaps', f'{i}.png')
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                taskmaps.create_dataset(str(i), data=img.numpy())

            # Iterate over rows in the DataFrame to save gaze heatmap images
            for index, row in data.iterrows():
                gazeheat_path = os.path.join(self.root, row.iloc[0], 'gazeheat')
                for idx, gh in enumerate(self.gazeheatmap):
                    img_path = os.path.join(gazeheat_path, gh)
                    img = Image.open(img_path)
                    if self.transform:
                        img = self.transform(img)
                    gaze_images.create_dataset(f'{index}_{idx}', data=img.numpy())

            # Save other data
            file.create_dataset('image_paths', data=data.iloc[:, 0].astype('S'))  # 存储图像路径
            file.create_dataset('mmse', data=data.iloc[:, 1].to_numpy(), dtype='f')
            file.create_dataset('moca', data=data.iloc[:, 2].to_numpy(), dtype='f')
            file.create_dataset('other', data=data.iloc[:, 3:8].to_numpy(), dtype='f')
            file.create_dataset('age_edu', data=data.iloc[:, 8:10].to_numpy(), dtype='f')
            file.create_dataset('icls', data=data.iloc[:, 11].to_numpy(), dtype='f')
    
    def __getitem__(self, idx):
        mmse = torch.tensor(self.hdf5_data['mmse'][idx], dtype=torch.float32)
        moca = torch.tensor(self.hdf5_data['moca'][idx], dtype=torch.float32)
        other = torch.tensor(self.hdf5_data['other'][idx], dtype=torch.float32)
        age_edu = torch.tensor(self.hdf5_data['age_edu'][idx], dtype=torch.float32)
        icls = torch.tensor(self.hdf5_data['icls'][idx], dtype=torch.float32)

        if self.label == 'mmse':
            target = mmse
        elif self.label == 'moca':
            target = moca
        elif self.label == 'other':
            target = other
        elif self.label == 'cls':
            target = icls
        else:
            raise ValueError('Invalid label specified')
        imgs = [torch.tensor(self.hdf5_data['gaze_images'][f'{idx}_{i}'][()], dtype=torch.float32) for i in range(len(self.gazeheatmap))]
        image_path = self.hdf5_data['image_paths'][idx].astype('str')
        # print(gazeheat_path)
        return imgs, self.taskmaps, age_edu, target, icls

    def __len__(self):
        return len(self.data)
    

import os
import pandas as pd
from PIL import Image
import numpy as np

class ML_Dataloader():
    def __init__(self, root, data_list, label, transform=None):  
        self.root = root
        self.data = data_list 
        self.label = label
        self.transform = transform

    def load_data(self):
        features = []
        targets = []

        for idx in range(len(self.data)):
            mmse = self.data.iloc[idx, 1]
            moca = self.data.iloc[idx, 2]
            other = self.data.iloc[idx, 3:8].values
            age_edu = self.data.iloc[idx, 8:10].values

            if self.label == 'mmse':
                target = mmse
            elif self.label == 'moca':
                target = moca
            elif self.label == 'cls':
                icls = self.data.iloc[idx, 12]
                target = icls
            else:
                raise ValueError('Label must be mmse, moca, or cls')
            
            feature_vector = np.concatenate([other, age_edu])
            features.append(feature_vector)
            targets.append(target)

        return np.array(features), np.array(targets)