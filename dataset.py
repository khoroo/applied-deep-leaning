import torch
from torch.utils import data
import numpy as np
import pickle

import mylibrosa
from SpecAugment.spec_augment_pytorch import spec_augment


class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode, spec_augment, freq_mask, time_mask):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode
        self.spec_augment = spec_augment
        self.freq_mask = freq_mask
        self.time_mask = time_mask
    
    def transform_feature(self, feature):
        feature_shape = feature.shape
        feature = mylibrosa.db_to_power(feature)
        feature = feature.reshape(-1, *feature_shape)
        feature = torch.from_numpy(feature)
        feature = spec_augment(
            feature,
            frequency_masking_para = self.freq_mask,
            time_masking_para = self.time_mask,
        )
        feature = feature.numpy().reshape(feature_shape)
        feature = mylibrosa.power_to_db(feature)
        return feature
    
    def __getitem__(self, index):
        item = self.dataset[index]
        item_features = item['features']
        cst = np.concatenate((item_features['chroma'],
                              item_features['spectral_contrast'],
                              item_features['tonnetz']))
        mfcc = item_features['mfcc']
        lms = item_features['logmelspec']
            
        
        if self.mode == 'LMC':
            feature = np.concatenate((lms,cst))
        elif self.mode == 'MC':
            feature = np.concatenate((mfcc,cst))
        elif self.mode == 'MLMC':
            feature = np.concatenate((mfcc,lms,cst))
        else:
            raise ValueError(f'Invalid mode: {self.mode} is not one of LMC, MC or MLMC')
        
        
        if (self.spec_augment is not 0) and (np.random.uniform(0,1,1)[0] <= self.spec_augment):
            feature = self.transform_feature(feature)
        feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        label = item['classID']
        fname = item['filename']
        return feature, label, fname


    def __len__(self):
        return len(self.dataset)
