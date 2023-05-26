
from fake_face_detection.utils.compute_weights import compute_weights
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import torch
import os

class FakeFaceDetectionDataset(Dataset):
    
    def __init__(self, fake_path: str, real_path: str, id_map: dict, transformer, **transformer_kwargs):
        
        # let us load the images 
        self.fake_images = glob(os.path.join(fake_path, "*"))
        
        self.real_images = glob(os.path.join(real_path, "*"))
        
        self.images = self.fake_images + self.real_images
        
        # let us recuperate the labels
        self.fake_labels = [int(id_map['fake'])] * len(self.fake_images)
        
        self.real_labels = [int(id_map['real'])] * len(self.real_images)
        
        self.labels = self.fake_labels + self.real_labels
        
        # let us recuperate the weights
        self.weights = torch.from_numpy(compute_weights(self.labels))
        
        # let us recuperate the transformer
        self.transformer = transformer
        
        # let us recuperate the length
        self.length = len(self.labels)
        
        # let us recuperate the transformer kwargs
        self.transformer_kwargs = transformer_kwargs
        
    def __getitem__(self, index):
        
        # let us recuperate an image
        image = self.images[index]
        
        with Image.open(image) as img:
        
            # let us recuperate a label
            label = self.labels[index]
            
            # let us add a transformation on the images
            if self.transformer:
                
                image = self.transformer(img, **self.transformer_kwargs)
            
        # let us add the label inside the obtained dictionary
        image['labels'] = label
        
        return image    
        
    def __len__(self):
        
        return self.length
        
