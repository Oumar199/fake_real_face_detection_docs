
from fake_face_detection.data.fake_face_dataset import FakeFaceDetectionDataset
from fake_face_detection.metrics.compute_metrics import compute_metrics
from torch.utils.tensorboard import SummaryWriter
from PIL.JpegImagePlugin import JpegImageFile
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import pandas as pd
from math import *
import numpy as np
import torch
import os

def get_attention(image: str, attention: torch.Tensor, size: tuple, patch_size: tuple):
    
    # recuperate the image as a numpy array
    with Image.open(image) as img:
    
        img = np.array(transforms.Resize(size)(img))
        
    # recuperate the attention provided by the last patch (notice that we eliminate 1 because of the +1 added by the convolutation layer)
    attention = attention[:, -1, 1:]

    # calculate the mean attention
    attention = attention.mean(axis = 0)

    # let us reshape transform the image to a numpy array

    # calculate the scale factor
    scale_factor = size[0] * size[1] / (patch_size[0] * patch_size[1])

    # rescale the attention with the nearest scaler
    attention = F.interpolate(attention.reshape(1, 1, -1), scale_factor=scale_factor,
                            mode='nearest')

    # let us reshape the attention to the right size
    attention = attention.reshape(size[0], size[1], 1)
    
    # recuperate the result
    attention_image = img / 255 * attention.numpy()
    
    return attention_image


def make_predictions(test_dataset: FakeFaceDetectionDataset,
                     model,
                     log_dir: str = "fake_face_logs",
                     tag: str = "Attentions",
                     batch_size: int = 3,
                     size: tuple = (224, 224), 
                     patch_size: tuple = (14, 14),
                     figsize: tuple = (24, 24)):
    
    with torch.no_grad():
        
        _ = model.eval()
        
        # initialize the logger
        writer = SummaryWriter(os.path.join(log_dir, "attentions"))
        
        # let us recuperate the images and labels
        images = test_dataset.images
        
        labels = test_dataset.labels
        
        # let us initialize the predictions
        predictions = {'attentions': [], 'predictions': [], 'true_labels': labels, 'predicted_labels': []}

        # let us initialize the dataloader
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        # get the loss
        loss = 0
        
        for data in test_dataloader:
            
            # recuperate the pixel values
            pixel_values = data['pixel_values'][0].cuda()
            
            # recuperate the labels
            labels_ = data['labels'].cuda()
            
            # # recuperate the outputs
            outputs = model(pixel_values, labels = labels_, output_attentions = True)
            
            # recuperate the predictions
            predictions['predictions'].append(torch.softmax(outputs.logits.detach().cpu(), axis = -1).numpy())
            
            # recuperate the attentions of the last encoder layer
            predictions['attentions'].append(outputs.attentions[-1].detach().cpu())
            
            # add the loss
            loss += outputs.loss.detach().cpu().item()
        
        predictions['predictions'] = np.concatenate(predictions['predictions'], axis = 0)
        
        predictions['attentions'] = torch.concatenate(predictions['attentions'], axis = 0)
        
        predictions['predicted_labels'] = np.argmax(predictions['predictions'], axis = -1).tolist()
        
        # let us calculate the metrics
        metrics = compute_metrics((predictions['predictions'], np.array(predictions['true_labels'])))
        metrics['loss'] = loss / len(test_dataloader)
        
        # for each image we will visualize his attention
        nrows = ceil(sqrt(len(images)))
        
        fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize = figsize)
        
        axes = axes.flat
        
        for i in range(len(images)):
            
            attention_image = get_attention(images[i], predictions['attentions'][i], size, patch_size)
        
            axes[i].imshow(attention_image)
            
            axes[i].set_title(f'Image {i + 1}')
            
            axes[i].axis('off')
            
        fig.tight_layout()
        
        [fig.delaxes(axes[i]) for i in range(len(images), nrows * nrows)]
        
        writer.add_figure(tag, fig)    
        
        # let us remove the predictions and the attentions
        del predictions['predictions']
        del predictions['attentions']
        
        # let us recuperate the metrics and the predictions
        return pd.DataFrame(predictions), metrics
        
        
        
