import torch
from PIL import Image
import os
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_images(mode, file_name, magnification, pretrained_model_name):
    
    train_classes = {'adenosis': 0, 'fibroadenoma': 1, 'ductal_carcinoma': 2, 'lobular_carcinoma': 3}
    val_classes = {'phyllodes_tumor':0,'papillary_carcinoma':1}
    test_classes = {'tubular_adenoma':0,'mucinous_carcinoma':1}
    
    split = {'train': train_classes,
            'val': val_classes,
            'test': test_classes}
    
    classes = split[mode]
    tumour_type = ['benign','malignant']
    magnification = f'{magnification}X'

    labels = []
    data = []
    print("im here wuhe")
    for name in classes.keys():
        for type in tumour_type:
            path = os.path.join(file_name, 'breast', type, 'SOB', name)
            if os.path.exists(path):
                print("tumour type: ",type, "name: ",name)
                for patient in os.listdir(path):
                    path_im=os.path.join(path,patient,magnification)
                    if os.path.exists(path_im):
                        for image in os.listdir(path_im):
                            img = Image.open(os.path.join(path_im,image))
                            embedding = get_embedding(img, pretrained_model_name)
                            data.append(embedding)
                            labels.append(classes[name])
    
    samples = torch.tensor(data, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.int32)
    print(f'The shapes of the embeddings and labels:{samples.shape}, {targets.shape}')

    path_folder = os.path.join(file_name,'embeddings-with-'+pretrained_model_name)
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    path_sample_save = os.path.join(path_folder, f'{mode}_data_{magnification}.pt')
    path_labels_save = os.path.join(path_folder,f'{mode}_labels_{magnification}.pt')

    torch.save(samples, path_sample_save)
    torch.save(targets, path_labels_save)

    return samples, targets 

def get_embedding(image, pretrained_model_name):
    # Load a pretrained model and remove the last layer to access the embeddings
    model = models.__dict__[pretrained_model_name](pretrained=True)
    #model = models.efficientnet_b0(pretrained=True)
    embedding_layer = nn.Sequential(*list(model.children())[:-1])
    embedding_layer.eval()
    embedding_layer.to(device)

    if pretrained_model_name=="convnext_base":
            # Preprocess the image
            preprocess = transforms.Compose([
                    transforms.Resize(232),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    elif pretrained_model_name=="efficientnet_b0":
            # Preprocess the image
            preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    

    img = preprocess(image)
    img = img.unsqueeze(0).to(device)

    embedding = embedding_layer(img)
    embedding = embedding.squeeze().detach().cpu().numpy()
    
    return embedding

