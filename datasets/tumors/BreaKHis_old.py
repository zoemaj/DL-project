from abc import ABC
import numpy as np
from torch.utils.data import DataLoader
import torch
from datasets.tumors.utils import *
from datasets.dataset import *
#from keras.applications.vgg16 import VGG16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

''' QUESTIONS POUR ASSISTANTS:
- there is a problem with python3 run.py MNSIT -> it quits. Problem of memory ?but we tried with higher cpu: 15Go and gpu: 500Go. -> fixed
- sort the magnifications before or test with everythings before? yes
- do we need to do a model without few shot to show that it doesn't work?
-> pretrained and train on every types of tumour to see if it works !
- n shots ? choose one and keep it for all the experiments
- do we need to do utils.py for the class in datasets?
'''
class TMDataset(FewShotDataset, ABC):
    _dataset_name = 'BreakHis'
    _dataset_url = 'https://drive.google.com/file/d/1-sPUJg55j8nRPTNujGKA18yAeDICtY5k/view?usp=share_link'
    def load_data(self,mode='train', magnification=40, min_samples=20):
        mode='train'
        # load data from the file Kather_texture_2016_image_tiles_5000 by compressing each image as a tensor and with the label as the name of the folder
        file_name='data/BreaKHis_v1/breast'

        #classes={'adenosis':0,'fribroadenoma':1,'phyliodes_tumor':2,'tubular_adenoma':3,'ductal_carcinoma':4,'lobular_carcinoma':5,'mucinous_carcinoma':6,'papillary_carcinoma':7}

        #need to find sources for these:
        # ductual carninoma is the most common type of breast cancer, starting in the cells that line the milk ducts. (chatgpt)

        # most RARE BENIGN TUMOR
        # phyliodes tumor is rare type of breast tumor that forms in the connective tissue (stroma) of the breast. (chatgpt)
        # tubular adenoma is a rare type of benign breast tumor that forms in the milk ducts. It is typically well-defined and consists of small tube-like structures. (chatgpt)

        # most RARE MALIGNANT TUMOR
        # papillary carinoma is a rare type of breast cancer that forms in a milk duct. It grows inside the duct but has a frond-like growth that projects into the duct's lumen (central cavity). (chatgpt)
        # Mucinous carcinoma is a rare type of invasive breast cancer characterized by the production of mucin. It is less common compared to more typical forms of breast cancer. (chatgpt)

        train_labels = {'adenosis': 0, 'fibroadenoma': 1, 'ductual_carcinoma': 2, 'lobular_carcinoma': 3}

        val_labels={'phyliodes_tumor':0,'papillary_carcinoma':1} #one most rare benign and one most rare malignant
        test_labels={'tubular_adenoma':0,'mucinous_carcinoma':1} #one most rare benign and one most rare malignant

        split = {'train': train_labels,
                    'val': val_labels,
                    'test': test_labels}
        classes = split[mode]

        print("------------- load the images and labels of benign tumour --------------")
        data,labels=get_data(file_name, classes, magnification)

        #ADD SOMETHING FOR MIN_SAMPLES HERE! 

        #only keep 0:min_samples of embedding
        sample = data
        #same things for the label
        target = labels
        #print("we took only ",min_samples," samples from the embedding and labels")
        print("sample.shape=",sample.shape)
        print("target.shape=",target.shape)
        return sample,target

class TMSimpleDataset(TMDataset):
   # def __init__(self, batch_size, root='./data/', mode='train', magnification=40, min_samples=20):
    def __init__(self, batch_size, root='./data/', mode='train', min_samples=20):
        self.magnification=40 #TEMPORARY SOLUTION
        self.initialize_data_dir(root, download_flag=False)
        self.samples, self.targets = self.load_data(mode, self.magnification, min_samples)
        self.batch_size = batch_size
        super().__init__()

    def __getitem__(self, i):
        return self.samples[i], self.targets[i]

    def __len__(self):
        return self.samples.shape[0]

    @property
    def dim(self):
        return self.samples.shape[1]

    def get_data_loader(self) -> DataLoader:
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self, **data_loader_params)

        return data_loader
    
class TMSetDataset(TMDataset):

    #def __init__(self, n_way, n_support, n_query, n_episode=100, root='./data', mode='train', magnification=40):
    def __init__(self, n_way, n_support, n_query, n_episode=100, root='./data', mode='train'):
        self.initialize_data_dir(root, download_flag=False)

        self.n_way = n_way
        self.n_episode = n_episode
        min_samples = n_support + n_query
        #TEMPORARY SOLUTION
        self.magnification=40
        samples_all, targets_all = self.load_data(mode, self.magnification, min_samples)
        self.categories = np.unique(targets_all)  # Unique cell labels
        self.x_dim = samples_all.shape[1]

        self.sub_dataloader = []

        sub_data_loader_params = dict(batch_size=min_samples,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.categories:
            samples = samples_all[targets_all == cl, ...]
            sub_dataset = FewShotSubDataset(samples, cl)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

        super().__init__()

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.categories)

    @property
    def dim(self):
        return self.x_dim

    def get_data_loader(self) -> DataLoader:
        sampler = EpisodicBatchSampler(len(self), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler=sampler, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self, **data_loader_params)
        return data_loader


    