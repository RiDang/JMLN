import os.path
import sys
sys.path.append('.')
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import random
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from util import custom_transforms
import torchvision.transforms as transforms
import pdb
import torch
import cv2
import json
import copy

from util.smart_crop_transforms import RandomCropDramaticlly
from util import augment

class RetrievalWorkshopBaselineEval3DDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
       
        self.sketch_file = []
        self.sketch_label = []

        self.sketch_path = os.path.join(opt.dataroot, 'sketch')
        self.render_path = os.path.join(opt.dataroot, 'render5')
        self.label_model = json.load(open(os.path.join(opt.dataroot, 'data_info','label_model.json'),'r'))
        self.level_model = json.load(open(os.path.join(opt.dataroot, 'data_info','level_model.json'),'r'))
        self.model_file = sorted(self.label_model.keys())
        self.data_size = len(self.label_model) 
        
        self.sketch_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        # render
        self.render_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    def __getitem__(self, index):
        """
        Get training samples
        
        query_img: random choosed rgb sample image;
        positive_img: rendered no-texture sample image, same as query_img;
        negative_img: rendered no-texture sample image, random choosed from leve3 or leve2 dict
        """
        query_img = self._load_pool_image_notexture(int(self.model_file[index]))
               
        query_label = torch.Tensor([self.label_model[self.model_file[index]]]).long()
        
        return {'query_img':query_img,  'query_label':query_label, 'name':self.model_file[index]}
        #return {'query_img':query_img, 'positive_img':positive_img, 'negative_img':negative_img, 'center_label':center_label, 'cate_label':cate_label}

    def __len__(self):
        return self.data_size
    

    def _load_query_image_notexture(self, image_file):
        
        img = Image.open(image_file).convert('RGB')
        trans_img = self.sketch_transform(img)
        return trans_img 

    def _load_pool_image_notexture(self, shape_id):
        syn_id = [30,60,90,120,150]
        img_file = [os.path.join(self.render_path, '%04d'%shape_id, 'image_' + '%03d'%syn_id[i] + '.png') for i in range(5)]
        #img_file= [os.path.join(self.bicycle_image_path,  '%07d.png'%(int(shape_id)*5+i)) for i in range(5)]
        trans_img = []
        for i in img_file:   
            img = Image.open(i).convert('RGB')
            trans_img.append(self.render_transform(img))
        
        #img = Image.open(img_file).convert('RGB')
        #trans_img = self.pool_transform(img)

        return torch.stack(trans_img)


    
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

if __name__ == '__main__':
    from options.train_options import TestOptions
    opt = TrainOptions().parse()
        # hard-code some parameters for test
    opt.level2_dic = 'level2_dic_v1.npy'
    opt.level3_dic = 'level3_dic_v1.npy'
    opt.dataset_name = '3d_validation_set.npy' 
    opt.inplanes = 64
    opt.reverse = True
    opt.pose_num = 12
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
    data = RetrievalWorkshopBaselineDataset(opt)
    for i in data:
        print([i[k].shape for n,k in enumerate(i) if n<3 ])
