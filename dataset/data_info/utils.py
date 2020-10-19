import os
import shutil
from tqdm import tqdm
import json
import re
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn



def train_test():
    data_path = ['..\\dataset\\train_data',
                '..\\dataset\\test_data']
    cls = os.listdir(data_path[0])
    train_file = [os.path.join(data_path[0],f) for f in cls]
    test_file = [os.path.join(data_path[1],f) for f in cls]

    for tr,te in zip(tqdm(train_file), test_file):
        shutil.move(os.path.join(te,'test'),tr)

def generate_json():
    root = 'D:\\Temp\\Data\\SHREC\\SHREC13\\dataset\\data_info'

    path = ['SHREC13_SBR_Model.cla',\
        'SHREC13_SBR_Sketch_Train.cla',\
        'SHREC13_SBR_Sketch_Test.cla']
    cls_path = ['..\\dataset\\sketch\\']
    cls = sorted(os.listdir(cls_path[-1]))
    cls_dict = {}
    for c,n in zip(cls,range(len(cls))):
        cls_dict[c] = n

    json_file = [
        'level_model.json',
        'level_stech.json',
        ]

    # 处理model
    # for pi in path:
    fp = open(os.path.join(root, path[1]), 'r')
    fj = open(os.path.join(root, json_file[1]), 'w')  # label
    level_dict = {} # {label : obj}
    data = fp.readlines()
    for idx, line in enumerate(data):
        mark = line.split()
        if len(mark)>= 3:
            level_dict[cls_dict[mark[0]]] = sorted([int(item.split()[0]) for item in data[idx+1 : idx+ 1 + int(mark[-1])]])
    json.dump(level_dict, fj)
    count = 0
    for i in level_dict:
        count += len(level_dict[i])
    print('==', count)
    fp.close()
    fj.close()


    # # 处理model
    # # for pi in path:
    # fp = open(os.path.join(root, path[0]), 'r')
    # fj = open(os.path.join(root, json_file[0]), 'w')  # label
    # level_dict = {} # {label : obj}
    # data = fp.readlines()
    # for idx, line in enumerate(data):
    #     mark = line.split()
    #     if len(mark)>= 3:
    #         level_dict[cls_dict[mark[0]]] = sorted([int(item.split()[0]) for item in data[idx+1 : idx+ 1 + int(mark[-1])]])
    # json.dump(level_dict, fj)
    # count = 0
    # for i in level_dict:
    #     count += len(level_dict[i])
    # print('==', count)
    # fp.close()
    # fj.close()

def model_train_test():
    path = '..\\dataset\\data_info\\level_model_all.json'
    orign_path = '..\\dataset\\models'
    target_path = ['..\\dataset\\models2\\train\\',
                '..\\dataset\\models2\\test\\',]

    # 获取train+test
    all = json.load(open(path, 'r'))
    train = {}
    test = {}
    for id in all:
        lens = len(all[id])

        if lens < 6:
            index = random.randint(0, lens)
            test[id] = [all[id][index]]
            all[id].remove(all[id][index])
            train[id] =all[id]
            tr_len = len(train[id])
            te_len = len(test[id])

        else:
            index = sorted(random.choice(range(lens), (lens//6), replace=False))  # 直接使用 random.randint 导致重复
            test[id] = [all[id][i] for i in index]
            for c in test[id]:
                all[id].remove(c)
            train[id]= all[id]

            tr_len = len(train[id])
            te_len = len(test[id])
            # print('test:%s--'% id, test[id])
            # print('train:%s--'% id, train[id])

        # 检验是否满足要求
        # print('+++:%d,%d,%d'%(lens,tr_len, te_len))
        # if lens !=(tr_len+te_len): print('error!!!!')
    # all = json.load(open(path, 'r'))
    # fs = os.listdir('..\\dataset\\models\\')
    # fs = [i.split('.')[0][] for i in fs]


    #exit(-1)
    json.dump(train, open('..\\dataset\\data_info\\level_model.json', 'w'))

    for tr in tqdm(train):
        #print('----tr:', train[tr])
        for i in range(len(train[tr])):
            shutil.copy(os.path.join(orign_path, 'm'+'%d'% train[tr][i]+'.off'), target_path[0])
        for i in range(len(test[tr])):
            shutil.copy(os.path.join(orign_path, 'm'+'%d'%test[tr][i]+'.off'), target_path[1])

def obj_label():
    path = ['..\\dataset\\data_info\\level_model.json', \
            '..\\dataset\\data_info\\level_sketch.json', \
            ]
    save_path = ['..\\dataset\\data_info\\label_model.json', \
            '..\\dataset\\data_info\\label_sketch.json', \
            ]
    index = 1
    fo = json.load(open(path[index],'r'))
    fs = open(save_path[index],'w')
    item = {}
    for i in fo:
        for j in fo[i]:
            item[j]=int(i)
    json.dump(item, fs)
def other():
    root = 'D:\\Temp\\Data\\SHREC\\SHREC13\\dataset\\data_info'

    path = ['SHREC13_SBR_Model.cla',\
        'SHREC13_SBR_Sketch_Train.cla',\
        'SHREC13_SBR_Sketch_Test.cla']
    cls_path = ['..\\dataset\\sketch\\']
    cls = sorted(os.listdir(cls_path[-1]))
    cls_dict = {}
    for c,n in zip(cls,range(len(cls))):
        cls_dict[c] = n

    json_file = [
        'level_model.json',
        'level_stech.json',
        ]

    # 处理model
    # for pi in path:
    fp = open(os.path.join(root, path[1]), 'r')
    fj = open(os.path.join(root, json_file[1]), 'w')  # label
    level_dict = {} # {label : obj}
    data = fp.readlines()
    for idx, line in enumerate(data):
        mark = line.split()
        if len(mark)>= 3:
            level_dict[cls_dict[mark[0]]] = sorted([int(item.split()[0]) for item in data[idx+1 : idx+ 1 + int(mark[-1])]])
    json.dump(level_dict, fj)
    count = 0
    for i in level_dict:
        count += len(level_dict[i])
    print('==', count)
    fp.close()
    fj.close()

def other():
    path = '..\\dataset\\sketch\\'
    save_path = '..\\dataset\\data_info\\label_sketch.json'
    cls = sorted(os.listdir(path))
    dict = {}
    for id,c in enumerate(cls):
        tr = os.listdir(os.path.join(path, c, 'train'))
        te = os.listdir(os.path.join(path, c, 'test'))
        tr  = [int(i.split('.')[0]) for i in tr]
        te  = [int(i.split('.')[0]) for i in te]
        temp = sorted(tr+te)
        for k in temp:
            dict[k]=id
    json.dump(dict, open(save_path,'w'))

def yield_contour():
    path = ['0000012.png', \
            '0000025.png', \
            'm0.4.png'
            ]
    img1 = Image.open(path[2]).convert('L')
    img1.show()

    img1_data = transforms.ToTensor()(img1).float()

    # img2_data = (img1_data>0).float()
    # img2 = transforms.ToPILImage()(img2_data)
    # img2.show()

    img3_data_r = abs(img1_data[:,1:,:] - img1_data[:,:-1,:])
    img3_data_l = abs(img1_data[:, :, 1:] - img1_data[:, :, :-1])

    img3 = transforms.ToPILImage()((img3_data_r[:,:,1:] + img3_data_l[:,1:,:])*2)
    #img3 = transforms.ToPILImage()((img3_data_r[:,:,1:]))
    img3.show()

def random_sample():
    pass

def sketch_swell(in_path, out_path,resolution=100,iter=5):

    img1 = Image.open(in_path).convert('L')
    img1 = 1-transforms.ToTensor()(img1).float()

    ker = nn.Conv2d(1, 1, 2, padding=1, bias=False)
    ker.weight.requires_grad=False
    ker.weight.data = torch.Tensor([[[[0.125,0.125,0.125],[0.125,1,0.125,],[0.125,0.125,0.125]]]])
    data = img1.unsqueeze(0)
    for i in range(iter):
        data = ker(data)
    data[data>0.05] = 1  # 转换为二值图片
    data = data.squeeze(0)
    img1 = transforms.ToPILImage()(data)
    img1 = transforms.Resize(resolution)(img1)
    img1.save(out_path)
    img1.close()

def yield_sketch_swell():
    pass
if __name__ == '__main__':
    # path = ['sketch/1.png','sketch/t.png']
    # sketch_expand(path[0],path[1])
    root = ''
    in_path = 'sketch'
    out_path = 'sketch_swell_200'
    mode = 'train' #'test' #
    for cls in tqdm(os.listdir(os.path.join(root, in_path))):
        if not os.path.isdir(os.path.join(root, out_path, cls)): os.mkdir(os.path.join(root, out_path, cls))
        for obj in  os.listdir(os.path.join(root, in_path, cls, mode)):
            if not os.path.isdir(os.path.join(root, out_path, cls,mode)): os.mkdir(os.path.join(root, out_path, cls, mode))
            in_img = os.path.join(root, in_path, cls, mode, obj)
            out_img = os.path.join(root, out_path, cls, mode, obj)
            sketch_swell(in_img, out_img,resolution=200)




