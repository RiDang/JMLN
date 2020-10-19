import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pdb
from .networks import Normalize
#from pytorch_metric_learning import losses
from .pytorch_metric_learning.pytorch_metric_learning import losses
import torch.nn as nn
import torch.nn.functional as F

class RetrievalWorkshopBaselineTuningModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """
        Init model
        """
        BaseModel.__init__(self, opt)
        self.pose_corr = 0.0
        self.center_corr = 0.0
        self.cate_corr = 0.0
        self.pose_center_corr = 0.0
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = [ 'all', 'metric1', 'metric2', 'triplet', 'mi'] #, 'mask', 'G', 'D']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['_backbone_sketch', '_further_conv', '_cate_estimator','_backbone_model','_tricenter','_attn','_trans','_bala' ] 
        else :  # during test time, only load Gs
            #self.model_names = ['_backbone', '_further_conv', '_cate_estimator']
            self.model_names = ['_backbone_sketch', '_further_conv', '_cate_estimator','_backbone_model','_tricenter', '_attn', '_trans', '_bala'] 

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.net_backbone_sketch = networks.define_retrieval_nets(opt, net_option='resnet34_pytorch', gpu_ids=self.gpu_ids)
        #self.net_further_conv = networks.define_retrieval_nets(opt, net_option='bn', gpu_ids=self.gpu_ids)
        self.net_further_conv = networks.define_retrieval_nets(opt, net_option='further_conv', gpu_ids=self.gpu_ids)
        
        self.net_backbone_model = networks.define_retrieval_nets(opt, net_option='resnet34_pytorch', gpu_ids=self.gpu_ids)
        #self.net_further_conv_model = networks.define_retrieval_nets(opt, net_option='further_conv', gpu_ids=self.gpu_ids)
        
        opt.input_dim = 256
        opt.cate_num = 90  # from 7 to 34
        self.net_cate_estimator = networks.define_retrieval_nets(opt, net_option='cate_estimator', gpu_ids=self.gpu_ids)
        self.net_tricenter = networks.define_retrieval_nets(opt, net_option='triplet_center', gpu_ids=self.gpu_ids)
        self.net_attn = networks.define_retrieval_nets(opt, net_option='attn', gpu_ids=self.gpu_ids)
        opt.input_dim = 256*5
        opt.cate_num = 256  # from 7 to 34
        self.net_trans = networks.define_retrieval_nets(opt, net_option='cate_estimator', gpu_ids=self.gpu_ids)
        opt.input_dim = 256
        self.net_bala = networks.define_retrieval_nets(opt, net_option='cate_estimator', gpu_ids=self.gpu_ids)

        #self.net_cate_estimator_model = networks.define_retrieval_nets(opt, net_option='cate_estimator', gpu_ids=self.gpu_ids)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.l2norm = Normalize(2) 
        self.criterionBce = torch.nn.BCEWithLogitsLoss()

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            # define loss functions
            self.criterionSoftmax  = torch.nn.CrossEntropyLoss()
            self.criterionMetric = losses.TripletMarginLoss(triplets_per_anchor="all")
            self.criterionTriplet = torch.nn.TripletMarginLoss(margin=1.0, p=2.0)
            #self.optimizer = torch.optim.SGD(itertools.chain(self.net_backbone.parameters(), self.net_further_conv.parameters(), self.net_center_estimator.parameters(), self.net_cate_estimator.parameters()), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
            self.optimizer = torch.optim.SGD(itertools.chain(self.net_backbone_sketch.parameters(), self.net_further_conv.parameters(), self.net_cate_estimator.parameters(),self.net_backbone_model.parameters(),self.net_tricenter.parameters(),self.net_attn.parameters(),self.net_trans.parameters(),self.net_bala.parameters()), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        self.input_query = input['query_img'].to(self.device)
        self.input_positive = input['positive_img'].to(self.device)
        self.input_negative = input['negative_img'].to(self.device)
        
        shapes = self.input_positive.shape
        self.input_positive = self.input_positive.view(shapes[0]*shapes[1], 3, shapes[-2], shapes[-1])
        self.input_negative = self.input_negative.view(shapes[0]*shapes[1], 3, shapes[-2], shapes[-1])
        
        self.label_query = input['query_label'].to(self.device).view(-1) 
        self.label_positive = input['positive_label'].to(self.device).view(-1) 
        self.label_negative = input['negative_label'].to(self.device).view(-1) 
        #print('------',self.input_query.shape, self.input_positive.shape, self.input_negative.shape)
        #print('======',self.label_query, self.label_positive, self.label_negative) 
    def dist_elur(self, fts_q, fts_c):
        #fts_q = self.l2norm(fts_q)
        #fts_c = self.l2norm(fts_c)
        fts_q = fts_q/torch.norm(fts_q, dim=-1,keepdim=True)
        fts_c = fts_c/torch.norm(fts_c, dim=-1,keepdim=True)
        fts_qs = torch.sum((fts_q)**2,dim=-1,keepdim=True)*1.0
        fts_cs = torch.sum((fts_c)**2,dim=-1,keepdim=True).t()
        qc = torch.mm(fts_q,fts_c.t())
        dist = fts_qs + fts_cs - 2.0 * qc +1e-4
        return torch.sqrt(dist)

    def mi_dist(self, fts1, fts2,la1, la2, margin=1.0):
        dist = self.dist_elur(fts1,fts2)
        index = (la1.reshape(-1,1)==la2.reshape(1,-1))#.bool()
        ap = dist[index]
        lens = len(ap)
        an = torch.sort(dist[~index])[0][:lens]
        if lens*2 > (dist.shape[0]**2):
            ap = ap.mean().unsqueeze(0)
            an = an.mean().unsqueeze(0)
        loss = nn.MarginRankingLoss(margin)(ap,an,torch.Tensor([-1.0]).to(self.device)        )
        return loss,0 #ap.mean()

    def forward(self): 
        
        # baseline xxx
        self.feat_query = self.net_bala(self.net_further_conv(self.net_backbone_sketch(self.input_query)))
        
        self.bb_p = self.net_further_conv(self.net_backbone_model(self.input_positive))
        shps = self.bb_p.shape
        self.bb_n = self.net_further_conv(self.net_backbone_model(self.input_negative))
       
        # attn_cat
        self.bb_p = self.net_attn(self.bb_p.reshape(-1, 5, *(shps[1:]))) #.flatten(2))
        self.bb_n = self.net_attn(self.bb_n.reshape(-1, 5, *(shps[1:]))) #.flatten(2))
        
        self.feat_positive = self.net_trans(self.bb_p.flatten(1))  #.max(dim=1)[0]
        self.feat_negative = self.net_trans(self.bb_n.flatten(1))  #.max(dim=1)[0]

        self.feat_all = torch.cat((self.feat_query, self.feat_positive, self.feat_negative),dim=0)
        self.score_all = self.net_cate_estimator(self.feat_all)
         
    def backward(self):
        """Calculate the loss"""
        self.label_all = torch.cat((self.label_query, self.label_positive, self.label_negative),dim=0)
        self.loss_all = self.criterionSoftmax(self.score_all, self.label_all)*1.0

        self.loss_metric1, _ = self.net_tricenter(self.feat_all[:self.opt.batch_size,:], self.label_all[:self.opt.batch_size])
        self.loss_metric2, _ = self.net_tricenter(self.feat_all[self.opt.batch_size:,:], self.label_all[self.opt.batch_size:])

        #self.loss_metric = self.net_tricenter(self.feat_query, self.label_query)*1.0 + self.net_tricenter(self.feat_all[self.opt.batch_size:,:], self.label_all[self.opt.batch_size:, :])*1.0
        self.loss_triplet = self.criterionTriplet(self.l2norm(self.feat_query), self.l2norm(self.feat_positive), self.l2norm(self.feat_negative))*5.0

        self.loss_mi,ap = self.mi_dist(self.feat_query, self.feat_positive, self.label_query, self.label_positive) 
        

        lam =0.3
        self.loss = self.loss_all +\
                        self.loss_metric1*lam +self.loss_metric2*(1-lam) +\
                        self.loss_triplet + self.loss_mi*5.0
        


        
        self.score_query = self.score_all[:self.opt.batch_size,:]
        self.score_model = self.score_all[self.opt.batch_size:,:]
        self.label_model = torch.cat((self.label_positive,self.label_negative),dim=0)
        _, max_cate = torch.max(self.score_query.data, 1)
        cate_corr = torch.sum(max_cate == self.label_query.data)
        self.cate_corr += cate_corr.item()
        
        _, max_center = torch.max(self.score_model.data, 1)
        center_corr = torch.sum(max_center == self.label_model.data)
        self.center_corr += center_corr.item()
        
        self.loss.backward()
    
    def forward_eval(self):
        ################ 3d data without mask
        #query_backbone_feat = self.net_further_conv(self.net_backbone(self.input_query))
        #self.cate_feat, self.query_cate_score = self.net_cate_estimator(query_backbone_feat, return_feat=True)
        self.cate_feat = self.net_bala(self.net_further_conv(self.net_backbone_sketch(self.input_query)))
        if self.cate_feat.shape[0]>1:
            self.cate_feat = self.net_further_conv(self.net_backbone_model(self.input_query))
            shps = self.cate_feat.shape
            self.cate_feat = self.net_attn(self.cate_feat.reshape(-1,5,shps[-1]))
            self.cate_feat = self.net_trans(self.cate_feat.flatten(1))
            #self.cate_feat = self.net_further_conv([self.net_backbone_model(self.input_query)[0].max(dim=0,keepdim=True)[0]])
            
            #self.query_cate_score = self.query_cate_score.mean(dim=0, keepdim=True)#[0]
         
        return self.cate_feat, self.label_query, torch.Tensor([0])#self.query_cate_score

    def set_input_eval(self, input):
        self.input_query = input['query_img'].to(self.device)
        self.label_query = input['query_label'].to(self.device)
        if len(self.input_query.shape)==5: # 1*5*3*256*256 or 1*3*256*256 
            self.input_query = self.input_query.squeeze()
    
    
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()              # compute fake images and reconstruction images.
        
        self.optimizer.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward()             # calculate gradients for G_A and G_B
        self.optimizer.step()       # update G_A and G_B's weights
        
