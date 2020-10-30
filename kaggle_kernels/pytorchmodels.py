import sys
import pkg_resources
from pip._internal import main as pipmain
if __name__ == '__main__':
    pipmain(['install', '../input/geffnet/gen-efficientnet-pytorch/gen-efficientnet-pytorch-master'])
    pipmain(['install', '../input/pretrained-models/pretrained-models.pytorch-master'])
import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
import copy
from collections import OrderedDict

class Noop(nn.Module):
    def __init__(self,*args):
        super(Noop, self).__init__()
    def forward(self,x):
        return x

class NoopAddDim(nn.Module):
    def __init__(self,dim=-1):
        super(NoopAddDim, self).__init__()
        self.dim=dim
    def forward(self,x):
        return x.unsqueeze(self.dim)

class NoopSqueezeDim(nn.Module):
    def __init__(self,dim=-1):
        super(NoopSqueezeDim, self).__init__()
        self.dim=dim
    def forward(self,x):
        return x.squeeze(self.dim)

def add_to_dim(x,num_dims,dim=0):
    while len(x.shape)<num_dims:
        x=x.unsqueeze(dim)
    return x

class DummyEmbd(nn.Module):
    def __init__(self,out_size,dtype=torch.float32):
        super(DummyEmbd, self).__init__()
        self.out_size=out_size
        self.dtype=dtype
    def forward(self,x):
        return torch.zeros(x.shape+(self.out_size,),dtype=self.dtype,device=x.device)

def soft_cross_entropy (input, target):
    return  -(target * F.log_softmax (input, dim = 1)).sum(1).mean(0)

class FocalCrossEntropy():
    def __init__(self,gamma):
        self.gamma=gamma

    def __call__(self,pred,target):
        if pred.shape!=target.shape:
            targets = torch.zeros_like(pred)
            targets[torch.arange(target.shape[0]),target]=1
        else:
            targets=target
        return -(torch.pow(1-F.softmax(pred, dim = 1),self.gamma) * targets * F.log_softmax (pred, dim = 1)).sum(1).mean(0)

class ExtraModel(nn.Module):
    def __init__(self,model,last_layer,extras,mid_linear,mlps=[],extra_activation=nn.ReLU(),dropout=0,bn=False,
                 patient_emdb=None,return_features=False):
        super(ExtraModel, self).__init__()
        self.base_model=copy.deepcopy(model)
        self.return_features=return_features
        last = self.base_model._modules[last_layer]
        in_last=last.in_features
        out_last=last.out_features
        bias_last=last.bias
        added_ins=0
        for i,l in enumerate(extras):
            if isinstance(l,(list,tuple)) and len(l)==2:
                self.add_module(f'extra_layers{i}',nn.Embedding(l[0], l[1]))
                added_ins+=l[1]
            elif isinstance(l,int):
                if l>1:
                    self.add_module(f'extra_layers{i}',nn.Sequential(NoopAddDim(),nn.Linear(1, l)))
                    added_ins+=l
                else:
                    self.add_module(f'extra_layers{i}',NoopAddDim())
                    added_ins+=1
            else:
                raise ValueError(f'extras {i} is {l} which is not an interger or a list/tuple of size 2')
        self.extra_linear=nn.Linear(added_ins,mid_linear)
        self.extra_bn=nn.BatchNorm1d(mid_linear) if bn else Noop()
        self.extra_dropout = nn.Dropout(dropout) if dropout>0 else Noop()
        self.patient_embd=patient_emdb if patient_emdb is None else nn.Embedding.from_pretrained(patient_emdb)
        em_size=0 if self.patient_embd is None else patient_emdb.shape[1]
        if len(mlps)==0:
            self.last_linear=nn.Linear(mid_linear+in_last+em_size,out_last,bias=bias_last is not None)
        else:
            nmlps=[mid_linear+in_last+em_size]+mlps
            sq = [x for s in [[nn.Linear(nmlps[i],nmlps[i+1]),
                               extra_activation,
                               nn.BatchNorm1d(nmlps[i+1]) if bn else Noop(),
                               nn.Dropout(dropout) if dropout>0 else Noop()] for i in range(len(mlps))] for x in s]
            self.add_module('mlps',nn.Sequential(*tuple(sq)))
            self.last_linear=nn.Linear(mlps[-1],out_last,bias=bias_last is not None)
        self.extra_activation=extra_activation
        self.base_model._modules[last_layer]=Noop()


    def forward(self,x,*extra):
        x = self.base_model(x)
        if self.patient_embd is not None:
            x=torch.cat([x,self.patient_embd(extra[0])],1)
            extra=extra[1:]
        extra=torch.cat([self.extra_activation(getattr(self,f'extra_layers{i}')(extra[i])) for i in range(len(extra))],1)
        extra=self.extra_activation(self.extra_linear(extra))
        x=torch.cat([x,extra],1)
        if hasattr(self,'mlps'):
            x=self.mlps(x)
        out = self.last_linear(x)
        return (out,x) if self.return_features else out

import math
def calc_positional_encoder(d_model, max_seq_len = 32):
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        return pe/(d_model**0.5)

def build_FCN(layers,activation=nn.ReLU):
    sq=OrderedDict()
    for i in range(len(layers)-2):
        sq[f'linear_{i}']=nn.Linear(layers[i],layers[i+1])
        sq[f'activation_{i}']=activation()
    sq[f'linear_{len(layers)-2}']=nn.Linear(layers[-2],layers[-1])
    return nn.Sequential(sq)

MAX_INSTANCE=3000
class TransformerModel(nn.Module):
    def __init__(self,in_size,
                 dim_feedforward,
                 n_heads=4,
                 n_encoders=4,
                 num_outputs=13,
                 linear_embd=None,
                 embedings=None,
                 classifier_in=Ellipsis,
                 dropout=0.1,
                 use_position_enc=True,
                 max_seq_len=32):
        super(TransformerModel, self).__init__()
        self.in_size=in_size
        self.encoder_layer =nn.TransformerEncoderLayer(in_size,
                                                       n_heads,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout)
        self.encoder=nn.TransformerEncoder(self.encoder_layer, n_encoders)
        if linear_embd is not None:
            for key in linear_embd.keys():
                s = [linear_embd[key]] if isinstance(linear_embd[key],int) else linear_embd[key]
                m = build_FCN([1]+s+[in_size],activation=nn.ReLU)
                self.add_module(key,m)
        self.linear_embd = None if linear_embd is None else list(linear_embd.keys())
        if embedings is not None:
            for key in embedings.keys():
                if isinstance(embedings[key],int):
                    self.add_module(key,nn.Embedding(embedings[key],in_size))
                else:
                    self.add_module(key,nn.Embedding.from_pretrained(embedings[key]))
        self.embedings = None if embedings is None else list(embedings.keys())
        self.pos_embd=calc_positional_encoder(in_size,max_seq_len) if use_position_enc else None
        self.classifier_in=classifier_in
        self.classifier = nn.Linear(in_size, num_outputs)



    def forward(self, x, *inputs, mask=None):
        img=x
        if self.pos_embd is not None:
            if self.pos_embd.device!=x.device:
                self.pos_embd = self.pos_embd.to(x.device)
        if self.linear_embd is not None:
            for i,key in enumerate(self.linear_embd):
                x=x+getattr(self,key)(inputs[i].unsqueeze(-1))
            n=len(self.linear_embd)
        else:
            n=0
        if self.embedings is not None:
            for i,key in enumerate(self.embedings):
                x=x+getattr(self,key)(inputs[i+n])
        x = x if self.pos_embd is None else x + self.pos_embd[:x.shape[1]][None]
        x = self.encoder(x.permute(1,0,-1),src_key_padding_mask=mask)
        x = x.permute(1,0,-1)
        out = self.classifier(x[:,self.classifier_in])
        return out



def get_transformer_model(in_size=256,
                          dim_feedforward=1024,
                          n_heads=4,
                          n_encoders=4,
                          num_outputs=14,
                          classifier_in=Ellipsis,
                          linear_embd=OrderedDict([('slice',[16,16])]),
                          embedings=None,
                          position_from_value=MAX_INSTANCE,
                          use_position_enc=False,
                          max_seq_len=320):

    if position_from_value>0:
        embedings = OrderedDict() if embedings is None else embedings
        embedings['pe_from_value'] = calc_positional_encoder(in_size,position_from_value)

    return TransformerModel(in_size=in_size,
                            dim_feedforward=dim_feedforward,
                            n_heads=n_heads,
                            n_encoders=n_encoders,
                            num_outputs=num_outputs,
                            linear_embd=linear_embd,
                            embedings=embedings,
                            classifier_in=classifier_in,
                            use_position_enc=use_position_enc,
                            max_seq_len=max_seq_len)


# From RSNA
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class Window(nn.Module):
    def forward(self, x):
        return torch.clamp(x,0,1)

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features,weights=None):
        super(ArcMarginProduct, self).__init__()
        if weights is None:
            self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
            self.reset_parameters()
        else:
            self.weight = nn.Parameter(weights)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
#        self.k.data=torch.ones(1,dtype=torch.float)

    def forward(self, features):
        cosine = F.linear(l2_norm(features), l2_norm(self.weight))
        return cosine

class ArcClassifier(nn.Module):
    def __init__(self,in_features, out_features,weights=None):
        super(ArcClassifier, self).__init__()
        self.classifier = ArcMarginProduct(in_features, out_features,weights=weights)
        self.dropout1=nn.Dropout(p=0.5, inplace=True)

    def forward(self, x,eq):
        out = self.dropout1(x-eq)
        out = self.classifier(out)
        return out

    def no_grad(self):
        for param in self.parameters():
            param.requires_grad=False

    def do_grad(self):
        for param in self.parameters():
            param.requires_grad=True


WSO=((-600,1500),(100,700),(40,400))

class SplitCT(nn.Module):
    def __init__(self,wso=WSO,do_bn=False):
        super(SplitCT, self).__init__()
        self.wso = wso
        if self.wso is not None:
            self.conv = nn.Conv2d(1,3, kernel_size=(1, 1))
            self.conv.weight.data.copy_(torch.tensor([[[[1./wso[0][1]]]],[[[1./wso[1][1]]]],[[[1./wso[2][1]]]]],dtype=torch.float32))
            self.conv.bias.data.copy_(torch.tensor([0.5 - wso[0][0]/wso[0][1],
                                                    0.5 - wso[1][0]/wso[1][1],
                                                    0.5 -wso[2][0]/wso[2][1]],dtype=torch.float32))
            self.sigmoid=nn.Sigmoid()
            self.norm = nn.BatchNorm2d(3) if do_bn else nn.InstanceNorm2d(3)

    def forward(self,x):
        x=torch.clamp(x,-2047,2047)
        if self.wso is not None:
            x = self.conv(x)
            x = self.sigmoid(x)
            x = self.norm(x)
        else:
            x =  x.repeat((1,3,1,1))
        return x

from collections import OrderedDict
import torch.nn as nn
import torchvision.models as models
import pretrainedmodels

import geffnet
try:
    from torch.cuda.amp import autocast
except:
    autocast=None
class AutocastModule(nn.Module):
    def __init__(self,module,do_autocast=True):
        super(AutocastModule, self).__init__()
        self.module=module
        self.do_autocast=do_autocast
    def forward(self,x):
        if autocast is None:
            x=x=self.module(x)
        else:
            with autocast(self.do_autocast):
                x=self.module(x)
        return [xx.to(torch.float32) for xx in x] if isinstance(x,list)\
               else tuple([xx.to(torch.float32) for xx in x]) if isinstance(x,tuple)\
               else x.to(torch.float32)

def get_model(model_name,output_size,pretrained=True, feature_size =256,pool=False,dropout=0.1,amp=False):
    if model_name.startswith('resne'):
        m=getattr(models,model_name)
        model=m(pretrained=pretrained)
        last_num=model.fc.in_features
        last = 'fc'
    elif model_name.startswith('se'):
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet' if pretrained else None)
        model.dropout = None
        last_num=model.last_linear.in_features
        last='last_linear'
    elif model_name.startswith('densenet'):
        m=getattr(models,model_name)
        model=m(pretrained=pretrained)
        last_num=model.classifier.in_features
        last='classifier'
    elif model_name.startswith('my_densenet'):
        model=models.DenseNet(32,block_config=(6, 12,32),num_init_features=64,num_classes=output_size)
    elif model_name in ['efficientnet_b0','efficientnet_b1','efficientnet_b2','efficientnet_b3'] or model_name.startswith('tf_'):
        model=geffnet.create_model( model_name, pretrained=pretrained)
        last_num=model.classifier.in_features
        last = 'classifier'
    else:
        raise ValueError('no model named '+model_name)
    if not pool:
        setattr(model,last,nn.Linear(last_num,feature_size))
    else:
        setattr(model,last,nn.Sequential(NoopAddDim(1),nn.AdaptiveMaxPool1d(feature_size),NoopSqueezeDim(1)))
    if amp:
        model = AutocastModule(model)
    sq=OrderedDict([('wso',SplitCT()),
                    ('base_model',model),
                    ('drop_out',nn.Dropout(dropout)),
                    ('last_linear',nn.Linear(feature_size,output_size))])
    return nn.Sequential(sq)

