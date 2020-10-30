import pandas as pd
import numpy as np
from tqdm import notebook
from misclib import *
import torch
import PIL.Image
import pydicom
import cv2
from torchvision import transforms
import torchvision.transforms.functional as TF
from collections import defaultdict
import pydicom
import io
import zipfile

from torch.utils.data import Dataset
class DatasetCat(Dataset):
    '''
    Concatenate datasets for Pytorch dataloader
    The normal pytorch implementation does it only for raws. this is a "column" implementation
    Arges:
        datasets: list of datasets, of the same length
    Updated: Yuval 12/10/2019
    '''

    def __init__(self,datasets):
        '''
        Args: datasets - an iterable containing the datasets
        '''
        super(DatasetCat, self).__init__()
        self.datasets=datasets
        assert len(self.datasets)>0
        for dataset in datasets:
            assert len(self.datasets[0])==len(dataset),"Datasets length should be equal"

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        outputs = tuple(dataset.__getitem__(idx) for i in self.datasets for dataset in (i if isinstance(i, tuple) else (i,)))
        return tuple(output for i in outputs for output in (i if isinstance(i, tuple) else (i,)))



def create_folds(df,num_folds,SEED,split_col='SeriesInstanceUID',errors=lambda x: x.pxl_min.isna()):
    df = df.reset_index(drop=True)
    df=df if errors is None else df[~errors(df)]
    items=np.sort(df[split_col].unique())
    np.random.seed(SEED)
    np.random.shuffle(items)
    nu=int(np.ceil(items.shape[0]/num_folds))
    items_val=[set(items[i*nu:(i+1)*nu]) for i in range(num_folds)]
    val_folds=[np.sort(df[df[split_col].isin(items_val[i])].index.values) for i in range(num_folds)]
    train_folds=[np.sort(np.setdiff1d(df.index.values,val_folds[i])) for i in range(num_folds)]
    return val_folds, train_folds, [np.array(list(s)) for s in items_val]



from PIL import  ImageDraw
import torch

def randint(low,high):
    return torch.randint(low,high,(1,))[0]

def randfloat(low,high):
    return torch.rand((1,))[0]*(high-low)+low

class CutoutTransform():
    def __init__(self,p=0.5,size=0.1,fill=None):
        self.p=p
        self.size = (size,size) if isinstance(size,float) else size
        self.fill = fill
    def __call__(self,img):
        if torch.rand((1,))<self.p:
            s0,s1 = int(img.shape[-2]*randfloat(0,self.size[0])),int(img.shape[-1]*randfloat(0,self.size[1]))
            sx=torch.randint(0,img.shape[-2]-s0,(1,))
            sy=torch.randint(0,img.shape[-1]-s1,(1,))
            img[...,sx:sx+s0,sy:sy+s1]=img.min() if self.fill is None else self.fill
        return img

from skimage import transform as sktransform

def np_tensor_transform(img,transform,*args,**kwarg):
    npa = isinstance(img, np.ndarray)
    img = img if npa else img.numpy() if len(img.shape)==2 else img.permute(1,2,0).numpy()
    img = transform(img,*args,**kwarg)
    img = img if npa else torch.tensor(img) if len(img.shape)==2 else torch.tensor(img).permute(2,1,0)
    return img

def pad_cut(a,plen,dim=0,value=0):
    if plen>0:
        r=tuple([plen//a.shape[dim]+1]+[1]*(len(a.shape)-1))
        return np.concatenate([a,np.moveaxis(np.tile(np.moveaxis(np.ones_like(a),0,dim),r)[:plen],0,dim)*value],dim)
    elif plen<0:
        return np.moveaxis(np.moveaxis(a,0,dim)[:a.shape[dim]+plen],0,dim)
    else:
        return a

def simple_resize(img,shape,pad_value=0):
    shape= (shape,shape) if isinstance(shape,int) else shape
    return pad_cut(pad_cut(img,shape[0]-img.shape[0],0,-1000),shape[1]-img.shape[1],1,pad_value)

class SimpleResizeTransform():
    def __init__(self,*args,**kwargs):
        self.args=args
        self.kwargs=kwargs
    def __call__(self,img):
        return np_tensor_transform(img,simple_resize,*self.args,**self.kwargs)

def resize(img,shape,anti_aliasing=True):
    shape= (shape,shape) if isinstance(shape,int) else shape
    return np_tensor_transform(img,sktransform.resize,shape,anti_aliasing=anti_aliasing)

class ResizeTransform():
    def __init__(self,*args,**kwargs):
        self.args=args
        self.kwargs=kwargs
    def __call__(self,img):
        return resize(img,*self.args,**self.kwargs)

def _crop(img,x,y,width,height,const=None):
    const = const if const is not None else img.min()
    d = len(img.shape)
    img = img[:,:,None] if d==2 else img
    if width>img.shape[1]:
        img=np.concatenate([np.ones((img.shape[0],(width-img.shape[1])//2+1,img.shape[-1]))*const,
                            img,
                            np.ones((img.shape[0],(width-img.shape[1])//2+1,img.shape[-1]))*const],1)
    if height>img.shape[0]:
        img=np.concatenate([np.ones(((height-img.shape[0])//2+1,img.shape[1],img.shape[-1]))*const,
                            img,
                            np.ones(((height-img.shape[0])//2+1,img.shape[1],img.shape[-1]))*const],0)

    return img[x:x+width,y:y+height] if d==3 else img[x:x+width,y:y+height,0]

def _center_crop(img,width,height,const=None):
    return _crop(img,max(0,(img.shape[0]-width)//2),max(0,(img.shape[1]-height)//2),width,height,const)

def crop (img,x,y,shape,const=None):
    width,height = (shape,shape) if isinstance(shape,int) else shape
    return np_tensor_transform(img,_crop,x,y,const)

class CropTransform():
    def __init__(self,*args,**kwargs):
        self.args=args
        self.kwargs=kwargs
    def __call__(self,img):
        return crop(img,*self.args,**self.kwargs)

def center_crop (img,shape,const=None):
    width,height = (shape,shape) if isinstance(shape,int) else shape
    return np_tensor_transform(img,_center_crop,width,height,const)

class CenterCropTransform():
    def __init__(self,*args,**kwargs):
        self.args=args
        self.kwargs=kwargs
    def __call__(self,img):
        return center_crop(img,*self.args,**self.kwargs)

def random_resized_crop(img, scale=(0.9, 1.1), ratio=(0.75, 1.3333333333333333)):
    shape= img.shape[:2] if isinstance(img, np.ndarray) else img.shape[-2:]
    s = randfloat(*scale)
    r = randfloat(*ratio) if ratio is not None else 1
    return center_crop(resize(img,(int(s*r*shape[0]),int(s/r*shape[1]))),shape)

class RandomResizedCropTransform():
    def __init__(self,*args,**kwargs):
        self.args=args
        self.kwargs=kwargs
    def __call__(self,img):
        return random_resized_crop(img,*self.args,**self.kwargs)

def rotate(img,angle,resize=False):
    return np_tensor_transform(img,sktransform.rotate , angle, resize=resize, center=None, order=1,
                                mode='constant', cval=img.min(), clip=True, preserve_range=False)

class RotateTransform():
    def __init__(self,*args,**kwargs):
        self.args=args
        self.kwargs=kwargs
    def __call__(self,img):
        return rotate(img,*self.args,**self.kwargs)

def random_rotate(img,angle,resize=False):
    return rotate(img,randfloat(-angle,angle),resize=False)

class RandomRotateTransform():
    def __init__(self,*args,**kwargs):
        self.args=args
        self.kwargs=kwargs
    def __call__(self,img):
        return random_rotate(img,*self.args,**self.kwargs)

def _flip(img,axis):
    return np.flip(img,axis).copy()

def flip(img,axis=1):
    return np_tensor_transform(img,_flip,axis=axis)

def hflip(img):
    return flip(img)

def vflip(img):
    return flip(img,axis=0)

def random_flip(img,h=0,v=0):
    img = img if randfloat(0,1)>v else vflip(img)
    img = img if randfloat(0,1)>h else hflip(img)
    return img

class RandomFlipTransform():
    def __init__(self,*args,**kwargs):
        self.args=args
        self.kwargs=kwargs
    def __call__(self,img):
        return random_flip(img,*self.args,**self.kwargs)

def random_change_mean_std(img,mean,std):
    s=randfloat(-std,std)
    s = 1+s if s>=0 else 1/(1-s)
    img = img*s + randfloat(-mean,mean)
    return img

class RandomChangeMeanStdTransform():
    def __init__(self,*args,**kwargs):
        self.args=args
        self.kwargs=kwargs
    def __call__(self,img):
        return random_change_mean_std(img,*self.args,**self.kwargs)


from multiprocessing import Lock
def find_cog(ar,p=0.15):
    c=(ar.max()-ar.min())*p+ar.min()
    return (ar.mean(),ar.std()) if (ar>c).sum()==0 else (ar[ar>c].mean(),ar[ar>c].std())

# def read_image(image_path,images_file):
#     with images_file.open(image_path) as zf:
#             img_dicom=pydicom.read_file(io.BytesIO(zf.read()))
#     img = img_dicom.pixel_array.astype(np.float)
#     return img+float(img_dicom.RescaleIntercept)

class Singleton(type):
    _instances = {}
    __singleton_lock = Lock()
    def __call__(cls, *args, **kwargs):
        with cls.__singleton_lock:
            if cls not in cls._instances:
                cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# class ImageReader(object, metaclass=Singleton):
#     def __init__(self,filepath):
#         self.file_handler = zipfile.ZipFile(filepath, mode = 'r', allowZip64 = True)
#         self.lock = Lock()
#         self.k=0
#     def __call__(self,filename):
#         with self.lock:
#             with self.file_handler.open(filename) as zf:
#                 img_dicom=pydicom.read_file(io.BytesIO(zf.read()))
#                 img = img_dicom.pixel_array.astype(np.float)+float(img_dicom.RescaleIntercept)
#             return img

class ImageReader():
    def __init__(self,filepath,image_type='pkl',return_pos=False):
        self.filepath = filepath
        assert image_type in ['pkl','dicom'], f"image type must be 'pkl' or 'dicom' and not {image_type} "
        self.image_type=image_type
        self.return_pos=return_pos
    def __call__(self,filename):
        if self.image_type=='pkl':
            with gzip.open(self.filepath+filename,'rb') as zf:
                img=pickle.load(zf).astype(np.float64)
        elif self.image_type=='dicom':
            try:
                img_dicom = pydicom.read_file(self.filepath+filename)
                img = img_dicom.pixel_array.astype(np.float)+float(img_dicom.RescaleIntercept)
                pos = img_dicom.InstanceNumber
            except Exception as e:
                print (filename,e)
                img = np.zeros((512,512),dtype=np.float)
        return (img , pos,) if self.return_pos else img




class ImageDataset(Dataset):

    def __init__(self,image_reader,df,transform=ResizeTransform((512,512)),file_ext='.pkl',return_simple=None):
        super(ImageDataset, self).__init__()
        self.image_reader=image_reader
        self.df=df
        self.names=(df.StudyInstanceUID+'/'+df.SeriesInstanceUID+'/'+df.SOPInstanceUID+file_ext).values
        self.calc_pos = ('instance_number' not in df.columns) and (file_ext=='.dcm') and (self.image_reader.return_pos)
        if self.calc_pos:
            self.pos = np.zeros(df.shape[0],dtype=np.long)
        self.return_true='pe_present_on_image' in df.columns
        if self.return_true:
            self.pe_present_on_image = df.pe_present_on_image.values
            self.negative_exam_for_pe=df.negative_exam_for_pe.values
            self.qa_motion=df.qa_motion.values
            self.qa_contrast = df.qa_contrast.values
            self.flow_artifact = df.flow_artifact.values
            self.rv_lv_ratio_gte_1 = df.rv_lv_ratio_gte_1.values*df.pe_present_on_image.values
            self.rv_lv_ratio_lt_1 = df.rv_lv_ratio_lt_1.values*df.pe_present_on_image.values
            self.leftsided_pe = df.leftsided_pe.values*df.pe_present_on_image.values
            self.chronic_pe = df.chronic_pe.values*df.pe_present_on_image.values
            self.true_filling_defect_not_pe = df.true_filling_defect_not_pe.values
            self.rightsided_pe  = df.rightsided_pe.values*df.pe_present_on_image.values
            self.acute_and_chronic_pe = df.acute_and_chronic_pe.values*df.pe_present_on_image.values
            self.central_pe = df.central_pe.values*df.pe_present_on_image.values
            self.indeterminate = df.indeterminate.values*df.pe_present_on_image.values
#         self.rel_slice=df.rel_slice.values
        self.transform=transform
        self.basic_transform=return_simple


    def __len__(self):
        return len(self.names)


    def __getitem__(self,idx):
        image=self.image_reader(self.names[idx])
        if self.calc_pos:
            img=image[0]
            self.pos[idx]=image[1]
        else:
            img=image
        img = torch.tensor(img)[None]
        out = (self.transform(img).to(dtype=torch.float32),)
        out = out if self.basic_transform is None else out +(self.basic_transform(img).to(dtype=torch.float32),)
        if self.return_true:
            out = out + (torch.tensor([  self.pe_present_on_image[idx],
                                self.qa_motion[idx],
                                self.qa_contrast[idx],
                                self.flow_artifact[idx],
                                self.rv_lv_ratio_gte_1[idx],
                                self.rv_lv_ratio_lt_1[idx],
                                self.leftsided_pe[idx],
                                self.chronic_pe[idx],
                                self.rightsided_pe[idx],
                                self.acute_and_chronic_pe[idx],
                                self.central_pe[idx]],dtype=torch.float32),)

        return  out






def fair_split(a,max_len):
    k=int(np.ceil(len(a)/max_len))
    return [a[i::k] for i in range(k)]

def pad(a,plen,dim=0,value=0):
    r=tuple([plen//a.shape[dim]+1]+[1]*(len(a.shape)-1))
    return torch.cat([a,torch.ones_like(a).transpose(0,dim).repeat(r)[:plen].transpose(0,dim)*value],dim)

MAX_INSTANCE=3000
class PatientFeaturesDataset(Dataset):

    def __init__(self,features,df,series_ids,max_len=250,rand_split=True,rep=1,fnoise=0.):
        super(PatientFeaturesDataset, self).__init__()
        self.df=df
        self.features=features
        self.max_len=max_len
        self.series_ids=series_ids
        self.rand_split=rand_split
        self.subset = df.SeriesInstanceUID.isin(self.series_ids).values
        self.rep=rep
        self.fnoise=fnoise
        self.rel_slice=torch.tensor(df.rel_slice.values,dtype=torch.float32)
        self.instance_number=torch.tensor(np.clip(df.instance_number.values,0,MAX_INSTANCE-1),dtype=torch.long)
        self.reset()
        self.return_true='pe_present_on_image' in df.columns
        if self.return_true:
            self.pe_present_on_image = torch.tensor(df.pe_present_on_image.values,dtype=torch.float32)
            self.series_values=torch.stack([torch.tensor(df.true_filling_defect_not_pe,dtype=torch.float32),
                                          torch.tensor(df.qa_motion,dtype=torch.float32),
                                          torch.tensor(df.qa_contrast,dtype=torch.float32),
                                          torch.tensor(df.flow_artifact,dtype=torch.float32),
                                          torch.tensor(df.rv_lv_ratio_gte_1,dtype=torch.float32),
                                          torch.tensor(df.rv_lv_ratio_lt_1,dtype=torch.float32),
                                          torch.tensor(df.leftsided_pe,dtype=torch.float32),
                                          torch.tensor(df.chronic_pe,dtype=torch.float32),
                                          torch.tensor(df.negative_exam_for_pe,dtype=torch.float32),
                                          torch.tensor(df.rightsided_pe,dtype=torch.float32),
                                          torch.tensor(df.acute_and_chronic_pe,dtype=torch.float32),
                                          torch.tensor(df.central_pe,dtype=torch.float32),
                                          torch.tensor(df.indeterminate,dtype=torch.float32)],1)


    def __len__(self):
        return len(self.idx_list)

    def reset(self):
        gp=self.df[self.subset].groupby('SeriesInstanceUID')
        self.idx_list=[]
        for j in range(self.rep):
            for g in gp.groups.items():
                idxs=g[1].values
                if self.rand_split:
                    idxs=idxs[torch.randperm(len(idxs))]
                self.idx_list.extend(fair_split(idxs,self.max_len-1))



    def __getitem__(self,idx):
        idxs=self.idx_list[idx]
        idxs=idxs[np.argsort(self.instance_number[idxs])]
        a= torch.randint(0,self.features.shape[0],(len(idxs),))
        plen=len(idxs)
        out = (pad(self.features[a,idxs]*(1+self.fnoise*torch.randn_like(self.features[a,idxs])),self.max_len-plen,dim=0,value=0),\
               pad(self.rel_slice[idxs],self.max_len-plen,dim=0,value=-1),\
               pad(self.instance_number[idxs],self.max_len-plen,dim=0,value=0),)
        if self.return_true:
            out = out + (pad(self.pe_present_on_image[idxs],self.max_len-plen,dim=0,value=-1),self.series_values[idxs[0]],)
        out=out+(pad(torch.tensor(idxs,dtype=torch.long),self.max_len-plen,dim=0,value=-1),)
        return out


