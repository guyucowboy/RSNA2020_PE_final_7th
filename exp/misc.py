
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/misc.ipynb

import json
class Parameters():
    def __init__(self,**kargs):
        for key in kargs.keys():
            setattr(self,key,kargs[key])
    def __call__(self,param,value=None):
        if value is not None:
            setattr(self,param,value)
        return self.__getattribute__(param)

def add_attr_rec(C,att_dict):
    for key in att_dict.keys():
        if isinstance(att_dict[key],dict):
            setattr(C,key,Parameters())
            add_attr_rec(C.__dict__[key],att_dict[key])
        else:
            setattr(C,key,att_dict[key])

def json_to_parameters(config_file):
    with open(config_file) as json_data_file:
        data = json.load(json_data_file)
    params=Parameters()
    add_attr_rec(params,data)
    return params



def file_log(file_name,config_file=None,**kargs):
    fname=file_name[:file_name.rfind('.')]+'.json'
    if config_file is None:
        log_dict={**{'ref_file_name':file_name},**kargs}
    else:
        with open(config_file) as json_data_file:
            log_dict={**{'ref_file_name':file_name},**json.load(json_data_file),**kargs}
    with open(fname,'wt') as f:
        json.dump(log_dict, f, indent=4, sort_keys=True)
        f.write('\n')

from torch.utils.data import Dataset
import torch
import smtplib
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

def device_by_name(name):
    ''' Return reference to cuda device by using Part of it's name

        Args:
            name: part of the cuda device name (shuuld be distinct

        Return:
            Reference to cuda device

        Updated: Yuval 12/10/19
    '''
    assert torch.cuda.is_available(),"No cuda device"
    device=None
    for i in range(torch.cuda.device_count()):
        dv=torch.device("cuda:{}".format(i))
        if name in torch.cuda.get_device_name(dv):
            device=dv
            break
    assert device, "device {} not found".format(name)
    return device

def get_model_device(model):
    if not torch.cuda.is_available():
        return torch.device('cpu')
    else:
        device_num = next(model.parameters()).get_device()
        if device_num<0:
            return torch.device('cpu')
        else:
            return torch.device("cuda:{}".format(device_num))


class Email_Progress():
    ''' class  - Email progress to myself

        Args:
            source email    : Gmail user name, don't need the @gmail.com (don't use your mail account - risky)
            source_password : Gmail Password (RiSK!!!!!!!!!!!)
            target_email    : The recipt email
            title           : Email's Title

       Methods:
           __call__
           Args:
               history - directory with the data to send

       Update: Yuval 12/10/19
    '''

    def __init__(self,source_email,source_password,target_email,title):
        self.source_email=source_email
        self.source_password=source_password
        self.target_email=target_email
        self.title=title

    def __call__(self,history):
        str_list=['Subject:{}\n'.format(self.title)]+[str(d)[1:-2].replace("'",'').replace(':','')+'\n' for d in history]
        email_text=''.join(str_list)
        try:
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.ehlo()
            server.login(self.source_email, self.source_password)
            server.sendmail(self.source_email, self.target_email, email_text)
            server.close()
            return 0
        except Exception as e:
            print(e)
            print ('Something went wrong...')
            return e



import urllib.request
def check_internet(host='http://google.com'):
    try:
        urllib.request.urlopen(host) #Python 3.x
        return True
    except:
        return False

from xml.dom import minidom
from collections import Mapping

def dict2element(root,structure,doc):
    """
    Gets a dictionary like structure and converts its
    content into xml elements. After that appends
    resulted elements to root element. If root element
    is a string object creates a new elements with the
    given string and use that element as root.
    This function returns a xml element object.
    """
    assert isinstance(structure,Mapping), 'Structure must be a mapping object such as dict'

    # if root is a string make it a element
    if isinstance(root,str):
        root = doc.createElement(root)

    for key,value in structure.items():
        el = doc.createElement(str(key))
        if isinstance(value,Mapping):
            dict2element(el,value,doc)
        else:
            el.appendChild(doc.createTextNode(str(value) if value is not None  else ''))
        root.appendChild(el)

    return root


def dict2xml(structure,tostring=False):
    """
    Gets a dict like object as a structure and returns a corresponding minidom
    document object.
    If str is needed instead of minidom, tostring parameter can be used

    Restrictions:
    Structure must only have one root.
    Structure must consist of str or dict objects (other types will
    converted into string)
    Sample structure object would be
    {'root':{'elementwithtextnode':'text content','innerelements':{'innerinnerelements':'inner element content'}}}
    result for this structure would be
    '<?xml version="1.0" ?>
    <root>
      <innerelements><innerinnerelements>inner element content</innerinnerelements></innerelements>
      <elementwithtextnode>text content</elementwithtextnode>
    </root>'
    """
    # This is main function call. which will return a document
    assert len(structure) == 1, 'Structure must have only one root element'
    assert isinstance(structure,Mapping), 'Structure must be a mapping object such as dict'

    root_element_name, value = next(iter(structure.items()))
    impl = minidom.getDOMImplementation()
    doc = impl.createDocument(None,str(root_element_name),None)
    dict2element(doc.documentElement,value,doc)
    return doc.toxml() if tostring else doc

import glob
import pandas as pd
import pydicom
from tqdm import notebook
import os
import pdb

import numpy as np

def use_only(d,key_list):
    return {key:d[key] for key in d.keys() if key in key_list}


def dicom_get_data_dict(img):
    img_data = {}
    for i in img.iterall():
        if i.name == "Pixel Data":
            continue
        name = i.name.replace(" ", "_").replace("(", "").replace(")", "").replace('-','_').replace('/','_').lower()
        img_data[name] = i.value
    return img_data

def dicom_get_list_data(imgs,image_stats=False,errored_files=None,key_list=None):
    list_data = []
    for k,i in enumerate(notebook.tqdm(imgs)):
        try:
            img = pydicom.read_file(i)
            img_data = dicom_get_data_dict(img)
            img_data['file_name']=i
            if image_stats:
                pic = img.pixel_array
                img_data['pxl_min'] = pic.min()
                img_data['pxl_max'] = pic.max()
                img_data['pxl_mean'] = pic.mean()
                img_data['pxl_std'] = pic.std()
            list_data.append(img_data if key_list is None else use_only(img_data,key_list))
            if (k+1)%10000==0:
                print(k)
        except BaseException as e:
            print (f"can't read file {i} with error {e}")
            img_data['file_name']=i
            list_data.append(img_data if key_list is None else use_only(img_data,key_list))
            if errored_files is not None:
                errored_files.append(i)
    return list_data

def dicom_get_df_data(imgs,image_stats=False,errored_files=None,read_images=True,key_list=None):
    list_data = dicom_get_list_data(imgs,image_stats=image_stats,errored_files=errored_files,key_list=key_list)
    out = pd.DataFrame(list_data)
    del list_data
    return out

import gzip
import pickle
def gzip_pickle(o,name):
    os.makedirs(os.path.dirname(name), exist_ok=True)
    with gzip.open(name, 'wb') as f:
        pickle.dump(o,f)
        del o