import scipy.io
import numpy
import os
#import random


from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt





root_eegmat="./feature_extract/data/eegmat/"
root_stew="./feature_extract/data/stew/"
root_nback="./feature_extract/data/nback/"

def openfile(file_name):

    #读取
    
    read_dict=scipy.io.loadmat(file_name)

    raw_data=read_dict['data']
    raw_data=raw_data[:,:7680]
    #print(raw_data.shape)
    #raw_data=numpy.transpose(raw_data,axes=(0,2,1))
    #print(raw_data.shape)
    return raw_data

def slices(data):                                       #interval=2.3,5-2,5-3


    #切片
    newdata=[]
    for i in range(5):
        newdata.append(data[:,i*1280:1280*i+2560])
    
    #print(1280*i+2560)
    newdata=numpy.stack(newdata)
    
    return newdata
    

def normalization(data):

    std=StandardScaler()
    data=std.fit_transform(data)
    
    return data
    


###########################################
#总样本5，训练5，验证1，测试4
    
class Datas:
    def __init__(self,root,tag="train"):

        self.datas=[]
        self.labels=[]
        self.classes=[]
        
        list_dirs=os.listdir(root)
        sumclass=int(len(list_dirs)/2)
        if tag=="train":
            nums=5
            idxs=0
        elif tag=="validate":
            nums=1
            idxs=0
        elif tag=="test":
            nums=4
            idxs=1
            
        for sub_path in list_dirs:
                
            samples=slices(openfile(root+sub_path))
            
            if (sub_path[-5]=='0'):
                self.labels.append(numpy.zeros(nums,dtype=numpy.int64))
            elif (sub_path[-5]=='1'):
                self.labels.append(numpy.ones(nums,dtype=numpy.int64))

            self.classes.append(numpy.ones(nums,dtype=numpy.int64)*int(sub_path[:-6]))
                    
            self.datas.append(samples[idxs:idxs+nums])
              
        self.datas=numpy.stack(self.datas)
        #print(self.datas.shape)
        #breakpoint()
        #self.datas=self.datas.reshape(sumclass,-1,self.datas.shape[2],self.datas.shape[3])
        self.datas=self.datas.reshape(-1,self.datas.shape[-2],self.datas.shape[-1])
        self.labels=numpy.stack(self.labels)
        #self.labels=self.labels.reshape(sumclass,-1)
        self.labels=self.labels.reshape(-1)
        self.classes=numpy.stack(self.classes)
        self.classes=self.classes.reshape(-1)
        
        
        #standard
        #self.datas=self.datas.reshape(self.datas.shape[2])
        #self.datas=min_max(self.datas)
        #self.datas=self.datas.reshape(-1,14*9)
        #self.datas=normalization(self.datas)
        #self.datas=self.datas.reshape(sumclass,2,-1,14,9)
        
        self.shape=self.get_shape()
        #print(self.shape)
    def __getitem__(self,idx):
        return self.datas[idx],self.labels[idx],self.classes[idx]

    def __len__(self):
        return len(self.datas)

    def get_shape(self):
        return {"data":self.datas.shape,"label":self.labels.shape,"class":self.classes.shape}

    def get_data(self):
        return list(zip(self.datas,self.labels,self.classes))            

eegmat=Datas(root_eegmat,"train")
stew=Datas(root_stew,"train")
nback=Datas(root_nback,"train")

eegmat_test=Datas(root_eegmat,"test")
stew_test=Datas(root_stew,"test")
nback_test=Datas(root_nback,"test")

eegmat_val=Datas(root_eegmat,"validate")
stew_val=Datas(root_stew,"validate")
nback_val=Datas(root_nback,"validate")


if __name__=='__main__':
    None
    breakpoint()
    print(eegmat[0])
    print(len(eegmat))
    print(eegmat.shape)
    a,b,c=zip(*eegmat.get_data())
    print(len(a),len(b))
    
    



    
