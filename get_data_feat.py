import scipy.io
import numpy
import os
#import random


from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt





root_eegmat="./feature/eegmat/"
root_stew="./feature/stew/"
root_nback="./feature/nback/"

def openfile(file_name):

    #读取
    
    read_dict=scipy.io.loadmat(file_name)

    raw_data=read_dict['data']
    
    return raw_data


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
                
            samples=openfile(root+sub_path)
            
            if (sub_path[-5]=='0'):
                self.labels.append(numpy.zeros(nums,dtype=numpy.int64))
            elif (sub_path[-5]=='1'):
                self.labels.append(numpy.ones(nums,dtype=numpy.int64))

            self.classes.append(numpy.ones(nums,dtype=numpy.int64)*int(sub_path[:-6]))
                    
            self.datas.append(samples[idxs:idxs+nums])
              
        self.datas=numpy.stack(self.datas)
        self.datas=self.datas.reshape(-1,self.datas.shape[-2],self.datas.shape[-1])
        
        self.labels=numpy.stack(self.labels)
        self.labels=self.labels.reshape(-1)
        
        self.classes=numpy.stack(self.classes)
        self.classes=self.classes.reshape(-1)
        
        
        self.shape=self.get_shape()

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
    
    



    
