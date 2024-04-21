import numpy
import torch
import matplotlib.pyplot as plt
from get_data_feat import eegmat,stew,nback,eegmat_test,stew_test,nback_test,eegmat_val,stew_val,nback_val
from model.SCVCNet import SCVCNet

numpy.random.seed(42)
torch.manual_seed(42)

chs=['F3','F7','T7','P7','O1','O2','P8','T8','F8','F4']

class Main:
    def __init__(self,train=None,valid=None,test=None):

        self.train=train
        self.valid=valid
        self.test=test

    def select_data(self,data):
        d=[]
        l=[]
        for i,j,_ in data:
            #print(i.shape)
            d.append(i)
            l.append(j)
                    
        d=numpy.stack(d)
        l=numpy.stack(l)
        d=torch.tensor(d)
        l=torch.tensor(l)
        
        return d,l

    def split(self,d):
        x=d[:,:,:16]  #16
        y=d[:,:,16:]

        x=x.permute(0,2,1)
        y=y.permute(0,2,1)
        
        return x,y
    
    def clf(self,d1,t1,d2,t2,d3,t3):#d:data,t:label,1:train,2:validation,3:test
        x1,y1=self.split(d1) #split theta and alpha feature
        x2,y2=self.split(d2)
        x3,y3=self.split(d3)
        
        #model
        scvcnet=SCVCNet(in1_channels=16,in2_channels=16,out_channels=76,outputs=1,kernel_size=3,\
                           groups=1,kernel_share=False,div=False,function="sigmoid",bias=True,rcond=None,\
                           reduce_dim="avg",norm=1.5e-10,random_distribution="kaiming_uniform"

)                                                   
        scvcnet.train(x1,y1,t1)                        #train model
        acc_train,f1_train=scvcnet.score(x1,y1,t1)    #test on train set
        acc_val,f1_val=scvcnet.score(x2,y2,t2)        #test on val set
        acc_test,f1_test=scvcnet.score(x3,y3,t3)      #test on test set
        print("train(acc,f1):",acc_train,f1_train)
        print("val(acc,f1):",acc_val,f1_val)
        print("test(acc,f1):",acc_test,f1_test)
        
        return
    
    def forward(self):
        # data extract
        d1,t1=self.select_data(self.train)
        d2,t2=self.select_data(self.valid)
        d3,t3=self.select_data(self.test)
        
        self.clf(d1,t1,d2,t2,d3,t3)
        return

        
    
    
if __name__=='__main__':
    
    main=Main(train=nback,valid=stew_val,test=eegmat)
    main.forward()
    print()
   
    
        
    
    
