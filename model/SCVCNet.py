import math
import torch
import torchsummary
import numpy
from typing import Optional, List, Tuple, Union
from sklearn.metrics import accuracy_score,f1_score
#import matplotlib.pyplot as plt

#torch.manual_seed(42)
#numpy.random.seed(42)
#chs=['F3','F7','T7','P7','O1','O2','P8','T8','F8','F4']



### cross product by batch and shape review
def batch_cross_product(x:torch.Tensor,y:torch.Tensor,C:torch.Tensor,div=False):
    ###     x,y:shape(batch,channels,feat_lenth)
    dx=x.size(-1)
    dy=y.size(-1)
    
    x=x.reshape(x.size(0),-1)
    y=y.reshape(y.size(0),-1)
    if div is True:
        y=1/y
    result=torch.einsum('ni,nj->nij',[x,y])
    result=torch.split(result,dy,dim=-1)
    result=torch.stack(result)
    result=torch.split(result,dx,dim=-2)
    result=torch.stack(result)
    result=torch.flatten(result,start_dim=0,end_dim=1)
    
    result=result.permute(1,0,2,3)
    result=torch.einsum('ncij,oc->noij',[result,C])
    
    return result
        
###param initial by standard_norm or kaiming_uniform
def param_init(W:torch.Tensor,method:str,param:str,in_chs:int=None):
    if method == "standard_norm":
        return torch.nn.init.normal_(W,mean=0.0,std=1.0)
    if method == "kaiming_uniform" and param == "weight":
        return torch.nn.init.kaiming_uniform_(W,a=math.sqrt(5))
    if method == "kaiming_uniform" and param == "bias":
        bound=1/math.sqrt(in_chs)
        return torch.nn.init.uniform_(W,-bound,bound)

###perform 1D_Conv by random param     
class RandomConv1d(torch.nn.Module):

    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int,
                 padding:int,
                 dilation:int,
                 function:str,
                 bias:bool=False,
                 precision:torch.dtype=torch.float64,
                 random_distribution:str="standard_norm",
                 ):
        super(RandomConv1d,self).__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.function=function
        self.precision=precision
        self.random_distribution=random_distribution
        if function == "linear":
            self.weight=torch.ones(out_channels,in_channels,kernel_size)
        else:
            self.weight=torch.empty(out_channels,in_channels,kernel_size)
            self.weight=param_init(W=self.weight,method=random_distribution,param="weight")
            self.weight*=3.0/self.in_channels**0.5
        
        if bias is True and function != "linear":
            self.bias=torch.empty(out_channels)
            self.bias=param_init(W=self.bias,method=random_distribution,param="bias",in_chs=in_channels)
            
            self.bias=self.bias.to(precision)   
        else:
            self.register_parameter('bias',None)
        self.weight=self.weight.to(precision)
        
    def forward(self,x:torch.Tensor):
        h=torch.conv1d(input=x,weight=self.weight,bias=self.bias,stride=self.stride,padding=self.padding,dilation=self.dilation,groups=1)
        return h
        
        
###SCVC operation    
class SCVC(torch.nn.Module):

    def __init__(self,
                 in1_channels:int,
                 in2_channels:int,
                 out_channels:int,
                 function:str,
                 kernel_size:int=3,
                 bias:bool=True,
                 kernel_share:bool=False,
                 groups:int=1,
                 div:bool=False,
                 random_distribution:str="standard_norm",
                 precision:torch.dtype=torch.float64,
                 ):
        super(SCVC,self).__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in1_channels % groups != 0:
            raise ValueError('in1_channels must be divisible by groups')
        if in2_channels % groups != 0:
            raise ValueError('in2_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if kernel_size % 2 != 1:
            raise ValueError('kenel_size must be a odd')
        self.in1_channels = in1_channels
        self.in2_channels = in2_channels
        self.out_channels = out_channels
        self.kernel_size=kernel_size
        self.groups=groups
        self.kernel_share=kernel_share
        self.div=div
        self.function=function
        self.random_distribution=random_distribution
        self.precision=precision
        if groups !=1:
            layers=[]
            for i in range(groups):
                if kernel_share is True:
                    layers.append(SCVC(in1_channels=1,in2_channels=1,out_channels=self.out_channels//groups,
                                               bias=bias,groups=1,kernel_size=kernel_size,kernel_share=True,div=self.div,function=function,
                                               precision=precision,random_distribution=random_distribution))
                else:
                    layers.append(SCVC(in1_channels=self.in1_channels//groups,in2_channels=self.in2_channels//groups,
                                               out_channels=self.out_channels//groups,bias=bias,kernel_size=kernel_size,groups=1,div=self.div,
                                               function=function,precision=precision,random_distribution=random_distribution))
            self.layers=torch.nn.ModuleList(layers)
        else:

            if kernel_share is True:
                if function == "linear":
                    self.kernel_C=torch.ones(out_channels,1)
                else:
                    self.kernel_C=torch.empty(out_channels,1)
                    self.kernel_C=param_init(W=self.kernel_C,method=random_distribution,param="weight")
                    self.kernel_C*=3.0/(1)**0.5
                        
                if kernel_size>1:
                    conv_A=[]
                    conv_B=[]
                    for j in range(kernel_size):
                        if (j+1)%2==0:
                            conv_A.append(RandomConv1d(in_channels=1,out_channels=out_channels,\
                                                       kernel_size=2,stride=1,padding=(j+1)//2,dilation=j+1,bias=False,\
                                                       function=function,precision=precision,random_distribution=random_distribution))
                            conv_B.append(RandomConv1d(in_channels=1,out_channels=out_channels,\
                                                       kernel_size=2,stride=1,padding=(j+1)//2,dilation=j+1,bias=False,\
                                                       function=function,precision=precision,random_distribution=random_distribution))        
                    self.conv_A=torch.nn.ModuleList(conv_A)
                    self.conv_B=torch.nn.ModuleList(conv_B)
            else:

                if function == "linear":
                    self.kernel_C=torch.ones(out_channels,in1_channels*in2_channels)
                else:
                    self.kernel_C=torch.empty(out_channels,in1_channels*in2_channels)
                    self.kernel_C=param_init(W=self.kernel_C,method=random_distribution,param="weight")
                    self.kernel_C*=3.0/(in1_channels*in2_channels)**0.5
                        
                if kernel_size>1:
                    conv_A=[]
                    conv_B=[]
                    for j in range(kernel_size):
                        if (j+1)%2==0:
                            conv_A.append(RandomConv1d(in_channels=in1_channels*in2_channels,out_channels=out_channels,\
                                                       kernel_size=2,stride=1,padding=(j+1)//2,dilation=j+1,bias=False,\
                                                       function=function,precision=precision,random_distribution=random_distribution))
                            conv_B.append(RandomConv1d(in_channels=in1_channels*in2_channels,out_channels=out_channels,\
                                                       kernel_size=2,stride=1,padding=(j+1)//2,dilation=j+1,bias=False,\
                                                       function=function,precision=precision,random_distribution=random_distribution))        
                    self.conv_A=torch.nn.ModuleList(conv_A)
                    self.conv_B=torch.nn.ModuleList(conv_B)
            
            if bias and function != "linear":
                self.bias=torch.empty(out_channels)
                self.bias=param_init(W=self.bias,method=random_distribution,param="bias",in_chs=in1_channels)
                self.bias=self.bias.to(precision)
            else:
                self.register_parameter('bias',None)
            self.kernel_C=self.kernel_C.to(precision)
            
    def forward(self,x:torch.Tensor,y:torch.Tensor):
        dx=x.size(-1)
        dy=y.size(-1)
        
        if self.groups !=1:
            
            out=[]
            for i in range(self.groups):
                
                out_layer=self.layers[i](x[:,(self.in1_channels//self.groups)*i:(self.in1_channels//self.groups)*(i+1),:],\
                                               y[:,(self.in2_channels//self.groups)*i:(self.in2_channels//self.groups)*(i+1),:])
                out.append(out_layer)
            output=torch.cat(out,dim=1)

        else:
            if self.kernel_share is True:
                
                output=batch_cross_product(x,y,self.kernel_C,self.div)
                if self.kernel_size>1:
                    for i in range(len(self.conv_A)):
                        x_=torch.nn.functional.pad(input=x,pad=[(self.kernel_size-1)//2,(self.kernel_size-1)//2])
                        x_=x_.reshape(x_.size(0),1,-1)
                        a=self.conv_A[i](x_)
                        a=a.reshape(a.size(0),a.size(1),x.size(1),-1)
                        _,a,_=torch.split(a,[(self.kernel_size-1)//2,dx,(self.kernel_size-1)//2],dim=-1)
                        a=torch.sum(a,dim=-2)*y.size(1)
                        
                        y_=torch.nn.functional.pad(input=y,pad=[(self.kernel_size-1)//2,(self.kernel_size-1)//2])
                        y_=y_.reshape(y_.size(0),1,-1)
                        b=self.conv_B[i](y_)
                        b=b.reshape(b.size(0),b.size(1),y.size(1),-1)
                        _,b,_=torch.split(b,[(self.kernel_size-1)//2,dy,(self.kernel_size-1)//2],dim=-1)
                        b=torch.sum(b,dim=-2)*x.size(1)
                        
                        a=a.unsqueeze(3).expand(a.size(0),a.size(1),a.size(2),b.size(2))
                        b=b.unsqueeze(2).expand(b.size(0),b.size(1),a.size(2),b.size(2))
                        output=output+a+b
                        

                
            else:  
                output=batch_cross_product(x,y,self.kernel_C,self.div)
                if self.kernel_size>1:
                    x_A=x.repeat(1,self.in2_channels,1)
                    y_B=y.repeat(1,self.in1_channels,1)
                    for i in range(len(self.conv_A)):
                        a=self.conv_A[i](x_A)
                        b=self.conv_B[i](y_B)
                        a=a.unsqueeze(3).expand(a.size(0),a.size(1),a.size(2),b.size(2))
                        b=b.unsqueeze(2).expand(b.size(0),b.size(1),a.size(2),b.size(2))
                        output=output+a+b
                        
            if self.bias!=None:
                output=output+self.bias.view(1,-1,1,1)
        return output

###SCVCNet
class SCVCNet(object):
    def __init__(self,
                 in1_channels:int,
                 in2_channels:int,
                 out_channels:int,
                 outputs:int,
                 function:str,
                 rcond:float=None,
                 norm:float=None,
                 kernel_size:int=3,
                 bias:bool=True,
                 kernel_share:bool=False,
                 groups:int=1,
                 div:bool=False,
                 batch:int=1000,
                 precision:torch.dtype=torch.float64,
                 reduce_dim:str="avg",
                 random_distribution:str="standard_norm",
                 ):
        
        self.func_list = ("linear", "sigmoid", "tanh")
        self.reduce_list = ("max","avg","maxpool","avgpool","glbavg","glbmax")

        
        self.func={}
        self.func["sigmoid"] = lambda H: 1 / (1 + torch.exp(H))
        self.func["tanh"] = lambda H: torch.tanh(H)
        self.func["linear"] = lambda H: H
        
        
        if norm is None:
            norm = 50*torch.finfo(precision).eps
            
        assert batch>0,"batch should be positive"

        if function not in self.func_list:
            raise ValueError("function must be one of {}, but got function='{}'".format(self.func_list, function))

        if reduce_dim not in self.reduce_list:
            raise ValueError("reduce_dim must be one of {}, but got reduce_dim='{}'".format(self.reduce_list, reduce_dim))

        
        self.in1_channels = in1_channels
        self.in2_channels = in2_channels
        self.out_channels = out_channels
        self.outputs=outputs
        self.kernel_size=kernel_size
        self.groups=groups
        self.kernel_share=kernel_share
        self.div=div
        self.function=function
        self.norm=norm
        self.batch=batch
        self.precision=precision
        self.rcond=rcond
        self.reduce_dim=reduce_dim
        self.random_distribution=random_distribution
        self.HtH=None
        self.HHt=None
        self.HT=None
        self.Beta=None
        self.H=None
        self.T=None
        self.N=None
        self.L=None
        
        self.scvc=SCVC(in1_channels=in1_channels,in2_channels=in2_channels,out_channels=out_channels,kernel_size=kernel_size,\
                               groups=groups,bias=bias,kernel_share=kernel_share,div=div,function=function,precision=precision,random_distribution=random_distribution)

    ###SCVC operation, activation and reduce dim.
    def project(self,X:torch.Tensor,Y:torch.Tensor):
        X=X.to(self.precision)
        Y=Y.to(self.precision)
        
        H=self.scvc(X,Y)
        
        H=self.func[self.function](H)

        if self.reduce_dim=="avg":
            H=H.mean(dim=1)
        elif self.reduce_dim=="max":
            H,_=H.max(dim=1)
        #elif self.reduce_dim=="maxpool":
        #    k_s=H.shape[-1]//(int(numpy.ceil(float(H.shape[-1])/float(math.sqrt(self.out_channels)))))
        #    H=torch.nn.functional.max_pool2d(H,kernel_size=k_s,stride=k_s)
        #elif self.reduce_dim=="avgpool":
        #    k_s=H.shape[-1]//(int(numpy.ceil(float(H.shape[-1])/float(math.sqrt(self.out_channels)))))
        #    H=torch.nn.functional.avg_pool2d(H,kernel_size=k_s,stride=k_s)
        elif self.reduce_dim=="glbmax":
            H=torch.nn.functional.max_pool2d(H,kernel_size=H.shape[-1],stride=H.shape[-1])
        elif self.reduce_dim=="glbavg":
            H=torch.nn.functional.avg_pool2d(H,kernel_size=H.shape[-1],stride=H.shape[-1])

        H=H.reshape(H.size(0),-1)
        
        return H

    ### predict test label
    def predict(self,X:torch.Tensor,Y:torch.Tensor):
        assert self.Beta is not None, "Solve the task before predicting"
        H = self.project(X,Y)

        Y = numpy.dot(H, self.Beta)
        return Y
    
    def get_Beta(self):
        return self.Beta

    def get_single_value(self):
        return numpy.linalg.svd(self.HH,full_matrices=False,compute_uv=False)

    def add_data(self,X:torch.Tensor,Y:torch.Tensor,T:torch.Tensor):

        nb = int(numpy.ceil(float(X.shape[0]) / self.batch))
        
        for X0,Y0,T0 in zip(torch.tensor_split(X, nb, dim=0),
                            torch.tensor_split(Y, nb, dim=0),
                            torch.tensor_split(T, nb, dim=0)):
            
            self.add_batch(X0,Y0,T0)

    ###solve by batch 
    def add_batch(self,X:torch.Tensor,Y:torch.Tensor,T:torch.Tensor):
        
        H=self.project(X,Y)
        
        self.N=N=len(H)
        self.L=L=H.shape[-1]
        H=H.numpy()
        T=T.numpy()
        T=T.reshape(len(T),self.outputs)
        self.H=H
        self.T=T
        
        if self.HtH is None and N > L:                                         # initialize space for self.HH, self.HT
            self.HtH = numpy.zeros((L, L),dtype=numpy.float64)
            self.HT = numpy.zeros((L, self.outputs),dtype=numpy.float64)
            numpy.fill_diagonal(self.HtH, self.norm)
        if self.HHt is None and N <= L:
            self.HHt = numpy.zeros((N, N),dtype=numpy.float64)
            numpy.fill_diagonal(self.HHt, self.norm)

        if N > L:
            self.HtH += numpy.dot(H.T, H)
            self.HT += numpy.dot(H.T, T)
        else:
            self.HHt += numpy.dot(H, H.T)

    ### solve Beta
    def solve(self):
        
        if self.N > self.L:
            if self.rcond is not None:
                HH_pinv = numpy.linalg.pinv(self.HtH,rcond=self.rcond)
            else:
                HH_pinv = numpy.linalg.pinv(self.HtH)
                
            self.Beta = numpy.dot(HH_pinv, self.HT)
 
        else:
            if self.rcond is not None:
                HH_pinv = numpy.linalg.pinv(self.HHt,rcond=self.rcond)
            else:
                HH_pinv = numpy.linalg.pinv(self.HHt)
                
            self.Beta = numpy.dot(numpy.dot(self.H.T, HH_pinv), self.T)

    ### train        
    def train(self,X:torch.Tensor,Y:torch.Tensor,T:torch.Tensor):
        
        self.reset()
        T=T.to(self.precision)
        self.add_data(X,Y,T)
        self.solve()
        
    def reset(self):
        self.HH=None
        self.HT=None
        self.Beta=None
    
    def prob2label(self,prob,label):
        bound=numpy.mean(label)
        out=numpy.ones(prob.shape)*label[0]
        for idx in range(len(prob)):
            if prob[idx]>bound:
                out[idx]=label[1]
    
        return out
    ###compute f1 and acc score on test
    def score(self,X,Y,T,label:Tuple[int,...]=[0,1]):
        
        T=T.to(self.precision)
        assert len(label)==2, "only support 2 class"
        P=self.predict(X,Y)
        L=self.prob2label(P,label)
        
        acc=accuracy_score(y_true=T,y_pred=L)
        f1=f1_score(y_true=T,y_pred=L,average="macro")
        
        return acc,f1#,Hr
    ### return loss
    def error(self,T,P):   #T: real label
        
        T=T.to(self.precision)
        return numpy.mean((P - T)**2)
    def set_params(self,**param_dict):
        if param_dict['out_channels'] is not None:
            self.out_channels=param_dict['out_channels']
        if param_dict['function'] is not None:
            self.function=param_dict['function']
        if param_dict['norm'] is not None:
            self.norm=param_dict['norm']
        if param_dict['kernel_size'] is not None:
            self.kernel_size=param_dict['kernel_size']
        if param_dict['share_div'] is not None:
            self.kernel_share=param_dict['share_div'][0]
            self.div=param_dict['share_div'][1]
        if param_dict['groups'] is not None:
            self.groups=param_dict['groups']
        if param_dict['reduce_dim'] is not None:
            self.reduce_dim=param_dict['reduce_dim']
        
        self.scvc=SCVC(in1_channels=self.in1_channels,in2_channels=self.in2_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,\
                               groups=self.groups,kernel_share=self.kernel_share,div=self.div,function=self.function,precision=self.precision)
        #print(self.function)
        self.reset()
        return
if __name__=="__main__":

    
    #x=torch.tensor([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,11,6,7,8,9,10],[1,2,10,4,5,6,7,8,9,10]],dtype=torch.float)
    #y=torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],dtype=torch.float)
    
    #x=x.reshape(2,2,-1)
    #y=y.reshape(2,4,-1)
    x=torch.randn(6,10,16)
    y=torch.randn(6,10,16)
    t=torch.hstack([torch.ones(3,dtype=torch.long),torch.zeros(3,dtype=torch.long)])
    
    scvc=SCVCNet(in1_channels=10,in2_channels=10,out_channels=100,outputs=1,kernel_size=3,groups=1,kernel_share=False,div=True,function="tanh")
    
    scvc.train(x,y,t)
    print(scvc.score(x,y,t,label=[0,1]))
    #for m in blc.named_parameters():
    #    print(m)



    
