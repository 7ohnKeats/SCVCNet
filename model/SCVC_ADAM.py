import math
import torch
import torchsummary


torch.manual_seed(42)


def batch_cross_product(x,y,C,div=False):
    
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
        
    
    
class SCVC(torch.nn.Module):

    ###      kernel size:(A:2,B:2,C:1)
    
    def __init__(self,
                 in1_channels:int,
                 in2_channels:int,
                 out_channels:int,
                 kernel_size:int=3,
                 bias:bool=True,
                 kernel_share:bool=False,
                 groups:int=1,
                 div:bool=False
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
        if groups !=1:
            layers=[]
            for i in range(groups):
                if kernel_share is True:
                    layers.append(SCVC(in1_channels=1,in2_channels=1,out_channels=self.out_channels//groups,
                                               bias=bias,groups=1,kernel_share=True))
                else:
                    layers.append(SCVC(in1_channels=self.in1_channels//groups,in2_channels=self.in2_channels//groups,out_channels=self.out_channels//groups,
                                               bias=bias,groups=1))
            self.layers=torch.nn.ModuleList(layers)
        else:

            if kernel_share is True:
                self.kernel_C=torch.nn.Parameter(torch.empty(out_channels,1))
                torch.nn.init.kaiming_uniform_(self.kernel_C,a=math.sqrt(5))

                if kernel_size>1:
                    conv_A=[]
                    conv_B=[]
                    for j in range(kernel_size):
                        if (j+1)%2==0:
                            conv_A.append(torch.nn.Conv1d(in_channels=1,out_channels=out_channels,\
                                                          kernel_size=2,stride=1,padding=(j+1)//2,dilation=j+1,bias=False,padding_mode='zeros'))
                            conv_B.append(torch.nn.Conv1d(in_channels=1,out_channels=out_channels,\
                                                          kernel_size=2,stride=1,padding=(j+1)//2,dilation=j+1,bias=False,padding_mode='zeros'))        
                    self.conv_A=torch.nn.ModuleList(conv_A)
                    self.conv_B=torch.nn.ModuleList(conv_B)
            else:
                self.kernel_C=torch.nn.Parameter(torch.empty(out_channels,in1_channels*in2_channels))
                torch.nn.init.kaiming_uniform_(self.kernel_C,a=math.sqrt(5))
            
                if kernel_size>1:
                    conv_A=[]
                    conv_B=[]
                    for j in range(kernel_size):
                        if (j+1)%2==0:
                            conv_A.append(torch.nn.Conv1d(in_channels=in1_channels*in2_channels,out_channels=out_channels,\
                                                          kernel_size=2,stride=1,padding=(j+1)//2,dilation=j+1,bias=False,padding_mode='zeros'))
                            conv_B.append(torch.nn.Conv1d(in_channels=in1_channels*in2_channels,out_channels=out_channels,\
                                                          kernel_size=2,stride=1,padding=(j+1)//2,dilation=j+1,bias=False,padding_mode='zeros'))        
                    self.conv_A=torch.nn.ModuleList(conv_A)
                    self.conv_B=torch.nn.ModuleList(conv_B)
            
            if bias:
                self.bias=torch.nn.Parameter(torch.empty(out_channels))
                bound=1/math.sqrt(in1_channels)
                torch.nn.init.uniform_(self.bias,-bound,bound)
            else:
                self.register_parameter('bias',None)
        
    
    def forward(self,x:torch.Tensor,y:torch.Tensor):
        dx=x.size(-1)
        dy=y.size(-1)
        
        if self.groups !=1:
            
            out=[]
            for i in range(self.groups):
                out.append(self.layers[i](x[:,(self.in1_channels//self.groups)*i:(self.in1_channels//self.groups)*(i+1),:],\
                                               y[:,(self.in2_channels//self.groups)*i:(self.in2_channels//self.groups)*(i+1),:]))
            output=torch.cat(out,dim=1)

        else:
            if self.kernel_share is True:
                output=batch_cross_product(x,y,self.kernel_C,self.div)
                if self.kernel_size>1:
                    for i in range(len(self.conv_A)):
                        
                        x=torch.nn.functional.pad(input=x,pad=[(self.kernel_size-1)//2,(self.kernel_size-1)//2])
                        x=x.reshape(x.size(0),1,-1)
                        a=self.conv_A[i](x)
                        a=a.reshape(a.size(0),a.size(1),self.in1_channels,-1)
                        _,a,_=torch.split(a,[(self.kernel_size-1)//2,dx,(self.kernel_size-1)//2],dim=-1)
                        a=torch.sum(a,dim=-2)*self.in2_channels
                        
                        y=torch.nn.functional.pad(input=y,pad=[(self.kernel_size-1)//2,(self.kernel_size-1)//2])
                        y=y.reshape(y.size(0),1,-1)
                        b=self.conv_B[i](y)
                        b=b.reshape(b.size(0),b.size(1),self.in2_channels,-1)
                        _,b,_=torch.split(b,[(self.kernel_size-1)//2,dy,(self.kernel_size-1)//2],dim=-1)
                        b=torch.sum(b,dim=-2)*self.in1_channels
                        
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

if __name__=="__main__":

    
    x=torch.tensor([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]],dtype=torch.float)
    y=torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],dtype=torch.float)
    
    x=x.reshape(2,2,-1)
    y=y.reshape(2,3,-1)
    x=torch.randn(2,2,6)
    y=torch.randn(2,4,4)
    blc=SCVC(in1_channels=2,in2_channels=4,out_channels=6,kernel_size=3,groups=1,kernel_share=False,div=True)
    
    print(blc(x,y).shape)
    for m in blc.named_parameters():
        print(m)



    
