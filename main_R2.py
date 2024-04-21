import torch
import numpy

from sklearn.metrics import accuracy_score,f1_score
from model.SCVC_ADAM import SCVC

from get_data_feat_norm import eegmat,stew,nback,eegmat_test,stew_test,nback_test,eegmat_val,stew_val,nback_val

torch.manual_seed(42)
numpy.random.seed(42)

Eegmat=torch.utils.data.DataLoader(eegmat,batch_size=10,shuffle=True)
#eegmat_val=torch.utils.data.DataLoader(eegmat_val,batch_size=10,shuffle=True)
#eegmat_test=torch.utils.data.DataLoader(eegmat_test,batch_size=10,shuffle=True)
Stew=torch.utils.data.DataLoader(stew,batch_size=10,shuffle=True)
#stew_val=torch.utils.data.DataLoader(stew_val,batch_size=10,shuffle=True)
#stew_test=torch.utils.data.DataLoader(stew_test,batch_size=10,shuffle=True)
Nback=torch.utils.data.DataLoader(nback,batch_size=10,shuffle=True)
#nback_val=torch.utils.data.DataLoader(nback_val,batch_size=10,shuffle=True)
#nback_test=torch.utils.data.DataLoader(nback_test,batch_size=10,shuffle=True)



class SCVCNet(torch.nn.Module):
        def __init__(self,reduce_dim,func):
                super(SCVCNet,self).__init__()
                self.reduce_dim=reduce_dim
                self.func=func
                self.bc=SCVC(in1_channels=16,in2_channels=16,out_channels=88,kernel_size=3,groups=1,kernel_share=False,div=False,bias=True)
                
                self.fc=torch.nn.Linear(in_features=100,out_features=2,bias=False)
        def forward(self,x,y):
                out=self.bc(x,y)
                if self.func=="sigmoid":
                        out=torch.nn.functional.sigmoid(out)
                elif self.func=="tanh":
                        out=torch.nn.functional.tanh(out)
                
                if self.reduce_dim=="avg":
                        out=torch.mean(out,dim=1)
                elif self.reduce_dim=="max":
                        out,_=torch.max(out,dim=1)
                out=torch.flatten(out,start_dim=1)
                out=self.fc(out)
                return out
                
                
def select_data(data):
        d=[]
        l=[]
        c=[]
        for i,j,k in data:
            d.append(i)
            l.append(j)
            c.append(k)
            
        d=numpy.stack(d)
        l=numpy.stack(l)
        c=numpy.stack(c)

        sub=set(c.tolist())
        num_sub=len(sub)
        num_sample=len(c)
        sps=num_sample//num_sub

        dd=d.reshape(num_sub,sps,d.shape[-2],d.shape[-1])
        ll=l.reshape(num_sub,sps)
        cc=c.reshape(num_sub,sps)
        xx=dd[:,:,:,:16]
        yy=dd[:,:,:,16:]
        xx=xx.transpose(0,1,3,2)
        yy=yy.transpose(0,1,3,2)
        return xx,yy,ll,cc
    

model=SCVCNet(reduce_dim="avg",func="sigmoid")
crt=torch.nn.CrossEntropyLoss()
optim=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.00001)


for epoch in range(100):
    model.train()
    losses=[]
    for idx,(d,l,_) in enumerate(Eegmat):
        optim.zero_grad()
        d=d.to(dtype=torch.float)
        x=d[:,:,:16]
        y=d[:,:,16:]
        x=x.permute(0,2,1)
        y=y.permute(0,2,1)
        p=model(x,y)
        loss=crt(p,l)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    losses=sum(losses)
    print(epoch,losses)
        
print()
accs=[]
f1s=[]
model.eval()
with torch.no_grad():
    a,b,c,d=select_data(stew_test)
    for x,y,l,_ in zip(a,b,c,d):
        x=torch.tensor(x,dtype=torch.float)
        y=torch.tensor(y,dtype=torch.float)
        p=model(x,y)
        pp=p.argmax(axis=-1)
        acc=accuracy_score(l,pp)
        f1=f1_score(l,pp,average="macro")
        print(acc,'\t',f1)
        accs.append(acc)
        f1s.append(f1)
print(numpy.mean(accs),'\t',numpy.mean(f1),end="")


print()
accs=[]
f1s=[]
model.eval()
with torch.no_grad():
    a,b,c,d=select_data(nback)
    for x,y,l,_ in zip(a,b,c,d):
        x=torch.tensor(x,dtype=torch.float)
        y=torch.tensor(y,dtype=torch.float)
        p=model(x,y)
        pp=p.argmax(axis=-1)
        acc=accuracy_score(l,pp)
        f1=f1_score(l,pp,average="macro")
        print(acc,'\t',f1)
        accs.append(acc)
        f1s.append(f1)
print(numpy.mean(accs),'\t',numpy.mean(f1),end="")
        



